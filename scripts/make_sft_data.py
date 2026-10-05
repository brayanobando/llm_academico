"""
Paso 7 (lámina 8/8) - 1. Crear el dataset SFT (preguntas y respuestas en JSONL)
Arma pares {"pregunta", "respuesta"} desde tres fuentes:
    a) Wikipedia (data/raw/wikipedia): el primer párrafo de cada artículo responde "¿Qué es...?" / "¿Quién fue...?"
    b) La cronología de tu Word (data/raw/*.docx): líneas "1956 – ..." responden "¿Qué pasó en 1956?"
    c) Preguntas escritas a mano (data/sft/qa_manual.jsonl): conceptos y hechos clave

Uso:
    python scripts\\make_sft_data.py --out data\\sft\\qa.jsonl
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import leer_docx, limpiar_linea  # noqa: E402
from download_wikipedia import leer_titulos, nombre_archivo  # noqa: E402

# Títulos de la lista de Wikipedia que son personas (para preguntar "¿Quién fue...?")
PERSONAS = {
    "Ramon Llull", "Gottfried Leibniz", "Charles Babbage", "Ada Lovelace", "George Boole", "Kurt Gödel",
    "Alan Turing", "John von Neumann", "Claude Shannon", "Norbert Wiener", "Warren McCulloch", "Walter Pitts",
    "Grace Murray Hopper", "John McCarthy (científico)", "Marvin Minsky", "Herbert Alexander Simon",
    "Allen Newell", "Arthur L. Samuel", "Frank Rosenblatt", "Joseph Weizenbaum", "Terry Winograd",
    "Nils John Nilsson", "Seymour Papert", "Edward Feigenbaum", "Lotfi A. Zadeh", "Hans Moravec",
    "Rodney Brooks", "John Hopfield", "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Judea Pearl",
    "Lee Sedol", "Demis Hassabis", "Andrew Ng", "John Searle", "Raymond Kurzweil", "Nick Bostrom",
    "Isaac Asimov", "Karel Čapek",
}
ARTICULOS = {"El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Lo"}
VERBOS = {"es": "es", "son": "son", "fue": "fue", "fueron": "fueron", "era": "era"}
MAX_RESPUESTA = 480  # caracteres (~110 tokens): cabe de sobra en el contexto de 256


def quitar_parentesis(s):
    anterior = None
    while anterior != s:
        anterior = s
        s = re.sub(r"\s*\([^()]*\)", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def oraciones(texto):
    """Divide en oraciones sin cortar siglas como 'EE. UU.' o iniciales como 'A. Simon'."""
    texto = texto.replace("EE. UU.", "EE.UU.")
    partes = re.split(r"(?<=[a-záéíóúñü0-9\)»\"”])([.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡«])", texto)
    frases, actual = [], ""
    for p in partes:
        if p in (".", "!", "?"):
            frases.append((actual + p).strip())
            actual = ""
        else:
            actual += p
    if actual.strip():
        frases.append(actual.strip())
    return [f.replace("EE.UU.", "EE. UU.") for f in frases]


def recortar(texto, n_oraciones, maximo=MAX_RESPUESTA):
    salida = ""
    for f in oraciones(texto)[:n_oraciones]:
        if salida and len(salida) + len(f) + 1 > maximo:
            break
        salida = (salida + " " + f).strip()
    return salida


def primer_parrafo(ruta):
    """Primer párrafo de prosa de un artículo (salta títulos y líneas cortas)."""
    if not ruta.exists():
        return None, None
    lineas = [l.strip() for l in ruta.read_text(encoding="utf-8").splitlines()]
    parrafos = [limpiar_linea(l) for l in lineas if l and not l.startswith("=")]
    parrafos = [p for p in parrafos if len(p) >= 80]
    if not parrafos:
        return None, None
    return parrafos[0], (parrafos[1] if len(parrafos) > 1 else None)


def nombre_comun(titulo):
    """'Máquina analítica' -> 'máquina analítica'; 'Deep Blue', 'Prolog' o 'ELIZA' quedan igual."""
    nombre = re.sub(r"\s*\(.*?\)", "", titulo.split(":")[-1]).strip()  # 'Anexo:Historia...' -> 'Historia...'
    palabras = nombre.split()
    if len(palabras) >= 2 and palabras[1][:1].islower() and not palabras[0].isupper():
        return nombre[0].lower() + nombre[1:]
    return nombre


def sujeto_y_verbo(parrafo, titulo=None):
    """De 'La lógica difusa es una lógica...' saca ('la lógica difusa', 'es').
    Si no encuentra un sujeto que contenga el título, devuelve (None, None)."""
    primera = quitar_parentesis(oraciones(parrafo)[0])
    nucleo = re.sub(r"\s*\(.*?\)", "", titulo or "").strip()
    m = None
    if nucleo:  # 1) buscar "<artículo> <título> es/fue ..." dentro de la primera oración
        m = re.search(r"((?:\b(?:el|la|los|las|un|una|lo)\s+)?" + re.escape(nucleo) + r")[^.]{0,40}?\s(es|son|fue|fueron|era)\s",
                      primera, flags=re.IGNORECASE)
    if not m:   # 2) o el comienzo de la oración, si nombra lo mismo que el título
        m = re.search(r"^(.{2,70}?)(?:,.*?)?\s(es|son|fue|fueron|era)\s", primera)
        if m:
            sujeto_l = m.group(1).lower()
            claves = [w.lower()[:5] for w in nucleo.split() if len(w) > 3]
            if m.group(1).split()[0] in {"En", "Desde", "Durante", "Para", "Con", "A", "Según", "Dentro", "Tras"} \
                    or not all(c in sujeto_l for c in claves):
                m = None
    if not m:
        return None, None
    sujeto = re.split(r"[;:]| o ", m.group(1).strip(" ,"))[0].strip()
    palabras = sujeto.split()
    verbo = VERBOS[m.group(2).lower()]
    if palabras and palabras[0].capitalize() in ARTICULOS:
        sujeto = palabras[0].lower() + sujeto[len(palabras[0]):]
        if palabras[0].lower() in ("los", "las"):
            verbo = {"es": "son", "fue": "fueron", "era": "eran"}.get(verbo, verbo)
    return sujeto, verbo


def pares_wikipedia(titulos, carpeta):
    pares = []
    for titulo in titulos:
        lead, segundo = primer_parrafo(carpeta / nombre_archivo(titulo))
        if not lead:
            continue
        respuesta = recortar(quitar_parentesis(lead), 2)
        larga = recortar(quitar_parentesis(lead + " " + (segundo or "")), 4, maximo=600)
        nombre = re.sub(r"\s*\(.*?\)", "", titulo.split(":")[-1]).strip()
        if titulo in PERSONAS:
            m = re.search(r"\s(es|fue)\s", quitar_parentesis(oraciones(lead)[0]))
            verbo = m.group(1) if m else "fue"
            apellido = nombre.split()[-1]
            preguntas = [(f"¿Quién {verbo} {nombre}?", respuesta), (f"¿Quién {verbo} {apellido}?", respuesta),
                         (f"Háblame de {nombre}.", larga), (f"¿Qué sabes sobre {nombre}?", respuesta)]
        else:
            sujeto, verbo = sujeto_y_verbo(lead, titulo)
            if sujeto is None:
                sujeto, verbo = nombre_comun(titulo), "es"
            preguntas = [(f"¿Qué {verbo} {sujeto}?", respuesta), (f"Explícame qué {verbo} {sujeto}.", larga),
                         (f"¿Qué sabes sobre {sujeto}?", respuesta)]
        pares += [{"pregunta": q, "respuesta": a, "fuente": f"wikipedia:{titulo}"} for q, a in preguntas if a]
    return pares


def pares_cronologia(carpeta_raw):
    """Líneas tipo '1956 – Conferencia de Dartmouth: ...' y títulos tipo 'Primer invierno de la IA (1974–1980)'."""
    patron_hito = re.compile(r"^((?:\d{1,2} de )?(?:[A-Za-zé]+ de )?\d{4}(?:[–-]\d{4})?)\s+[–-]\s+(.{20,})$")
    patron_etapa = re.compile(r"^([A-ZÁÉÍÓÚ][^()]{5,80}?)\s*\((\d{4})[–-](\d{4}|actualidad)\)\s*$")
    pares, por_fecha = [], {}
    for docx in sorted(Path(carpeta_raw).glob("*.docx")):
        if docx.name.startswith("~$"):
            continue
        for linea in leer_docx(docx).splitlines():
            linea = limpiar_linea(linea)
            if m := patron_hito.match(linea):
                por_fecha.setdefault(m.group(1), []).append(m.group(2).rstrip("."))
            elif m := patron_etapa.match(linea):
                etapa, ini, fin = m.group(1).strip(), m.group(2), m.group(3)
                periodo = f"desde {ini} hasta hoy" if fin == "actualidad" else f"entre {ini} y {fin}"
                pares.append({"pregunta": f"¿Cuándo fue la etapa «{etapa}» de la historia de la IA?",
                              "respuesta": f"La etapa «{etapa}» de la historia de la IA va {periodo}.",
                              "fuente": f"cronologia:{docx.name}"})
    for fecha, hitos in por_fecha.items():
        fecha = re.sub(r"\b([A-Z][a-zé]+) de (\d{4})", lambda m: f"{m.group(1).lower()} de {m.group(2)}", fecha)
        en = "el" if fecha[0].isdigit() and " de " in fecha else "en"  # "el 30 de noviembre de 2022"
        texto = f"{en.capitalize()} {fecha}: {hitos[0]}."
        for h in hitos[1:]:
            texto += f" Además: {h}."
        for q in (f"¿Qué pasó {en} {fecha} en la historia de la IA?", f"¿Qué ocurrió {en} {fecha}?"):
            pares.append({"pregunta": q, "respuesta": recortar(texto, 4, maximo=600), "fuente": "cronologia"})
    return pares


def variantes_manuales(pares):
    """Para cada '¿Qué es X?' escrito a mano agrega 'Explícame qué es X.' (misma respuesta)."""
    extra = []
    for p in pares:
        m = re.match(r"^¿Qué (es|son|fue) (.+)\?$", p["pregunta"])
        if m:
            extra.append({**p, "pregunta": f"Explícame qué {m.group(1)} {m.group(2)}."})
    return extra


# ---------- Aumento de datos: otras formas de hacer la misma pregunta ----------
PARAFRASIS = [
    (re.compile(r"^¿Qué (?:pasó|ocurrió) (en|el) (.+?)(?: en la historia de la IA)?\?$"),
     lambda m: [f"¿Qué pasó {m[1]} {m[2]}?", f"¿Qué ocurrió {m[1]} {m[2]}?", f"¿Qué sucedió {m[1]} {m[2]}?",
                f"¿Qué pasó {m[1]} {m[2]} en la historia de la IA?", f"Cuéntame qué pasó {m[1]} {m[2]}.",
                f"¿Qué hito de la IA hubo {m[1]} {m[2]}?", f"¿Qué se logró {m[1]} {m[2]} en inteligencia artificial?"]),
    (re.compile(r"^¿Quién (fue|es) (.+)\?$"),
     lambda m: [f"¿Quién {m[1]} {m[2]}?", f"¿Quién era {m[2]}?", f"Dime quién {m[1]} {m[2]}.",
                f"¿Sabes quién {m[1]} {m[2]}?", f"Háblame de {m[2]}.", f"Cuéntame sobre {m[2]}.",
                f"¿Qué hizo {m[2]}?"]),
    (re.compile(r"^¿Qué (es|son|fue|fueron) (.+)\?$"),
     lambda m: [f"¿Qué {m[1]} {m[2]}?", f"Explícame qué {m[1]} {m[2]}.", f"Dime qué {m[1]} {m[2]}.",
                f"Define {m[2]}.", f"¿Me explicas qué {m[1]} {m[2]}?", f"Háblame de {m[2]}.",
                f"¿En qué consiste {m[2]}?"]),
]


def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def parafrasis(pregunta):
    for patron, generar in PARAFRASIS:
        m = patron.match(pregunta)
        if m:
            return [q for q in dict.fromkeys(generar(m)) if q != pregunta]
    return []


def aumentar(pares, n_parafrasis, seed):
    """Por cada par agrega n paráfrasis y una versión informal (minúsculas, sin tildes ni signos).
    Todas llevan la misma respuesta y se marcan con "variante": true."""
    rng = random.Random(seed)
    existentes = {p["pregunta"].lower() for p in pares}
    nuevos = []
    for par in pares:
        opciones = [q for q in parafrasis(par["pregunta"]) if q.lower() not in existentes]
        elegidas = rng.sample(opciones, min(n_parafrasis, len(opciones)))
        base = rng.choice([par["pregunta"]] + elegidas)
        informal = sin_tildes(base.strip("¿?¡!. ")).lower()
        for q in elegidas + [informal]:
            if q.lower() not in existentes:
                existentes.add(q.lower())
                nuevos.append({**par, "pregunta": q, "variante": True})
    return nuevos


def main():
    p = argparse.ArgumentParser(description="Crea el dataset SFT de preguntas y respuestas")
    p.add_argument("--titles", default="configs/wikipedia_titulos.txt")
    p.add_argument("--wiki", default="data/raw/wikipedia")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--manual", default="data/sft/qa_manual.jsonl")
    p.add_argument("--out", default="data/sft/qa.jsonl")
    p.add_argument("--parafrasis", type=int, default=2,
                   help="paráfrasis extra por pregunta (0 = sin aumento de datos)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wiki = pares_wikipedia(leer_titulos(args.titles), Path(args.wiki))
    crono = pares_cronologia(args.raw)
    manual = []
    if Path(args.manual).exists():
        manual = [json.loads(l) for l in Path(args.manual).read_text(encoding="utf-8").splitlines() if l.strip()]
        for m in manual:
            m.setdefault("fuente", "manual")
    manual += variantes_manuales(manual)

    vistos, todos = set(), []
    for par in manual + crono + wiki:  # si una pregunta se repite, gana la escrita a mano
        clave = par["pregunta"].lower().strip()
        if clave not in vistos and len(par["respuesta"]) >= 30:
            vistos.add(clave)
            todos.append(par)
    n_originales = len(todos)
    if args.parafrasis > 0:
        todos += aumentar(todos, args.parafrasis, args.seed)
    random.Random(args.seed).shuffle(todos)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for par in todos:
            f.write(json.dumps(par, ensure_ascii=False) + "\n")

    originales = [x for x in todos if not x.get("variante")]
    print("=== Dataset SFT creado ===")
    print(f"Desde Wikipedia   : {sum(x['fuente'].startswith('wikipedia') for x in originales)} pares")
    print(f"Desde la cronología: {sum(x['fuente'].startswith('cronologia') for x in originales)} pares")
    print(f"Escritos a mano   : {sum(x['fuente'] == 'manual' for x in originales)} pares")
    print(f"Variantes (aumento de datos): {len(todos) - n_originales} pares")
    print(f"TOTAL             : {len(todos)} pares  ->  {out}")
    print("\nEjemplos:")
    for par in todos[:6]:
        print(f"  P: {par['pregunta']}\n  R: {par['respuesta'][:200]}{'...' if len(par['respuesta']) > 200 else ''}\n")


if __name__ == "__main__":
    main()
