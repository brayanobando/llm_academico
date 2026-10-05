"""
Paso 3 (lámina 4/8) - 2. Limpiar el corpus
Lee todo lo que haya en data/raw (TXT, MD, JSONL y DOCX, también en subcarpetas),
normaliza a UTF-8, quita ruido y duplicados, y guarda:
    data/clean/corpus.txt          un documento por bloque, separados por una línea vacía
    data/clean/estadisticas.json   resumen de la limpieza

Uso:
    python scripts\\prepare_data.py --input data\\raw --out data\\clean
"""
import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# Secciones de Wikipedia que no son prosa útil para aprender a escribir
SECCIONES_EXCLUIDAS = {
    "referencias", "notas", "notas y referencias", "referencias y notas", "fuentes",
    "enlaces externos", "véase también", "bibliografía", "bibliografía adicional",
    "lecturas adicionales", "lectura adicional", "otras lecturas", "obras",
    "obras seleccionadas", "publicaciones", "publicaciones seleccionadas",
    "galería", "filmografía", "discografía", "premios", "premios y reconocimientos",
}
RE_TITULO_WIKI = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")
RE_CITAS = re.compile(r"\[(?:\d+|[a-z]|nota\s*\d+|n\.?\s*\d+|cita requerida|sin fuentes|"
                      r"aclaración requerida|página requerida|¿cuándo\?)\]", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_CITAS_IA = re.compile(r"【[^】]*】|\[\d+†[^\]]*\]")  # marcas de cita que dejan algunos asistentes de IA


# ---------- Lectura de archivos ----------

def leer_bytes_como_texto(datos):
    """Normalizar a UTF-8: intenta UTF-8 y, si falla, Windows-1252 (típico de archivos de Windows)."""
    try:
        return datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos.decode("cp1252", errors="replace")


def leer_docx(ruta):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(ruta) as z:
        raiz = ET.fromstring(z.read("word/document.xml"))
    parrafos = []
    for p in raiz.iter(f"{{{ns['w']}}}p"):
        partes = []
        for nodo in p.iter():
            etiqueta = nodo.tag.split("}")[-1]
            if etiqueta == "t" and nodo.text:
                partes.append(nodo.text)
            elif etiqueta in ("tab", "br"):
                partes.append(" ")
        parrafos.append("".join(partes))
    return "\n".join(parrafos)


def leer_jsonl(ruta):
    docs = []
    for linea in leer_bytes_como_texto(ruta.read_bytes()).splitlines():
        if not linea.strip():
            continue
        obj = json.loads(linea)
        texto = (obj.get("text") or obj.get("texto") or obj.get("content")) if isinstance(obj, dict) else None
        if texto:
            docs.append(texto)
    return docs


def leer_documentos(carpeta):
    """Devuelve una lista de (nombre, texto). Un JSONL aporta un documento por línea."""
    docs = []
    for ruta in sorted(Path(carpeta).rglob("*")):
        if not ruta.is_file() or ruta.name.startswith((".", "~$")):
            continue
        ext = ruta.suffix.lower()
        try:
            if ext in (".txt", ".md"):
                docs.append((ruta.name, leer_bytes_como_texto(ruta.read_bytes())))
            elif ext == ".docx":
                docs.append((ruta.name, leer_docx(ruta)))
            elif ext == ".jsonl":
                docs.extend((f"{ruta.name}#{i}", t) for i, t in enumerate(leer_jsonl(ruta)))
            else:
                print(f"  (ignorado, formato no soportado: {ruta.name})")
        except Exception as e:  # un archivo dañado no debe detener todo
            print(f"  (error leyendo {ruta.name}: {e})")
    return docs


# ---------- Limpieza ----------

def arreglar_mojibake(s):
    """Corrige textos mal decodificados, p. ej. 'informaciÃ³n' -> 'información'."""
    if "Ã" in s or "â€" in s:
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return s


def quitar_formulas(s):
    """Wikipedia deja fórmulas LaTeX como {\\displaystyle ...}; se eliminan con sus llaves."""
    marca = "{\\displaystyle"
    while marca in s:
        inicio = s.index(marca)
        nivel, fin = 0, len(s)
        for j in range(inicio, len(s)):
            if s[j] == "{":
                nivel += 1
            elif s[j] == "}":
                nivel -= 1
                if nivel == 0:
                    fin = j + 1
                    break
        s = s[:inicio] + s[fin:]
    return s


def limpiar_linea(s):
    s = arreglar_mojibake(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = s.replace("​", "").replace("﻿", "").replace("­", "")
    s = "".join(c for c in s if c == "\t" or unicodedata.category(c)[0] != "C")
    s = quitar_formulas(s)
    s = RE_CITAS.sub("", s)
    s = RE_CITAS_IA.sub("", s)
    s = RE_URL.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)   # sin espacio antes de puntuación
    s = re.sub(r"([(\[¿¡])\s+", r"\1", s)        # sin espacio después de apertura
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)         # paréntesis vacíos que quedan al quitar fórmulas
    return re.sub(r"\s{2,}", " ", s).strip()


def es_prosa(s, min_chars):
    """Descarta títulos sueltos, tablas, listas de números y basura."""
    if len(s) < min_chars:
        return False
    letras = sum(c.isalpha() for c in s)
    return letras / len(s) >= 0.6


def limpiar_documento(texto, min_chars, stats):
    parrafos, saltar_hasta_nivel = [], None
    for linea in texto.splitlines():
        m = RE_TITULO_WIKI.match(linea.strip())
        if m:  # título de sección de Wikipedia
            nivel, titulo = len(m.group(1)), m.group(2).strip().lower()
            if saltar_hasta_nivel is not None and nivel <= saltar_hasta_nivel:
                saltar_hasta_nivel = None
            if titulo in SECCIONES_EXCLUIDAS:
                saltar_hasta_nivel = nivel
            continue
        if saltar_hasta_nivel is not None:
            stats["lineas_secciones_excluidas"] += 1
            continue
        if not linea.strip():
            continue
        limpia = limpiar_linea(linea)
        if es_prosa(limpia, min_chars):
            parrafos.append(limpia)
        else:
            stats["lineas_ruido"] += 1
    return parrafos


def main():
    p = argparse.ArgumentParser(description="Limpia el corpus crudo")
    p.add_argument("--input", default="data/raw")
    p.add_argument("--out", default="data/clean")
    p.add_argument("--min_chars", type=int, default=40, help="largo mínimo de un párrafo")
    p.add_argument("--min_doc_chars", type=int, default=300, help="largo mínimo de un documento")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crudos = leer_documentos(args.input)
    if not crudos:
        raise SystemExit(f"No encontré archivos en {args.input}. Revisa el Paso 3.1 (recolectar corpus).")

    stats = {"archivos_o_documentos_leidos": len(crudos), "caracteres_crudos": sum(len(t) for _, t in crudos),
             "lineas_secciones_excluidas": 0, "lineas_ruido": 0, "parrafos_duplicados": 0,
             "documentos_descartados": 0}
    vistos, documentos = set(), []
    for nombre, texto in crudos:
        unicos = []
        for par in limpiar_documento(texto, args.min_chars, stats):
            clave = re.sub(r"\W+", "", par.lower())
            if clave in vistos:
                stats["parrafos_duplicados"] += 1
                continue
            vistos.add(clave)
            unicos.append(par)
        if sum(len(x) for x in unicos) >= args.min_doc_chars:
            documentos.append("\n".join(unicos))
        else:
            stats["documentos_descartados"] += 1

    corpus = "\n\n".join(documentos) + "\n"
    (out / "corpus.txt").write_text(corpus, encoding="utf-8")

    stats.update({
        "documentos_finales": len(documentos),
        "parrafos_finales": sum(d.count("\n") + 1 for d in documentos),
        "caracteres_finales": len(corpus),
        "palabras_finales": len(corpus.split()),
        "megabytes": round(len(corpus.encode("utf-8")) / 1024 / 1024, 2),
    })
    (out / "estadisticas.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Limpieza terminada ===")
    print(f"Documentos leídos          : {stats['archivos_o_documentos_leidos']}")
    print(f"Documentos finales         : {stats['documentos_finales']}  (descartados por cortos: {stats['documentos_descartados']})")
    print(f"Párrafos finales           : {stats['parrafos_finales']}")
    print(f"Párrafos duplicados quitados: {stats['parrafos_duplicados']}")
    print(f"Líneas de ruido quitadas   : {stats['lineas_ruido']}")
    print(f"Líneas de referencias/enlaces quitadas: {stats['lineas_secciones_excluidas']}")
    print(f"Caracteres: {stats['caracteres_crudos']:,} -> {stats['caracteres_finales']:,}")
    print(f"Palabras finales           : {stats['palabras_finales']:,}")
    print(f"Tamaño del corpus limpio   : {stats['megabytes']} MB  ->  {out / 'corpus.txt'}")
    print("\nEjemplo de párrafo limpio:")
    print("  " + documentos[0].split("\n")[0][:300])


if __name__ == "__main__":
    main()
