"""
Normaliza la pregunta del usuario antes de dársela al modelo.

El tokenizer distingue mayúsculas y tildes: "QUE PASO EN 1950" se convierte en piezas raras
(▁Q, U, E, ▁P, AS, O...) que el modelo casi nunca vio. Esta clase la lleva a la forma en que
el modelo aprendió las preguntas: "¿Qué pasó en 1950?".

    1. Mayúsculas: cada palabra toma la forma más común en el corpus (alan -> Alan, ia -> IA).
       Solo cuentan las apariciones en medio de una oración (al inicio todas van con mayúscula),
       y si la palabra siguiente da contexto se usa el par ("deep learning" vs. "Deep Blue").
    2. Tildes: palabras desconocidas sin tilde toman la forma con tilde del corpus
       (computacion -> computación), y las palabras de pregunta se acentúan (que -> qué).
    3. Signos: si empieza con una palabra de pregunta, se agregan ¿ y ?.
Todo lo demás (guiones, números, símbolos como GPT-3 o A*) se deja tal como lo escribió el usuario.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

INTERROGATIVOS = {
    "que": "qué", "quien": "quién", "quienes": "quiénes", "cuando": "cuándo", "donde": "dónde",
    "como": "cómo", "cual": "cuál", "cuales": "cuáles", "cuanto": "cuánto", "cuantos": "cuántos",
    "cuanta": "cuánta", "cuantas": "cuántas",
}
# Verbos en pasado que suelen seguir a la palabra de pregunta ("qué pasó", "quién creó")
VERBOS_PASADO = {
    "paso": "pasó", "ocurrio": "ocurrió", "sucedio": "sucedió", "creo": "creó", "invento": "inventó",
    "acuno": "acuñó", "desarrollo": "desarrolló", "publico": "publicó", "gano": "ganó", "vencio": "venció",
    "lanzo": "lanzó", "fundo": "fundó", "nacio": "nació", "murio": "murió", "diseno": "diseñó",
    "diseño": "diseñó", "propuso": "propuso", "significo": "significó",
}
INTRODUCTORES = {"dime", "sabes", "explicame", "cuentame", "explica", "indica"}
# Peticiones frecuentes que casi no aparecen en el corpus (por eso no se pueden acentuar con él)
IMPERATIVOS = {"explicame": "explícame", "hablame": "háblame", "cuentame": "cuéntame",
               "describeme": "descríbeme", "ensename": "enséñame", "muestrame": "muéstrame"}
PALABRA = re.compile(r"[^\W_]+")          # letras de cualquier idioma (Gödel, Čapek) y números
INICIO_ORACION = set(".!?¿¡:;«\"“(")      # después de estos signos la mayúscula no dice nada
MINIMO = 3                                # apariciones necesarias para cambiar una palabra a mayúscula
PROPORCION = 0.75                         # ...y que el corpus la escriba así al menos el 75 % de las veces
# Palabras cortas que se quedan como las escribió el usuario ("el Turco", no "El Turco")
FUNCIONALES = {"a", "al", "con", "de", "del", "e", "el", "en", "la", "las", "lo", "los", "o", "para", "por",
               "se", "sin", "u", "un", "una", "y"}


def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _elegir(formas, por_defecto):
    """Forma escrita más común, solo si es clara (MINIMO apariciones y PROPORCION del total)."""
    if not formas:
        return por_defecto
    w, n = formas.most_common(1)[0]
    return w if n >= MINIMO and n / sum(formas.values()) >= PROPORCION else por_defecto


class Normalizador:
    def __init__(self, ruta_corpus):
        texto = Path(ruta_corpus).read_text(encoding="utf-8")
        minusculas = Counter()
        formas = defaultdict(Counter)   # "turing" -> {"Turing": 812}
        pares = defaultdict(Counter)    # ("deep", "learning") -> {"deep": 7, "Deep": 4}
        for linea in texto.splitlines():
            ps = list(PALABRA.finditer(linea))
            for k, m in enumerate(ps):
                w = m.group()
                minusculas[w.lower()] += 1
                antes = linea[ps[k - 1].end():m.start()] if k else ""
                if k == 0 or INICIO_ORACION & set(antes):
                    continue  # inicio de oración: no cuenta para decidir mayúsculas
                formas[w.lower()][w] += 1
                if k + 1 < len(ps) and not linea[m.end():ps[k + 1].start()].strip():
                    pares[(w.lower(), ps[k + 1].group().lower())][w] += 1
        self.conocidas = {w for w, n in minusculas.items() if n >= MINIMO}  # las muy raras pueden ser erratas
        self.forma = {k: _elegir(c, k) for k, c in formas.items()}
        self.pares = pares
        con_tilde = {}
        for w, n in minusculas.items():  # forma sin tildes -> forma con tildes más frecuente
            k = sin_tildes(w)
            if k not in con_tilde or n > con_tilde[k][1]:
                con_tilde[k] = (w, n)
        self.tilde = {k: v[0] for k, v in con_tilde.items()}

    def _palabra(self, w, siguiente=None):
        # Respeta lo que el usuario escribió con mayúsculas (Conferencia, ChatGPT); solo corrige minúsculas
        if not w.islower() or w in FUNCIONALES:
            return w
        bajo = w
        if sin_tildes(bajo) in IMPERATIVOS:
            return IMPERATIVOS[sin_tildes(bajo)]
        if len(bajo) == 1 and bajo not in "aeouy":  # letras sueltas: Q, K, V
            return bajo.upper()
        if bajo not in self.conocidas and sin_tildes(bajo) in self.tilde:
            bajo = self.tilde[sin_tildes(bajo)]
        par = self.pares.get((bajo, siguiente)) if siguiente else None
        if par and sum(par.values()) >= 2:  # hay contexto: "deep learning" se queda en minúscula
            return _elegir(par, bajo)
        return self.forma.get(bajo, bajo)

    def __call__(self, texto):
        texto = " ".join(texto.strip().split())
        if not texto:
            return texto
        ms = list(PALABRA.finditer(texto))
        letras = [c for c in texto if c.isalpha()]
        todo_mayus = bool(letras) and sum(c.isupper() for c in letras) / len(letras) > 0.7

        # En MAYÚSCULAS todo se trata como minúscula; la primera palabra, si solo trae la inicial
        bajas = [m.group().lower() if (todo_mayus or (k == 0 and m.group().istitle())) else m.group()
                 for k, m in enumerate(ms)]
        salida = []
        for k, w in enumerate(bajas):
            sig = bajas[k + 1].lower() if k + 1 < len(bajas) else None
            if sig and sig not in self.conocidas and sin_tildes(sig) in self.tilde:
                sig = self.tilde[sin_tildes(sig)]
            salida.append(self._palabra(w, sig))

        # Palabra de pregunta al inicio (o después de "dime", "sabes"...) y verbo en pasado justo después
        def clave(j):
            return sin_tildes(salida[j].lower())

        def acentuar_verbo(desde, hasta):
            for j in range(desde, min(hasta, len(salida))):
                if clave(j) in VERBOS_PASADO:
                    salida[j] = VERBOS_PASADO[clave(j)]
                    break

        es_pregunta = False
        if salida and clave(0) in INTRODUCTORES and len(salida) > 1 and clave(1) in INTERROGATIVOS:
            salida[1] = INTERROGATIVOS[clave(1)]  # "Dime quién fue X": es una petición, sin ¿?
            acentuar_verbo(2, 4)
        elif salida and clave(0) == "por" and len(salida) > 1 and clave(1) == "que":
            salida[0], salida[1], es_pregunta = "por", "qué", True
        elif salida and clave(0) in INTERROGATIVOS:
            salida[0], es_pregunta = INTERROGATIVOS[clave(0)], True
            acentuar_verbo(1, 3)

        # Reemplazar cada palabra en su lugar: guiones, números y símbolos quedan como estaban
        partes, fin = [], 0
        for m, w in zip(ms, salida):
            partes += [texto[fin:m.start()], w]
            fin = m.end()
        texto = "".join(partes) + texto[fin:]

        # Mayúscula inicial y signos de pregunta
        k = next((i for i, c in enumerate(texto) if c.isalnum()), 0)
        texto = texto[:k] + texto[k:k + 1].upper() + texto[k + 1:]
        if es_pregunta and not texto.startswith("¿"):
            texto = "¿" + texto
        if es_pregunta and not texto.endswith("?"):
            texto = texto.rstrip(".") + "?"
        return texto
