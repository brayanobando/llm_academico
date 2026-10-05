"""
Mide qué tan bien responde el asistente cuando la misma pregunta se escribe de otra forma.

Cada pregunta se prueba de 5 formas: original, sin signos, minúsculas sin tildes, MAYÚSCULAS y con
otras palabras (paráfrasis que NO están en el dataset). Una respuesta cuenta como correcta si se
parece al menos un 60 % a la respuesta del dataset. Se usa decodificación voraz (siempre la pieza
más probable) para que el resultado sea repetible.

Uso:
    python scripts\\probar_robustez.py --ckpt checkpoints\\sft\\best_sft.pt
    python scripts\\probar_robustez.py --ckpt checkpoints\\sft\\best_sft.pt --detalle
"""
import argparse
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import torch
from tokenizers import Tokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from generate import generar  # noqa: E402
from models import cargar_checkpoint  # noqa: E402
from normalizar import Normalizador  # noqa: E402
from sft import ids_prompt  # noqa: E402

# (pregunta del dataset, paráfrasis que no está en el dataset)
CASOS = [
    ("¿Qué pasó en 1950 en la historia de la IA?", "Cuéntame lo que pasó en 1950"),
    ("¿Qué pasó en 1997 en la historia de la IA?", "¿Qué evento importante hubo en 1997?"),
    ("¿Qué pasó en 2017 en la historia de la IA?", "En 2017, ¿qué pasó con la IA?"),
    ("¿Quién fue Alan Turing?", "Quisiera saber sobre Alan Turing"),
    ("¿Quién fue Frank Rosenblatt?", "Información sobre Frank Rosenblatt"),
    ("¿Quién creó ELIZA?", "¿Quién fue el creador de ELIZA?"),
    ("¿Qué es una red neuronal?", "Explícame las redes neuronales"),
    ("¿Qué es un Transformer?", "Quiero saber qué es un Transformer"),
    ("¿Qué fue la Conferencia de Dartmouth?", "Dame información de la Conferencia de Dartmouth"),
    ("¿Qué es la lógica difusa?", "¿Podrías explicarme la lógica difusa?"),
    ("¿Quién acuñó el término inteligencia artificial?", "¿Quién inventó el término inteligencia artificial?"),
    ("¿Qué es el aprendizaje profundo?", "¿Qué es el deep learning?"),
]
FORMAS = ["original", "sin signos", "minúsculas sin tildes", "MAYÚSCULAS", "otras palabras"]


def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def variantes(pregunta, parafrasis):
    sin_signos = pregunta.strip("¿?¡!. ")
    informal = sin_tildes(sin_signos).lower()
    return [pregunta, sin_signos, informal, informal.upper(), parafrasis]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/sft/best_sft.pt")
    p.add_argument("--data", default="data/sft/qa.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--corpus", default="data/clean/corpus.txt")
    p.add_argument("--umbral", type=float, default=0.6)
    p.add_argument("--detalle", action="store_true", help="mostrar cada respuesta")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    modelo, _ = cargar_checkpoint(args.ckpt, args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    normalizar = Normalizador(args.corpus)
    referencia = {}
    for linea in Path(args.data).read_text(encoding="utf-8").splitlines():
        if linea.strip():
            o = json.loads(linea)
            referencia.setdefault(o["pregunta"], o["respuesta"])

    cache = {}

    def responder(q):
        if q not in cache:
            torch.manual_seed(0)
            cache[q], _ = generar(modelo, tok, None, args.device, max_tokens=60, temperature=1.0, top_k=1,
                                  top_p=1.0, repeticion=1.0, mostrar=False, prefijo_ids=ids_prompt(tok, q))
        return cache[q]

    aciertos = {modo: [0] * len(FORMAS) for modo in ("sin normalizar", "con normalización")}
    usados = 0
    for canonica, parafrasis in CASOS:
        ref = referencia.get(canonica)
        if ref is None:
            print(f"(omitida, no está en {args.data}: {canonica})")
            continue
        usados += 1
        for i, q in enumerate(variantes(canonica, parafrasis)):
            for modo, entrada in (("sin normalizar", q), ("con normalización", normalizar(q))):
                r = responder(entrada)
                parecido = SequenceMatcher(None, r.lower()[:len(ref) + 20], ref.lower(), autojunk=False).ratio()
                ok = parecido >= args.umbral
                aciertos[modo][i] += ok
                if args.detalle:
                    print(f"[{'OK ' if ok else 'MAL'} {parecido:.2f}] ({modo}) {entrada}\n      -> {r[:110]}")

    print(f"\n=== Robustez del asistente ({usados} preguntas × {len(FORMAS)} formas, umbral {args.umbral:.0%}) ===")
    print(f"{'Forma de preguntar':24s} {'Sin normalizar':>15s} {'Con normalización':>18s}")
    for i, forma in enumerate(FORMAS):
        a, b = aciertos["sin normalizar"][i], aciertos["con normalización"][i]
        print(f"{forma:24s} {a:>8d}/{usados:<6d} {b:>10d}/{usados:<6d}")
    ta, tb = sum(aciertos["sin normalizar"]), sum(aciertos["con normalización"])
    total = usados * len(FORMAS)
    print(f"{'TOTAL':24s} {ta:>8d}/{total:<6d} {tb:>10d}/{total:<6d}   ({ta / total:.0%} -> {tb / total:.0%})")


if __name__ == "__main__":
    main()
