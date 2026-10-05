"""
Paso 3 (lámina 4/8) - 4. Crear secuencias
Convierte el corpus limpio en IDs con el tokenizer y lo corta en bloques de contexto.
Cada bloque tiene context+1 tokens: el modelo ve los primeros `context` y aprende a predecir
el siguiente token en cada posición (entrada = bloque[:-1], objetivo = bloque[1:]).

Guarda en data/train:
    train.npy, valid.npy, test.npy   matrices (n_bloques, context+1) de IDs (uint16)
    meta.json                        tamaño del vocabulario, contexto, conteos

Uso:
    python scripts\\make_sequences.py --input data\\clean --context 256
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description="Crea bloques de contexto para train/valid/test")
    p.add_argument("--input", default="data/clean")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out", default="data/train")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--valid", type=float, default=0.05, help="fracción de bloques para validación")
    p.add_argument("--test", type=float, default=0.05, help="fracción de bloques para prueba")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    bos, eos = tok.token_to_id("<s>"), tok.token_to_id("</s>")

    texto = (Path(args.input) / "corpus.txt").read_text(encoding="utf-8")
    docs = [d for d in texto.split("\n\n") if d.strip()]

    # 1) Texto -> IDs. Cada documento queda como: <s> ...tokens... </s>
    flujo = []
    for enc in tok.encode_batch(docs):
        flujo.extend([bos, *enc.ids, eos])
    flujo = np.array(flujo, dtype=np.uint16)
    print(f"Documentos: {len(docs)}  |  Tokens totales: {len(flujo):,}")

    # 2) Cortar en bloques de context+1 tokens (paso = context; comparten 1 token en el borde)
    L = args.context + 1
    n_bloques = (len(flujo) - 1) // args.context
    if n_bloques < 3:
        raise SystemExit("El corpus es demasiado pequeño para este contexto. Agrega más texto o baja --context.")
    bloques = np.stack([flujo[i * args.context: i * args.context + L] for i in range(n_bloques)])

    # 3) Repartir bloques al azar (con semilla fija) en train / valid / test
    rng = np.random.default_rng(args.seed)
    orden = rng.permutation(n_bloques)
    n_valid = max(1, int(n_bloques * args.valid))
    n_test = max(1, int(n_bloques * args.test))
    partes = {
        "valid": bloques[orden[:n_valid]],
        "test": bloques[orden[n_valid:n_valid + n_test]],
        "train": bloques[orden[n_valid + n_test:]],
    }
    for nombre, arr in partes.items():
        np.save(out / f"{nombre}.npy", arr)

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "context": args.context,
        "tokens_totales": int(len(flujo)),
        "bloques": {k: int(len(v)) for k, v in partes.items()},
        "tokens_especiales": {"pad": tok.token_to_id("[PAD]"), "unk": tok.token_to_id("[UNK]"), "bos": bos, "eos": eos},
        "tokenizer": args.tokenizer,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Secuencias creadas (contexto = {args.context}) ===")
    for k, v in partes.items():
        print(f"{k:5s}: {len(v):6,} bloques  ({len(v) * args.context:,} tokens)  ->  {out / (k + '.npy')}")
    print(f"Vocabulario: {meta['vocab_size']}  |  Forma de cada archivo: (n_bloques, {L})")

    ejemplo = partes["train"][0][:20].tolist()
    print("\nEjemplo, primeros 20 IDs de un bloque de entrenamiento:")
    print(f"  IDs   : {ejemplo}")
    print(f"  Piezas: {[tok.id_to_token(i) for i in ejemplo]}")
    print(f"  Texto : {tok.decode(ejemplo)}")


if __name__ == "__main__":
    main()
