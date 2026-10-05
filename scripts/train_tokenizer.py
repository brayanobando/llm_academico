"""
Paso 3 (lámina 4/8) - 3. Entrenar el tokenizer
Entrena un tokenizer BPE (subpalabras) con la librería `tokenizers` sobre el corpus limpio.
Guarda en data/tokenizer:
    tokenizer.json   el tokenizer completo (lo usan los siguientes pasos)
    vocab.json       el vocabulario: pieza -> ID
    merges.txt       las fusiones BPE aprendidas

Uso:
    python scripts\\train_tokenizer.py --input data\\clean --vocab_size 16000
"""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# Tokens especiales (sus IDs quedan fijos: 0, 1, 2, 3, ...)
#   [PAD] relleno | [UNK] desconocido | <s> inicio de documento | </s> fin de documento
#   <|usuario|> y <|asistente|> se reservan para el afinado SFT del Paso 7 (lámina 8/8)
TOKENS_ESPECIALES = ["[PAD]", "[UNK]", "<s>", "</s>", "<|usuario|>", "<|asistente|>"]


def leer_documentos(carpeta):
    texto = (Path(carpeta) / "corpus.txt").read_text(encoding="utf-8")
    return [d for d in texto.split("\n\n") if d.strip()]


def crear_tokenizer():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    # 1) espacios -> "▁" (como sentencepiece), 2) saltos de línea y 3) puntuación como piezas aparte
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
        pre_tokenizers.Split("\n", behavior="isolated"),
        pre_tokenizers.Punctuation(behavior="isolated"),
    ])
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    return tok


def main():
    p = argparse.ArgumentParser(description="Entrena el tokenizer BPE")
    p.add_argument("--input", default="data/clean")
    p.add_argument("--out", default="data/tokenizer")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--min_frequency", type=int, default=2)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = leer_documentos(args.input)
    print(f"Documentos para entrenar el tokenizer: {len(docs)}")

    tok = crear_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=TOKENS_ESPECIALES,
        limit_alphabet=200,  # caracteres base; los muy raros (chino, griego...) quedan como [UNK]
        show_progress=True,
    )
    tok.train_from_iterator(docs, trainer=trainer, length=len(docs))

    tok.save(str(out / "tokenizer.json"))
    tok.model.save(str(out))  # escribe vocab.json y merges.txt

    # ---- Reporte ----
    vocab_real = tok.get_vocab_size()
    ids_especiales = {t: tok.token_to_id(t) for t in TOKENS_ESPECIALES}
    muestra = docs[: min(len(docs), 50)]
    n_palabras = sum(len(d.split()) for d in muestra)
    n_tokens = sum(len(e.ids) for e in tok.encode_batch(muestra))
    n_unk = sum(e.ids.count(ids_especiales["[UNK]"]) for e in tok.encode_batch(muestra))
    (out / "info.json").write_text(json.dumps({
        "vocab_size": vocab_real, "tokens_especiales": ids_especiales,
        "tokens_por_palabra": round(n_tokens / max(n_palabras, 1), 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Tokenizer entrenado ===")
    print(f"Vocabulario pedido / real : {args.vocab_size} / {vocab_real}")
    if vocab_real < args.vocab_size:
        print("  (el corpus no tiene suficientes pares repetidos para llegar al tamaño pedido; es normal con corpus pequeños)")
    print(f"Tokens especiales         : {ids_especiales}")
    print(f"Tokens por palabra        : {n_tokens / max(n_palabras, 1):.2f}  (menos es mejor; ~1.3-1.6 es típico en español)")
    print(f"Tokens [UNK] en la muestra: {n_unk}")
    print(f"Archivos: {out / 'tokenizer.json'}, {out / 'vocab.json'}, {out / 'merges.txt'}")

    for frase in ["La inteligencia artificial ayuda", "En 1956 se celebró la Conferencia de Dartmouth."]:
        enc = tok.encode(frase)
        print(f"\nTexto  : {frase}")
        print(f"Tokens : {enc.tokens}")
        print(f"IDs    : {enc.ids}")
        print(f"Decodif: {tok.decode(enc.ids)}")


if __name__ == "__main__":
    main()
