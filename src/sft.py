"""
Paso 7 (lámina 8/8) - 2. Afinar el modelo con SFT (supervised fine-tuning)
Parte del LLM base (checkpoints/best.pt) y lo sigue entrenando con preguntas y respuestas,
para que aprenda a RESPONDER en lugar de solo continuar el texto.

Formato de cada ejemplo (los tokens especiales se reservaron en el Paso 3):
    <s> <|usuario|> ¿Qué es una red neuronal? <|asistente|> Una red neuronal es ... </s>
La pérdida solo se calcula sobre la respuesta: el modelo no necesita aprender a escribir la pregunta.

Uso:
    python src\\sft.py --base checkpoints\\best.pt --data data\\sft\\qa.jsonl
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import cargar_checkpoint, crear_modelo  # noqa: E402

IGNORAR = -100  # etiqueta que la función de pérdida ignora

PREGUNTAS_DE_PRUEBA = [
    "¿Qué es una red neuronal?",
    "¿Quién fue Alan Turing?",
    "¿Qué fue la Conferencia de Dartmouth?",
    "quien fue frank rosenblatt",          # informal: minúsculas, sin tildes ni signos
    "¿Qué sucedió en 1997?",               # otra forma de preguntar por un hito
]


def ids_prompt(tok, pregunta):
    """IDs de la parte que escribe el usuario: <s> <|usuario|> pregunta <|asistente|>"""
    return ([tok.token_to_id("<s>"), tok.token_to_id("<|usuario|>")] + tok.encode(pregunta.strip()).ids
            + [tok.token_to_id("<|asistente|>")])


def leer_pares(ruta):
    pares = []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if linea.strip():
            o = json.loads(linea)
            q = o.get("pregunta") or o.get("question") or o.get("instruction")
            a = o.get("respuesta") or o.get("answer") or o.get("output")
            if q and a:
                pares.append((q, a, o.get("fuente", ""), bool(o.get("variante"))))
    return pares


def codificar(pares, tok, contexto):
    """Convierte cada par en (entrada, objetivo). En el objetivo, la pregunta queda en -100."""
    eos = tok.token_to_id("</s>")
    ejemplos, recortados = [], 0
    for q, a, *_ in pares:
        prompt = ids_prompt(tok, q)
        ids = prompt + tok.encode(a.strip()).ids + [eos]
        if len(ids) > contexto + 1:
            ids, recortados = ids[:contexto + 1], recortados + 1
        x, y = ids[:-1], ids[1:]
        y = [IGNORAR if j < len(prompt) - 1 else t for j, t in enumerate(y)]
        ejemplos.append((x, y))
    return ejemplos, recortados


def armar_lote(ejemplos, pad_id, device):
    largo = max(len(x) for x, _ in ejemplos)
    X = torch.full((len(ejemplos), largo), pad_id, dtype=torch.long)
    Y = torch.full((len(ejemplos), largo), IGNORAR, dtype=torch.long)
    for i, (x, y) in enumerate(ejemplos):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


@torch.no_grad()
def evaluar(modelo, ejemplos, B, pad_id, device, ctx):
    modelo.eval()
    total, n = 0.0, 0
    for i in range(0, len(ejemplos), B):
        X, Y = armar_lote(ejemplos[i:i + B], pad_id, device)
        with ctx():
            _, loss = modelo(X, Y)
        k = (Y != IGNORAR).sum().item()
        total += loss.item() * k
        n += k
    modelo.train()
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="Afinado SFT del mini LLM")
    p.add_argument("--base", default="checkpoints/best.pt")
    p.add_argument("--data", default="data/sft/qa.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--valid", type=float, default=0.1, help="fracción de pares para validación")
    p.add_argument("--paciencia", type=int, default=3, help="épocas sin mejorar antes de parar")
    p.add_argument("--amp", choices=["fp16", "none"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("[PAD]")

    # ---- Modelo base (el preentrenado del Paso 5) ----
    modelo, info = cargar_checkpoint(args.base, device)
    cfg = info["config"]
    contexto = cfg["modelo"]["context"]
    modelo.train()

    # ---- Datos ----
    # Todas las preguntas originales se entrenan. La validación sale de las VARIANTES (otras formas de
    # preguntar lo mismo), así mide si el modelo responde bien aunque le pregunten distinto.
    # Sin variantes en el dataset, la validación sale de las preguntas automáticas (nunca de las escritas a mano).
    pares = leer_pares(args.data)
    random.shuffle(pares)
    n_valid = max(1, int(len(pares) * args.valid))
    variantes = [p for p in pares if p[3]]
    if variantes:
        reservadas = variantes[:n_valid]
        entrenables = [p for p in pares if not p[3]] + variantes[n_valid:]
    else:
        otros = [p for p in pares if p[2] != "manual"]
        reservadas = otros[:n_valid]
        entrenables = [p for p in pares if p[2] == "manual"] + otros[n_valid:]
    valid_ej, _ = codificar(reservadas, tok, contexto)
    train_ej, recortados = codificar(entrenables, tok, contexto)
    tokens_resp = sum(sum(t != IGNORAR for t in y) for _, y in train_ej)

    usar_amp = device == "cuda" and args.amp == "fp16"
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if usar_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    con_decay = [q for q in modelo.parameters() if q.dim() >= 2]
    sin_decay = [q for q in modelo.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": con_decay, "weight_decay": 0.1},
                             {"params": sin_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    B = args.batch_size
    pasos_epoca = math.ceil(len(train_ej) / B)
    total = pasos_epoca * args.epochs

    def lr_en(paso):
        if paso < args.warmup:
            return args.lr * (paso + 1) / args.warmup
        prog = min(1.0, (paso - args.warmup) / max(1, total - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("=== Afinado SFT ===")
    print(f"Modelo base: {args.base} ({modelo.num_parametros() / 1e6:.1f}M parámetros) | dispositivo: {device} | AMP: {'fp16' if usar_amp else 'no'}")
    print(f"Pares: {len(train_ej)} de entrenamiento, {len(valid_ej)} de validación "
          f"({tokens_resp:,} tokens de respuesta; {recortados} recortados al contexto)")
    print(f"Plan : {args.epochs} épocas × {pasos_epoca} pasos = {total} pasos (lote {B}, lr {args.lr})\n")

    def responder(pregunta, max_tokens=80):
        from generate import generar
        texto, _ = generar(modelo, tok, None, device, max_tokens=max_tokens, temperature=0.2, top_k=40,
                           top_p=0.9, repeticion=1.1, mostrar=False, prefijo_ids=ids_prompt(tok, pregunta))
        modelo.train()
        return texto.strip()

    loss_ini = evaluar(modelo, valid_ej, B, pad_id, device, ctx)
    print(f"Pérdida de validación ANTES del SFT: {loss_ini:.3f}  (el modelo base aún no sabe responder)")
    print(f"  P: {PREGUNTAS_DE_PRUEBA[0]}\n  R (base): {responder(PREGUNTAS_DE_PRUEBA[0])}\n")

    historial, mejor, sin_mejora, paso, t0 = [], float("inf"), 0, 0, time.time()
    for epoca in range(args.epochs):
        random.shuffle(train_ej)
        suma, n = 0.0, 0
        for i in range(pasos_epoca):
            X, Y = armar_lote(train_ej[i * B:(i + 1) * B], pad_id, device)
            for g in opt.param_groups:
                g["lr"] = lr_en(paso)
            with ctx():
                _, loss = modelo(X, Y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            paso += 1
            suma += loss.item()
            n += 1

        v = evaluar(modelo, valid_ej, B, pad_id, device, ctx)
        historial.append({"epoca": epoca + 1, "paso": paso, "train_loss": round(suma / n, 4),
                          "valid_loss": round(v, 4), "tiempo_s": round(time.time() - t0, 1)})
        marca = ""
        if v < mejor:
            mejor, sin_mejora = v, 0
            torch.save({"modelo": modelo.state_dict(), "config": cfg, "paso": paso, "epoca": epoca,
                        "valid_loss": v, "mejor_valid": v, "historial": historial, "tipo": "sft"},
                       out / "best_sft.pt")
            marca = "  <- mejor (best_sft.pt)"
        else:
            sin_mejora += 1
        print(f"Época {epoca + 1:2d}/{args.epochs} | train {suma / n:.3f} | valid {v:.3f}{marca}")
        if (epoca + 1) % 3 == 0:
            print(f"    P: {PREGUNTAS_DE_PRUEBA[1]}\n    R: {responder(PREGUNTAS_DE_PRUEBA[1])}")
        if sin_mejora >= args.paciencia:
            print(f"\nLa validación no mejora hace {args.paciencia} épocas: se detiene (parada temprana).")
            break

    with open(out / "sft_historial.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        w.writeheader()
        w.writerows(historial)

    # ---- Exportar y probar el mejor modelo afinado ----
    ck = torch.load(out / "best_sft.pt", map_location="cpu", weights_only=False)
    modelo.load_state_dict(ck["modelo"])
    try:
        from safetensors.torch import save_model
        copia = crear_modelo(cfg["modelo"])
        copia.load_state_dict(ck["modelo"])
        save_model(copia, str(out / "model.safetensors"))
        (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"(No se pudo exportar a safetensors: {e})")

    print(f"\n=== Resumen ===")
    print(f"Mejor validación: {mejor:.3f} (antes del SFT: {loss_ini:.3f}) | tiempo: {(time.time() - t0) / 60:.1f} min")
    print(f"Archivos: {out / 'best_sft.pt'}, {out / 'model.safetensors'}, {out / 'sft_historial.csv'}")
    print("\nPrueba del asistente:")
    for q in PREGUNTAS_DE_PRUEBA:
        print(f"  P: {q}\n  R: {responder(q, max_tokens=100)}\n")


if __name__ == "__main__":
    main()
