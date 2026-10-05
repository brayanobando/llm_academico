"""
Paso 4 (lámina 5/8) - Verificar la arquitectura
Construye el modelo desde un config, cuenta parámetros, hace una pasada hacia adelante con
datos reales y un paso de entrenamiento de prueba en la GPU. NO entrena el modelo.

Uso:
    python scripts\\check_model.py --config configs\\llm_13m.yaml
    python scripts\\check_model.py --config configs\\llm_77m.yaml
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import cargar_config, crear_modelo  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llm_13m.yaml")
    p.add_argument("--data", default="data/train/train.npy")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = cargar_config(args.config)
    mcfg, tcfg = cfg["modelo"], cfg.get("entrenamiento", {})
    torch.manual_seed(tcfg.get("seed", 42))
    modelo = crear_modelo(mcfg).to(args.device)

    # ---- 1. Arquitectura y parámetros ----
    d, V, L = mcfg["d_model"], mcfg["vocab_size"], mcfg["n_layers"]
    p_emb = V * d
    p_bloque = sum(x.numel() for x in modelo.bloques[0].parameters())
    total = modelo.num_parametros()
    print(f"=== Arquitectura ({args.config}) ===")
    print(f"Vocabulario {V} | contexto {mcfg['context']} | d_model {d} | cabezas {mcfg['n_heads']} "
          f"(d_head {d // mcfg['n_heads']}) | bloques {L} | MLP {mcfg['d_ff']}")
    print(f"Posición: {mcfg.get('pos_encoding')} | pre_norm: {mcfg.get('pre_norm')} | "
          f"pesos compartidos: {mcfg.get('tie_weights')}")
    print(f"\nParámetros:")
    print(f"  Embeddings de tokens : {p_emb:>12,}")
    print(f"  Cada bloque          : {p_bloque:>12,}  × {L} = {p_bloque * L:,}")
    print(f"  TOTAL                : {total:>12,}  (~{total / 1e6:.1f}M)  |  {total * 4 / 1024**2:.0f} MB en fp32")

    # ---- 2. Pasada hacia adelante con un lote real ----
    bloques = np.load(args.data)
    B = min(tcfg.get("batch_size", 8), len(bloques))
    lote = torch.from_numpy(bloques[:B].astype(np.int64)).to(args.device)
    x, y = lote[:, :-1], lote[:, 1:]  # entrada y objetivo (corrido una posición)
    modelo.eval()
    with torch.no_grad():
        logits, loss = modelo(x, y)
    print(f"\n=== Pasada hacia adelante (dispositivo: {args.device}) ===")
    print(f"Entrada {tuple(x.shape)}  ->  logits {tuple(logits.shape)}   (lote, tokens, vocabulario)")
    print(f"Pérdida inicial: {loss.item():.3f}   (esperada ≈ ln({V}) = {math.log(V):.3f}: el modelo aún adivina al azar)")

    # ---- 3. Un paso de entrenamiento de prueba (para medir memoria y velocidad) ----
    modelo.train()
    usar_amp = args.device == "cuda" and tcfg.get("amp", "fp16") == "fp16"
    opt = torch.optim.AdamW(modelo.parameters(), lr=tcfg.get("lr", 3e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    pasos = 5
    for _ in range(pasos):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            _, loss = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if args.device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / pasos
    print(f"\n=== Paso de entrenamiento de prueba (lote {B}, AMP {'fp16' if usar_amp else 'no'}) ===")
    print(f"Tiempo por paso: {dt * 1000:.0f} ms  |  pérdida tras {pasos} pasos sobre el mismo lote: {loss.item():.3f}")
    if args.device == "cuda":
        pico = torch.cuda.max_memory_allocated() / 1024**3
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Memoria de GPU usada (pico): {pico:.2f} GB de {total_gpu:.1f} GB  ->  "
              f"{'cabe bien' if pico < total_gpu * 0.85 else 'muy justo: baja batch_size'}")

    # ---- 4. Generación sin entrenar ----
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        ids = [tok.token_to_id("<s>")] + tok.encode("La inteligencia artificial").ids
        modelo.eval()
        salida = modelo.generate(torch.tensor([ids], device=args.device), max_new_tokens=15, top_k=50)
        print("\n=== Generación SIN entrenar (debe salir texto sin sentido) ===")
        print("  " + tok.decode(salida[0].tolist()))
    except Exception as e:
        print(f"\n(No se pudo probar la generación: {e})")

    print("\nResultado: la arquitectura base del LLM queda definida en PyTorch y lista para entrenar.")


if __name__ == "__main__":
    main()
