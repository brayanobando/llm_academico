"""
Paso 5 (lámina 6/8) - Entrenar el LLM en CPU o GPU
El modelo aprende prediciendo la siguiente pieza una y otra vez:

    1. Cargar batches   -> leer bloques de data/train y enviarlos al dispositivo
    2. Forward + loss   -> calcular logits y la pérdida (entropía cruzada) del siguiente token
    3. Backpropagation  -> calcular gradientes y actualizar pesos con AdamW
    4. Checkpoints      -> guardar el avance: best.pt (mejor validación) y last.pt (último estado)

Uso:
    python src\\train.py --device cuda --config configs\\llm_13m.yaml --amp fp16
    python src\\train.py --device cpu  --config configs\\llm_13m.yaml
    python src\\train.py --device cuda --config configs\\llm_13m.yaml --resume checkpoints\\last.pt
"""
import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import cargar_config, crear_modelo  # noqa: E402


# ---------------------------------------------------------------- utilidades

def elegir_amp(pedido, device):
    """Devuelve el tipo de precisión mixta a usar ('fp16', 'bf16' o 'none')."""
    if device != "cuda" or pedido in (None, "none"):
        return "none"
    if pedido == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        print("AVISO: esta GPU no soporta bf16 de forma nativa (requiere arquitectura Ampere o superior).")
        print("       Se usa fp16 con GradScaler, que en tu GPU es lo correcto.\n")
        return "fp16"
    return pedido


def tasa_aprendizaje(paso, total, cfg):
    """Calentamiento lineal y luego decaimiento en forma de coseno hasta min_lr."""
    lr, min_lr, warmup = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"]
    if paso < warmup:
        return lr * (paso + 1) / warmup
    progreso = min(1.0, (paso - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progreso))


def crear_optimizador(modelo, cfg):
    """AdamW: weight decay solo en matrices (no en sesgos ni LayerNorm)."""
    con_decay = [p for p in modelo.parameters() if p.dim() >= 2]
    sin_decay = [p for p in modelo.parameters() if p.dim() < 2]
    grupos = [{"params": con_decay, "weight_decay": cfg["weight_decay"]},
              {"params": sin_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(grupos, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluar(modelo, datos, batch_size, contexto_amp):
    """Pérdida promedio sobre datos que el modelo NO usa para entrenar."""
    modelo.eval()
    total, n = 0.0, 0
    for i in range(0, len(datos), batch_size):
        b = datos[i:i + batch_size]
        with contexto_amp():
            _, loss = modelo(b[:, :-1], b[:, 1:])
        total += loss.item() * len(b)
        n += len(b)
    modelo.train()
    return total / max(n, 1)


@torch.no_grad()
def muestra_de_texto(modelo, tok, device, prompt="La inteligencia artificial", tokens=40):
    modelo.eval()
    ids = [tok.token_to_id("<s>")] + tok.encode(prompt).ids
    salida = modelo.generate(torch.tensor([ids], device=device), max_new_tokens=tokens,
                             temperature=0.8, top_k=40, eos_id=tok.token_to_id("</s>"))
    modelo.train()
    return tok.decode(salida[0].tolist()).replace("\n", " ")


def guardar_historial(historial, carpeta):
    if not historial:
        return
    with open(carpeta / "historial.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        w.writeheader()
        w.writerows(historial)


# ---------------------------------------------------------------- entrenamiento

def main():
    p = argparse.ArgumentParser(description="Entrena el mini LLM")
    p.add_argument("--config", default="configs/llm_13m.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["fp16", "bf16", "none"], default=None, help="precisión mixta (por defecto, la del config)")
    p.add_argument("--data", default="data/train")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", default=None, help="checkpoint para continuar (p. ej. checkpoints/last.pt)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="cortar después de N pasos (para pruebas rápidas)")
    args = p.parse_args()

    cfg = cargar_config(args.config)
    mcfg, tcfg = cfg["modelo"], cfg["entrenamiento"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.batch_size:
        tcfg["batch_size"] = args.batch_size
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Pediste --device cuda pero PyTorch no detecta la GPU. Usa --device cpu o revisa el Paso 2.")
    amp = elegir_amp(args.amp or tcfg.get("amp", "none"), device)
    torch.manual_seed(tcfg.get("seed", 42))

    # ---- 1. Datos: bloques de context+1 tokens creados en el Paso 3 ----
    carpeta_datos = Path(args.data)
    meta = json.loads((carpeta_datos / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] != mcfg["vocab_size"] or meta["context"] != mcfg["context"]:
        raise SystemExit(f"Los datos (vocab {meta['vocab_size']}, contexto {meta['context']}) no coinciden con el "
                         f"config (vocab {mcfg['vocab_size']}, contexto {mcfg['context']}). Ajusta el .yaml o rehaz el Paso 3.")
    # El dataset es pequeño (~4 MB), así que se sube completo a la GPU una sola vez
    train = torch.from_numpy(np.load(carpeta_datos / "train.npy").astype(np.int64)).to(device)
    valid = torch.from_numpy(np.load(carpeta_datos / "valid.npy").astype(np.int64)).to(device)

    # ---- 2. Modelo, optimizador y precisión mixta ----
    modelo = crear_modelo(mcfg).to(device)
    opt = crear_optimizador(modelo, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    contexto_amp = (lambda: torch.autocast(device_type="cuda", dtype=dtype)) if dtype else nullcontext

    B = tcfg["batch_size"]
    pasos_por_epoca = math.ceil(len(train) / B)
    total_pasos = pasos_por_epoca * tcfg["epochs"]
    if args.max_steps:
        total_pasos = min(total_pasos, args.max_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paso, epoca_inicio, mejor_valid, historial, sin_mejora = 0, 0, float("inf"), [], 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["optimizador"])
        scaler.load_state_dict(ck["scaler"])
        paso, epoca_inicio = ck["paso"], ck["epoca"] + 1
        mejor_valid, historial = ck["mejor_valid"], ck["historial"]
        print(f"Reanudando desde {args.resume}: paso {paso}, época {epoca_inicio + 1}\n")

    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    except Exception:
        tok = None

    print("=== Entrenamiento ===")
    print(f"Modelo: {modelo.num_parametros() / 1e6:.1f}M parámetros | dispositivo: {device} | AMP: {amp}")
    print(f"Datos : {len(train):,} bloques de train, {len(valid):,} de validación, contexto {mcfg['context']}")
    print(f"Plan  : {tcfg['epochs']} épocas × {pasos_por_epoca} pasos = {total_pasos:,} pasos (lote {B})")
    print(f"Salida: {out / 'best.pt'}, {out / 'last.pt'}, {out / 'historial.csv'}\n")

    def guardar(nombre, epoca, valid_loss):
        torch.save({
            "modelo": modelo.state_dict(), "config": cfg, "paso": paso, "epoca": epoca,
            "valid_loss": valid_loss, "mejor_valid": mejor_valid, "historial": historial,
            "optimizador": opt.state_dict(), "scaler": scaler.state_dict(),
        }, out / nombre)

    generador = torch.Generator(device="cpu").manual_seed(tcfg.get("seed", 42))
    t_inicio, suma_loss, n_loss = time.time(), 0.0, 0
    paciencia = tcfg.get("paciencia", 0)
    detener = False
    epoca = epoca_inicio
    valid_loss = float("nan")
    modelo.train()
    try:
        for epoca in range(epoca_inicio, tcfg["epochs"]):
            orden = torch.randperm(len(train), generator=generador).to(device)
            barra = tqdm(range(pasos_por_epoca), desc=f"Época {epoca + 1}/{tcfg['epochs']}", ncols=100)
            for i in barra:
                if paso >= total_pasos:
                    detener = True
                    break
                # 1. Lote: entrada = bloque sin el último token, objetivo = bloque corrido una posición
                lote = train[orden[i * B:(i + 1) * B]]
                x, y = lote[:, :-1], lote[:, 1:]

                lr = tasa_aprendizaje(paso, total_pasos, tcfg)
                for g in opt.param_groups:
                    g["lr"] = lr

                # 2. Forward + loss
                with contexto_amp():
                    _, loss = modelo(x, y)

                # 3. Backpropagation + AdamW
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()

                paso += 1
                suma_loss += loss.item()
                n_loss += 1
                barra.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")

                # Validación periódica
                if paso % tcfg["eval_every"] == 0 or paso == total_pasos:
                    valid_loss = evaluar(modelo, valid, B, contexto_amp)
                    train_loss = suma_loss / max(n_loss, 1)
                    suma_loss, n_loss = 0.0, 0
                    historial.append({
                        "paso": paso, "epoca": epoca + 1, "lr": round(lr, 8),
                        "train_loss": round(train_loss, 4), "valid_loss": round(valid_loss, 4),
                        "valid_ppl": round(math.exp(min(valid_loss, 20)), 2),
                        "tiempo_s": round(time.time() - t_inicio, 1),
                    })
                    guardar_historial(historial, out)
                    marca = ""
                    if valid_loss < mejor_valid:
                        mejor_valid, sin_mejora = valid_loss, 0
                        guardar("best.pt", epoca, valid_loss)  # 4. Checkpoint del mejor modelo
                        marca = "  <- mejor (best.pt)"
                    else:
                        sin_mejora += 1
                    tqdm.write(f"  paso {paso:5d} | train {train_loss:.3f} | valid {valid_loss:.3f} "
                               f"| perplexity {math.exp(min(valid_loss, 20)):8.1f}{marca}")
                    if paciencia and sin_mejora >= paciencia:
                        tqdm.write(f"\nLa validación no mejora hace {paciencia} evaluaciones: posible sobreajuste. "
                                   f"Se detiene el entrenamiento (parada temprana).")
                        detener = True
                        break

            guardar("last.pt", epoca, valid_loss)  # 4. Checkpoint del último estado (para --resume)
            if tok is not None:
                tqdm.write(f"  Muestra al final de la época {epoca + 1}: {muestra_de_texto(modelo, tok, device)}\n")
            if detener:
                break
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido con Ctrl+C. Guardando checkpoints/last.pt ...")
        guardar("last.pt", epoca, valid_loss)

    # ---- Exportar el mejor modelo en formato safetensors ----
    minutos = (time.time() - t_inicio) / 60
    mejor = torch.load(out / "best.pt", map_location="cpu", weights_only=False) if (out / "best.pt").exists() else None
    if mejor is not None:
        try:
            from safetensors.torch import save_model
            modelo_cpu = crear_modelo(mcfg)
            modelo_cpu.load_state_dict(mejor["modelo"])
            save_model(modelo_cpu, str(out / "model.safetensors"))
            (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"(No se pudo exportar a safetensors: {e})")

    print("\n=== Resumen ===")
    print(f"Pasos: {paso:,} | tiempo: {minutos:.1f} min")
    if mejor is not None:
        print(f"Mejor validación: loss {mejor['valid_loss']:.3f} | perplexity {math.exp(mejor['valid_loss']):.1f} "
              f"(paso {mejor['paso']}, época {mejor['epoca'] + 1})")
        archivos = [out / n for n in ("best.pt", "last.pt", "model.safetensors", "historial.csv") if (out / n).exists()]
        print("Archivos: " + ", ".join(str(a) for a in archivos))


if __name__ == "__main__":
    main()
