"""
Paso 6 (lámina 7/8) - Generar texto con el modelo entrenado
El modelo produce la respuesta token por token: predice, elige una pieza y la agrega al texto.

Uso:
    python src\\generate.py --ckpt checkpoints\\best.pt --prompt "¿Qué es IA?"
    python src\\generate.py --ckpt checkpoints\\best.pt --prompt "Alan Turing fue" --n 3
    python src\\generate.py --ckpt checkpoints\\best.pt --interactivo

Parámetros de muestreo:
    --temperature  < 1 más conservador, > 1 más creativo/aleatorio
    --top_k        solo elige entre las k piezas más probables
    --top_p        solo entre las piezas que suman el p% de la probabilidad (nucleus sampling)
    --repeticion   > 1 castiga repetir piezas ya usadas (evita "de la de la de la")
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import cargar_checkpoint  # noqa: E402


@torch.no_grad()
def tokens_generados(modelo, ids, device, eos_id, max_tokens=100, temperature=0.8, top_k=40, top_p=0.95,
                     repeticion=1.2):
    """Generador: en cada vuelta predice la siguiente pieza, la elige al azar según su probabilidad
    (filtrada por temperature, top_k y top_p) y la devuelve. Se detiene en </s>."""
    ids = list(ids)
    n_prompt = len(ids)
    contexto = modelo.cfg.context
    modelo.eval()
    for _ in range(max_tokens):
        x = torch.tensor([ids[-contexto:]], device=device)
        logits = modelo(x)[0][0, -1].float()

        if repeticion != 1.0:  # penalización de repetición (estilo CTRL)
            usados = torch.tensor(sorted(set(ids[n_prompt - 1:])), device=device)
            valores = logits[usados]
            logits[usados] = torch.where(valores > 0, valores / repeticion, valores * repeticion)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            corte = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits[logits < corte] = float("-inf")
        if top_p and top_p < 1.0:
            orden, indices = torch.sort(logits, descending=True)
            acumulada = torch.cumsum(F.softmax(orden, dim=-1), dim=-1)
            quitar = acumulada > top_p
            quitar[1:] = quitar[:-1].clone()
            quitar[0] = False
            logits[indices[quitar]] = float("-inf")

        siguiente = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if siguiente == eos_id:
            return
        ids.append(siguiente)
        yield siguiente


def stream_texto(modelo, tok, ids_prompt, device, **opciones):
    """Devuelve el texto nuevo pedazo a pedazo (para mostrarlo token por token)."""
    ids = list(ids_prompt)
    previo = tok.decode(ids)
    for t in tokens_generados(modelo, ids_prompt, device, tok.token_to_id("</s>"), **opciones):
        ids.append(t)
        texto = tok.decode(ids)
        nuevo, previo = texto[len(previo):], texto
        if nuevo:
            yield nuevo


def generar(modelo, tok, prompt, device, max_tokens=100, temperature=0.8, top_k=40, top_p=0.95,
            repeticion=1.2, mostrar=True, prefijo_ids=None):
    """Genera a partir de un prompt de texto (o de IDs ya armados en prefijo_ids)."""
    ids = list(prefijo_ids) if prefijo_ids is not None else [tok.token_to_id("<s>")] + tok.encode(prompt).ids
    if mostrar:
        print(tok.decode(ids), end="", flush=True)
    partes = []
    for trozo in stream_texto(modelo, tok, ids, device, max_tokens=max_tokens, temperature=temperature,
                              top_k=top_k, top_p=top_p, repeticion=repeticion):
        partes.append(trozo)
        if mostrar:
            print(trozo, end="", flush=True)
    if mostrar:
        print()
    return "".join(partes).strip(), ids


def main():
    p = argparse.ArgumentParser(description="Genera texto con el mini LLM")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--prompt", default="La inteligencia artificial")
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repeticion", type=float, default=1.2)
    p.add_argument("--n", type=int, default=1, help="cuántas respuestas generar")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactivo", action="store_true", help="escribir varios prompts seguidos")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    modelo, info = cargar_checkpoint(args.ckpt, args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    detalle = f"paso {info['paso']}, valid loss {info['valid_loss']:.3f}" if "paso" in info else ""
    print(f"Modelo: {args.ckpt} ({modelo.num_parametros() / 1e6:.1f}M parámetros{', ' + detalle if detalle else ''})")
    print(f"Muestreo: temperature {args.temperature} | top_k {args.top_k} | top_p {args.top_p} | "
          f"repetición {args.repeticion} | máx. {args.max_tokens} tokens\n")

    opciones = dict(max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, repeticion=args.repeticion)
    if args.interactivo:
        print("Escribe un inicio de texto (Enter vacío o 'salir' para terminar).")
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() == "salir":
                break
            generar(modelo, tok, prompt, args.device, **opciones)
    else:
        for i in range(args.n):
            if args.n > 1:
                print(f"--- Respuesta {i + 1} ---")
            generar(modelo, tok, args.prompt, args.device, **opciones)
            print()


if __name__ == "__main__":
    main()
