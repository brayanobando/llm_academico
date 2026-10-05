"""
Paso 6 (lámina 7/8) - Evaluar el modelo
    1. Validar  -> medir con datos que el modelo no vio (valid y test)
    2. Métricas -> loss, perplexity y acierto del siguiente token
    3. Curvas   -> gráfica train vs valid desde checkpoints/historial.csv
    4. Comparar -> varios checkpoints lado a lado para detectar sobreajuste

Uso:
    python src\\eval.py --ckpt checkpoints\\best.pt
    python src\\eval.py --ckpt checkpoints\\best.pt checkpoints\\last.pt     (comparar)
"""
import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import cargar_checkpoint  # noqa: E402

# Colores validados para la gráfica (azul = train, naranja = validación, como en la lámina)
SUPERFICIE, TEXTO, TEXTO_2, REJILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"
AZUL, NARANJA = "#2a78d6", "#eb6834"


@torch.no_grad()
def medir(modelo, datos, device, batch_size=16):
    """Devuelve loss promedio, perplexity y % de acierto top-1 / top-5 del siguiente token."""
    usar_amp = device == "cuda"
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if usar_amp else nullcontext
    suma_loss, top1, top5, n_tokens = 0.0, 0, 0, 0
    for i in range(0, len(datos), batch_size):
        b = torch.from_numpy(datos[i:i + batch_size].astype(np.int64)).to(device)
        x, y = b[:, :-1], b[:, 1:]
        with ctx():
            logits, loss = modelo(x, y)
        n = y.numel()
        suma_loss += loss.item() * n
        mejores5 = logits.float().topk(5, dim=-1).indices
        top1 += (mejores5[..., 0] == y).sum().item()
        top5 += (mejores5 == y.unsqueeze(-1)).any(-1).sum().item()
        n_tokens += n
    loss = suma_loss / n_tokens
    return {"loss": round(loss, 4), "perplexity": round(math.exp(loss), 2),
            "acierto_top1": round(100 * top1 / n_tokens, 2), "acierto_top5": round(100 * top5 / n_tokens, 2),
            "tokens": n_tokens}


def graficar_curvas(ruta_csv, ruta_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(ruta_csv, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return None
    pasos = [int(r["paso"]) for r in filas]
    train = [float(r["train_loss"]) for r in filas]
    valid = [float(r["valid_loss"]) for r in filas]
    i_mejor = int(np.argmin(valid))

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    # Zona donde la validación ya no mejora
    if i_mejor < len(pasos) - 1:
        from matplotlib.transforms import blended_transform_factory
        ax.axvspan(pasos[i_mejor], pasos[-1], color="#efeeea", zorder=0, lw=0)
        ax.text((pasos[i_mejor] + pasos[-1]) / 2, 0.95,
                "Sobreajuste:\ntrain sigue bajando,\nvalid deja de mejorar",
                transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="top", fontsize=9, color=TEXTO_2)

    estilo = dict(lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(pasos, train, color=AZUL, label="Train (entrenamiento)", **estilo)
    ax.plot(pasos, valid, color=NARANJA, label="Validación (datos no vistos)", **estilo)

    # Mejor punto (best.pt): marcador con anillo del color de la superficie
    ax.plot(pasos[i_mejor], valid[i_mejor], "o", ms=9, color=NARANJA, mec=SUPERFICIE, mew=2, zorder=4)
    ax.annotate(f"best.pt\npaso {pasos[i_mejor]} · valid {valid[i_mejor]:.3f}\n"
                f"perplexity {math.exp(valid[i_mejor]):.1f}",
                (pasos[i_mejor], valid[i_mejor]), xytext=(0, 40), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=TEXTO,
                arrowprops=dict(arrowstyle="-", color=EJE, lw=1))

    # Etiquetas directas al final de cada línea (texto en tinta neutra, no en el color de la serie)
    for serie, nombre in ((train, "train"), (valid, "valid")):
        ax.annotate(f"{nombre} {serie[-1]:.2f}", (pasos[-1], serie[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXTO_2)

    ax.set_title("Curvas de pérdida del mini LLM", loc="left", fontsize=13, color=TEXTO, pad=22)
    ax.text(0, 1.02, "Entropía cruzada del siguiente token, medida cada 50 pasos (menos es mejor)",
            transform=ax.transAxes, fontsize=9.5, color=TEXTO_2)
    ax.set_xlabel("Paso de entrenamiento", color=TEXTO_2, fontsize=10)
    ax.set_ylabel("Pérdida (loss)", color=TEXTO_2, fontsize=10)
    ax.grid(axis="y", color=REJILLA, lw=1, ls="-")
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.set_xlim(0, pasos[-1] * 1.08)
    leyenda = ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0), frameon=False, fontsize=9.5)
    for t in leyenda.get_texts():
        t.set_color(TEXTO)
    fig.tight_layout()
    fig.savefig(ruta_png, facecolor=SUPERFICIE)
    plt.close(fig)
    return pasos[i_mejor], valid[i_mejor]


def main():
    p = argparse.ArgumentParser(description="Evalúa uno o varios checkpoints")
    p.add_argument("--ckpt", nargs="+", default=["checkpoints/best.pt"])
    p.add_argument("--data", default="data/train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    carpeta_datos = Path(args.data)
    conjuntos = {n: np.load(carpeta_datos / f"{n}.npy") for n in ("train", "valid", "test")}
    resultados = {}

    for ruta in args.ckpt:
        modelo, info = cargar_checkpoint(ruta, args.device)
        detalle = f"paso {info['paso']}, época {info['epoca'] + 1}" if "paso" in info else "safetensors"
        print(f"\n=== {ruta}  ({detalle}, {modelo.num_parametros() / 1e6:.1f}M parámetros) ===")
        print(f"{'Conjunto':10s} {'Loss':>8s} {'Perplexity':>11s} {'Acierto top-1':>14s} {'Acierto top-5':>14s}")
        res = {}
        for nombre, datos in conjuntos.items():
            m = medir(modelo, datos, args.device)
            res[nombre] = m
            print(f"{nombre:10s} {m['loss']:8.3f} {m['perplexity']:11.1f} {m['acierto_top1']:13.1f}% {m['acierto_top5']:13.1f}%")
        brecha = res["valid"]["loss"] - res["train"]["loss"]
        print(f"Brecha valid - train: {brecha:.2f}  ->  ", end="")
        if brecha > 1.0:
            print("sobreajuste fuerte: el modelo memoriza el corpus de entrenamiento.")
        elif brecha > 0.3:
            print("sobreajuste moderado: generaliza, pero ya memoriza parte del corpus.")
        else:
            print("buena generalización.")
        resultados[str(ruta)] = res

    if len(resultados) > 1:
        print("\n=== Comparación (datos de prueba que ningún modelo vio) ===")
        for ruta, res in resultados.items():
            print(f"  {ruta:35s} test loss {res['test']['loss']:.3f} | perplexity {res['test']['perplexity']:.1f}")
        mejor = min(resultados, key=lambda r: resultados[r]["test"]["loss"])
        print(f"  -> Generaliza mejor: {mejor}")

    carpeta_ck = Path(args.ckpt[0]).parent
    (carpeta_ck / "metricas.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_hist = carpeta_ck / "historial.csv"
    if csv_hist.exists():
        png = carpeta_ck / "curvas_entrenamiento.png"
        graficar_curvas(csv_hist, png)
        print(f"\nCurvas guardadas en: {png}")
    print(f"Métricas guardadas en: {carpeta_ck / 'metricas.json'}")
    print("\nCómo leerlo: perplexity = e^loss. Una perplexity de 160 significa que, en promedio, el modelo duda")
    print("como si eligiera entre ~160 piezas igual de probables (al inicio dudaba entre 16000).")


if __name__ == "__main__":
    main()
