"""
Paso 3 (lámina 4/8) - 1. Recolectar corpus
Descarga artículos de Wikipedia en español como texto plano y los guarda en data/raw.

Uso (PowerShell, desde la carpeta llm_academico):
    python scripts\\download_wikipedia.py --titles configs\\wikipedia_titulos.txt --out data\\raw\\wikipedia

Los textos de Wikipedia tienen licencia CC BY-SA 4.0: cítala como fuente en tu informe.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://es.wikipedia.org/w/api.php"
USER_AGENT = "llm-academico/1.0 (proyecto academico universitario; corpus historia de la IA)"


def leer_titulos(ruta):
    titulos = []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            titulos.append(linea)
    return list(dict.fromkeys(titulos))  # quita repetidos conservando el orden


def nombre_archivo(titulo):
    return re.sub(r"[^\w\-]+", "_", titulo).strip("_") + ".txt"


def descargar(titulo, reintentos=8):
    """Devuelve (titulo_real, texto) o (None, None) si el artículo no existe."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,          # texto plano, sin HTML
        "exsectionformat": "wiki",  # títulos de sección como == Título == (para limpiarlos después)
        "redirects": 1,
        "titles": titulo,
        "format": "json",
        "formatversion": 2,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    espera = 5
    for _ in range(reintentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                datos = json.load(r)
            pagina = datos["query"]["pages"][0]
            if pagina.get("missing"):
                return None, None
            return pagina["title"], pagina.get("extract", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:  # demasiadas peticiones o error del servidor
                retry_after = e.headers.get("Retry-After", "")
                pausa = int(retry_after) if retry_after.isdigit() else espera
                print(f"    Wikipedia pidió esperar (HTTP {e.code}), reintento en {pausa}s...", flush=True)
                time.sleep(pausa)
                espera = min(espera * 2, 60)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    Error de red ({e}), reintento en {espera}s...", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 60)
    raise RuntimeError(f"No se pudo descargar «{titulo}» después de {reintentos} intentos")


def main():
    p = argparse.ArgumentParser(description="Descarga artículos de Wikipedia (es) para el corpus")
    p.add_argument("--titles", default="configs/wikipedia_titulos.txt")
    p.add_argument("--out", default="data/raw/wikipedia")
    p.add_argument("--delay", type=float, default=1.0, help="segundos entre peticiones")
    p.add_argument("--force", action="store_true", help="volver a descargar aunque ya exista")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    titulos = leer_titulos(args.titles)
    print(f"Artículos en la lista: {len(titulos)}  ->  {out}\n")

    nuevos = saltados = faltantes = 0
    for i, titulo in enumerate(titulos, 1):
        destino = out / nombre_archivo(titulo)
        if destino.exists() and not args.force:
            saltados += 1
            continue
        real, texto = descargar(titulo)
        if real is None or not texto.strip():
            faltantes += 1
            print(f"[{i:3d}/{len(titulos)}] NO ENCONTRADO: {titulo}", flush=True)
        else:
            destino.write_text(texto, encoding="utf-8")
            nuevos += 1
            print(f"[{i:3d}/{len(titulos)}] {real}  ({len(texto) / 1024:.1f} KB)", flush=True)
        time.sleep(args.delay)

    archivos = list(out.glob("*.txt"))
    total = sum(f.stat().st_size for f in archivos)
    print("\n=== Resumen ===")
    print(f"Descargados ahora : {nuevos}")
    print(f"Ya existían       : {saltados}")
    print(f"No encontrados    : {faltantes}")
    print(f"Archivos en {out}: {len(archivos)}  |  Tamaño total: {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
