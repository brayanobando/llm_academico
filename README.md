# llm_academico — Mini LLM desde cero (Windows 11)

Implementación de la guía «Creando un LLM desde cero» (Parte 3, serie técnica Win11).
Corpus: **Historia de la Inteligencia Artificial** (en español).
Hardware: portátil con GTX 1650 Ti (CUDA, 4 GB VRAM).

## Probar el asistente (después de clonar)

```powershell
git clone https://github.com/brayanobando/llm_academico.git
cd llm_academico
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126   # sin GPU: .../whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

El chat se abre en `http://localhost:8501`. Funciona también sin GPU (más lento).

## Qué incluye el repositorio

| Carpeta | Contenido |
|---|---|
| `scripts/`, `src/`, `app.py` | Todo el código: datos, tokenizer, Transformer, entrenamiento, evaluación, SFT y chat |
| `configs/` | Hiperparámetros (13M y 77M) y los 164 títulos de Wikipedia |
| `data/raw/` | 164 artículos de Wikipedia en español (CC BY-SA 4.0) y 2 documentos Word propios |
| `data/clean/`, `data/tokenizer/`, `data/train/` | Corpus limpio, tokenizer BPE de 16 000 piezas y bloques de entrenamiento |
| `data/sft/` | Dataset de preguntas y respuestas (73 escritas a mano, 2 070 en total) |
| `checkpoints/model.safetensors` | Pesos del modelo base (12.5 M parámetros) |
| `checkpoints/sft/` | El asistente final: `best_sft.pt` (lo usa el chat) y `model.safetensors` |
| `checkpoints/*.csv, *.json, *.png` | Historial de entrenamiento, métricas y curvas |

`checkpoints/best.pt` y `checkpoints/last.pt` no están en el repositorio porque pesan 150 MB cada uno
(GitHub admite hasta 100 MB por archivo). Guardan los mismos pesos que `model.safetensors` más el estado
del optimizador, así que solo hacen falta para reanudar el entrenamiento; se vuelven a crear con `src\train.py`.
Sin ellos, el chat usa `checkpoints\model.safetensors` para el modelo base, y en los comandos de abajo
se puede escribir `--ckpt checkpoints\model.safetensors` donde dice `checkpoints\best.pt`.

## Ruta de pasos

| Paso | Lámina | Qué se hace | Archivos |
|---|---|---|---|
| 1 | 2/8 | Preparar Windows 11 y crear el proyecto | `winget`, `venv` |
| 2 | 3/8 | Instalar PyTorch (CUDA) y paquetes | `requirements.txt` |
| 3 | 4/8 | Preparar corpus y entrenar tokenizer | `scripts\download_wikipedia.py`, `prepare_data.py`, `train_tokenizer.py`, `make_sequences.py` |
| 4 | 5/8 | Construir el Transformer | `src\models\attention.py`, `src\models\transformer.py`, `configs\llm_13m.yaml`, `scripts\check_model.py` |
| 5 | 6/8 | Entrenar en CPU o GPU | `src\train.py` |
| 6 | 7/8 | Evaluar y generar texto | `src\eval.py`, `src\generate.py` |
| 7 | 8/8 | SFT y asistente académico | `scripts\make_sft_data.py`, `src\sft.py`, `app.py` |

## Comandos (PowerShell, desde esta carpeta)

```powershell
# Entorno (una vez por ventana)
& "$env:USERPROFILE\venvs\llm_academico\Scripts\Activate.ps1"

# Paso 3 - datos y tokenizer
python scripts\download_wikipedia.py --titles configs\wikipedia_titulos.txt --out data\raw\wikipedia
python scripts\prepare_data.py --input data\raw --out data\clean
python scripts\train_tokenizer.py --input data\clean --vocab_size 16000
python scripts\make_sequences.py --input data\clean --context 256

# Paso 4 - verificar la arquitectura
python scripts\check_model.py --config configs\llm_13m.yaml

# Paso 5 - preentrenamiento
python src\train.py --device cuda --config configs\llm_13m.yaml --amp fp16

# Paso 6 - evaluación y generación
python src\eval.py --ckpt checkpoints\best.pt checkpoints\last.pt
python src\generate.py --ckpt checkpoints\best.pt --prompt "Alan Turing fue" --n 3

# Paso 7 - SFT y chat
python scripts\make_sft_data.py --out data\sft\qa.jsonl
python src\sft.py --base checkpoints\best.pt --data data\sft\qa.jsonl
python scripts\probar_robustez.py --ckpt checkpoints\sft\best_sft.pt
streamlit run app.py
```

## Resultados del preentrenamiento (modelo de 12.5M)

| Conjunto | Loss | Perplexity | Acierto top-1 | Acierto top-5 |
|---|---|---|---|---|
| train | 3.573 | 35.6 | 31.5 % | 55.7 % |
| valid | 5.086 | 161.7 | 20.0 % | 40.3 % |
| test | 5.069 | 159.1 | 19.8 % | 40.1 % |

Mejor checkpoint en el paso 1450 (época 13); desde ahí la validación empeora (sobreajuste).
Curvas en `checkpoints\curvas_entrenamiento.png`.

## Resultados del afinado SFT

Se hicieron dos rondas en la GTX 1650 Ti, ambas desde `checkpoints\best.pt`:

| Ronda | Pares (train / valid) | Pasos por época | Pérdida de validación | Mejor época | Tiempo |
|---|---|---|---|---|---|
| 1. Primer asistente | 715 (644 / 71) | 41 | 3.969 (antes) → 1.668 | 16 (parada temprana en la 20) | 5.6 min |
| 2. Asistente final (con aumento de datos) | 2 070 (1 863 / 207) | 117 | 2.201 (época 1) → 0.0069 | 28 de 30 | 26.6 min |

La validación de la ronda 2 son otras formas de preguntar hechos que sí se entrenaron, por eso baja tanto;
no se puede comparar con la de la ronda 1.

- Evolución de la respuesta a "¿Quién fue Alan Turing?" (ronda 1): época 3 confunde la persona (responde sobre McCarthy), época 6 acierta en parte, época 9 en adelante responde correctamente.
- Respuestas del asistente final (decodificación voraz):

| Pregunta | Respuesta del asistente |
|---|---|
| ¿Qué es una red neuronal? | Una red neuronal artificial es un modelo de aprendizaje formado por unidades llamadas neuronas, organizadas en capas y conectadas por pesos… |
| ¿Quién fue Alan Turing? | Alan Mathison Turing fue un matemático, lógico, informático teórico, criptógrafo, filósofo y biólogo teórico británico… |
| ¿Qué fue la Conferencia de Dartmouth? | …un taller de investigación celebrado en el Dartmouth College en 1956… nacimiento oficial de la inteligencia artificial. |
| quien fue frank rosenblatt | Frank Rosenblatt fue un psicólogo estadounidense notable en el campo de inteligencia artificial. |
| ¿Qué sucedió en 1997? | En 1997: Deep Blue (IBM) vence al campeón mundial de ajedrez Garri Kaspárov. |
| ¿Cuál es la capital de Colombia? | Solo fui entrenado con textos sobre la historia de la inteligencia artificial… Pregúntame sobre IA. |

- Limitación: el asistente responde bien lo que está en el dataset y sus variantes, pero con preguntas muy distintas puede mezclar datos.

## Estructura

```
llm_academico/
├── app.py            # chat local (Streamlit)
├── configs/          # llm_13m.yaml, llm_77m.yaml, wikipedia_titulos.txt
├── data/
│   ├── raw/          # Wikipedia + documentos propios (.docx)
│   ├── clean/        # corpus limpio
│   ├── tokenizer/    # tokenizer.json, vocab.json, merges.txt
│   ├── train/        # bloques train / valid / test (.npy)
│   └── sft/          # qa_manual.jsonl (a mano) + qa.jsonl (dataset SFT completo)
├── scripts/          # datos, tokenizer, verificación, dataset SFT
├── src/
│   ├── models/       # attention.py, transformer.py
│   ├── train.py, eval.py, generate.py, sft.py
└── checkpoints/      # best.pt, last.pt, model.safetensors, sft/
```

## Ajustes respecto al PDF

- **Entorno virtual fuera de OneDrive** (`%USERPROFILE%\venvs\llm_academico`, con `--system-site-packages`) para no sincronizar ~4 GB de PyTorch.
- **PyTorch cu130** (ya instalado en el equipo) en lugar de cu126; el driver 592.82 soporta CUDA 13.1.
- **AMP en fp16 en lugar de bf16**: la GTX 1650 Ti (Turing) no soporta bf16.
- **Tokenizer BPE** con la librería `tokenizers`, estilo SentencePiece (espacios como `▁`).
- **Corpus**: 164 artículos de Wikipedia en español (CC BY-SA 4.0) + 2 documentos propios.
- **pyyaml** y **streamlit** añadidos a los paquetes.

## Robustez ante distintas formas de preguntar

Problema: el asistente respondía bien "¿Qué pasó en 1950?" pero mal "QUE PASO EN 1950". El tokenizer distingue
mayúsculas y tildes, así que "QUE PASO" se parte en piezas raras (`▁Q`, `U`, `E`, `▁P`, `AS`, `O`) que el modelo casi no vio.

Soluciones aplicadas:
1. **Normalización de la pregunta** (`src\normalizar.py`, activa en el chat): mayúsculas según el corpus
   (alan → Alan, ia → IA), tildes (que → qué, paso → pasó, computacion → computación) y signos ¿?.
2. **Aumento de datos** (`scripts\make_sft_data.py --parafrasis 2`): cada pregunta suma 2 paráfrasis
   ("¿Qué sucedió en 1950?", "Cuéntame qué pasó en 1950.") y una versión informal ("que paso en 1950").
   El dataset pasa de 715 a 2 070 pares; la validación del SFT se hace solo con variantes.

Medición (`scripts\probar_robustez.py`: 12 preguntas × 5 formas, decodificación voraz, acierto si la respuesta
se parece ≥ 60 % a la del dataset):

| Forma de preguntar | Primer asistente: sin normalizar | con normalización | Asistente final: sin normalizar | con normalización |
|---|---|---|---|---|
| original | 12/12 | 12/12 | 12/12 | 12/12 |
| sin signos | 10/12 | 12/12 | 12/12 | 12/12 |
| minúsculas sin tildes | 3/12 | 11/12 | 12/12 | 12/12 |
| MAYÚSCULAS | 0/12 | 11/12 | 0/12 | 12/12 |
| otras palabras (no vistas) | 9/12 | 8/12 | 9/12 | 9/12 |
| **total** | **57 %** | **90 %** | **75 %** | **95 %** |

Revisión final del normalizador: ya no pone mayúscula a palabras que en el corpus solo aparecen al inicio de una
oración ("¿Quién eres?" se convertía en "¿Quién Eres?" y el modelo respondía sobre otra persona), respeta guiones
y símbolos (GPT-3, 1957–1958, A*) y letras como ö o Č (Gödel, Čapek). Con eso el total pasó de 93 % a 95 %.
Las paráfrasis totalmente nuevas siguen siendo el punto débil de un modelo de 12.5M.
