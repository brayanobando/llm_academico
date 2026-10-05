"""
Paso 7 (lámina 8/8) - 3. Interfaz local: chat con Streamlit
Muestra el mini LLM como un asistente académico en el navegador.
El tema de colores está en .streamlit/config.toml.

Uso (PowerShell, desde la carpeta llm_academico, con el entorno activo):
    streamlit run app.py
"""
import html
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
from tokenizers import Tokenizer

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from generate import tokens_generados  # noqa: E402
from models import cargar_checkpoint  # noqa: E402
from normalizar import Normalizador  # noqa: E402
from sft import ids_prompt  # noqa: E402

BASE = RAIZ / "checkpoints" / "best.pt"
MODELOS = {
    "Asistente afinado (SFT)": RAIZ / "checkpoints" / "sft" / "best_sft.pt",
    # best.pt pesa 150 MB y no se sube a GitHub; si falta, se usan los mismos pesos desde model.safetensors
    "Modelo base": BASE if BASE.exists() else RAIZ / "checkpoints" / "model.safetensors",
}
DESCRIPCION_MODELOS = {
    "Asistente afinado (SFT)": "Responde preguntas",
    "Modelo base": "Solo continúa el texto que escribes",
}

# Épocas de la historia de la IA (de la cronología del corpus): inicio, fin, nombre, tipo, color
EPOCAS = [
    (1950, 1956, "Orígenes", "auge", "#F7DDB0"),
    (1956, 1974, "Nacimiento del campo", "auge", "#F4C982"),
    (1974, 1980, "Primer invierno", "invierno", None),
    (1980, 1987, "Sistemas expertos", "auge", "#F1B65E"),
    (1987, 1993, "Segundo invierno", "invierno", None),
    (1993, 2012, "Resurgimiento", "auge", "#EEA048"),
    (2012, 2017, "Aprendizaje profundo", "auge", "#EA8B3E"),
    (2017, 2022, "Era Transformer", "auge", "#E57A38"),
    (2022, None, "IA generativa", "auge", "#DE6A33"),
]
# Hitos clicables: sus respuestas están en el dataset SFT (cronología del Word)
HITOS = [
    ("1950", "Prueba de Turing", "¿Qué pasó en 1950 en la historia de la IA?"),
    ("1956", "Dartmouth", "¿Qué pasó en 1956 en la historia de la IA?"),
    ("1966", "ELIZA", "¿Qué pasó en 1966 en la historia de la IA?"),
    ("1997", "Deep Blue", "¿Qué pasó en 1997 en la historia de la IA?"),
    ("2012", "AlexNet", "¿Qué pasó en 2012 en la historia de la IA?"),
    ("2016", "AlphaGo", "¿Qué pasó en 2016 en la historia de la IA?"),
    ("2017", "Transformer", "¿Qué pasó en 2017 en la historia de la IA?"),
    ("2022", "ChatGPT", "¿Qué ocurrió el 30 de noviembre de 2022?"),
]
CONCEPTOS = [
    "¿Qué es una red neuronal?",
    "¿Qué es un Transformer?",
    "¿Quién fue Alan Turing?",
    "¿Qué es el invierno de la IA?",
    "¿Qué significan Q, K y V?",
    "¿Quién eres?",
]

st.set_page_config(page_title="Mini LLM · Historia de la IA", page_icon="🧠", layout="centered")

ESTILOS = """
<style>
:root {
  --tinta: #1B2033; --tinta-2: #5A6178; --tinta-3: #7A8199;
  --papel: #F3F5FA; --borde: #D5DBEA; --blanco: #FFFFFF;
  --cobalto: #2D5BE3; --indigo: #1B2559; --ambar: #F0A43A; --ambar-osc: #A86412;
  --hielo: #D6E8F5; --hielo-2: #C2DAEC; --turquesa: #129B8A; --turquesa-osc: #0E7468;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 880px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* ---------- Encabezado ---------- */
.titulo { font-size: clamp(2rem, 4.2vw, 2.75rem); font-weight: 650; line-height: 1.1; text-wrap: balance;
          letter-spacing: -0.025em; color: var(--tinta); margin: 0 0 .7rem; padding: 0; }
[data-testid="stMarkdownContainer"] p.bajada { font-size: 1.05rem; line-height: 1.55; color: var(--tinta-2); max-width: 62ch; margin: 0 0 1.6rem; }

/* ---------- Franja del tiempo ---------- */
.franja { display: flex; height: 44px; border-radius: 12px; overflow: hidden;
          box-shadow: inset 0 0 0 1px rgba(27,32,51,.08); }
.seg { display: flex; align-items: center; justify-content: center; min-width: 0;
       font-size: .8rem; font-weight: 500; color: var(--tinta); white-space: nowrap; overflow: hidden;
       border-right: 2px solid var(--papel); }
.seg:last-child { border-right: 0; }
.seg.invierno { background: repeating-linear-gradient(135deg, var(--hielo) 0 6px, var(--hielo-2) 6px 12px);
                color: #2A4A6B; font-size: .95rem; }
.escala { position: relative; height: 1.3rem; margin-top: .35rem; font-size: .74rem; color: var(--tinta-3); }
.escala span { position: absolute; top: 0; transform: translateX(-50%); }
.escala span.ini { transform: none; }
.escala span.fin { transform: translateX(-100%); }
.leyenda { display: flex; flex-wrap: wrap; gap: 1.2rem; font-size: .8rem; color: var(--tinta-2); margin: .3rem 0 1.4rem; }
.leyenda i { display: inline-block; width: 22px; height: 10px; border-radius: 3px; margin-right: .45rem; vertical-align: -1px; }
.leyenda .sw-auge { background: linear-gradient(90deg, #F7DDB0, #DE6A33); }
.leyenda .sw-invierno { background: repeating-linear-gradient(135deg, var(--hielo) 0 4px, var(--hielo-2) 4px 8px);
                        box-shadow: inset 0 0 0 1px #B4CDE2; }
[data-testid="stMarkdownContainer"] p.guia { font-size: .9rem; font-weight: 500; color: var(--tinta); margin: 0 0 .5rem; }

/* ---------- Hitos (botones) ---------- */
.st-key-hitos button { background: var(--blanco); border: 1px solid var(--borde); border-radius: 999px;
                       padding: .3rem .85rem; min-height: 2.3rem; }
.st-key-hitos button p::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                                 background: var(--ambar); margin-right: .5rem; vertical-align: 1px; }
.st-key-hitos button:hover { border-color: var(--ambar); background: #FFF7EA; color: var(--tinta); }
.st-key-hitos button p { font-size: .86rem; white-space: nowrap; }
.st-key-hitos button strong { font-weight: 650; margin-right: .2rem; }

/* ---------- Conceptos (pills) ---------- */
.st-key-conceptos { margin-top: .4rem; }
.st-key-conceptos button { background: var(--blanco); border: 1px solid var(--borde); }
.st-key-conceptos button:hover { border-color: var(--turquesa); color: var(--turquesa-osc); background: #EEF8F6; }
[data-testid="stMarkdownContainer"] p.nota { font-size: .8rem; line-height: 1.5; color: var(--tinta-3); margin: 1.1rem 0 1.4rem; }

/* ---------- Conversación ---------- */
.msg { display: flex; margin: .75rem 0; }
.msg.usuario { justify-content: flex-end; }
.msg.usuario .columna { display: flex; flex-direction: column; align-items: flex-end; max-width: 75%; }
.msg.usuario .globo { background: var(--cobalto); color: #fff; padding: .65rem 1rem;
                      border-radius: 18px 18px 4px 18px; line-height: 1.5; font-size: .98rem; }
.interpretada { font-size: .76rem; color: var(--tinta-3); margin-top: .35rem; }
.interpretada b { font-weight: 600; color: var(--tinta-2); }
.msg.modelo .tarjeta { background: var(--blanco); border: 1px solid #DDE2EE; border-left: 4px solid var(--turquesa);
                       border-radius: 4px 14px 14px 14px; padding: .8rem 1.1rem .7rem; max-width: 88%;
                       box-shadow: 0 1px 2px rgba(27,37,89,.05); }
.msg.modelo.base .tarjeta { border-left-color: var(--ambar); }
.quien { font-size: .78rem; font-weight: 600; color: var(--turquesa-osc); margin-bottom: .3rem; }
.msg.modelo.base .quien { color: var(--ambar-osc); }
.texto { line-height: 1.62; color: var(--tinta); font-size: 1rem; }
.texto strong { font-weight: 600; color: var(--tinta-2); }
.meta { font-size: .74rem; color: var(--tinta-3); margin-top: .5rem; }
.cursor { display: inline-block; width: .55ch; height: 1.05em; margin-left: 2px; vertical-align: -2px;
          background: var(--turquesa); animation: parpadeo 1s steps(1) infinite; }
@keyframes parpadeo { 50% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .cursor { animation: none; } }

/* ---------- Caja de texto ---------- */
[data-testid="stBottomBlockContainer"] { background: var(--papel); max-width: 880px; }
[data-testid="stChatInput"] > div { background: var(--blanco); border: 1px solid var(--borde); border-radius: 16px;
                                    box-shadow: 0 2px 6px rgba(27,37,89,.06); }
[data-testid="stChatInput"] > div:focus-within { border-color: var(--cobalto); }

/* ---------- Barra lateral ---------- */
.marca-titulo { font-size: 1.4rem; font-weight: 650; color: #FFFFFF; letter-spacing: -0.01em; }
.marca-sub { font-size: .85rem; color: #AEB6DA; margin: .15rem 0 1.4rem; }
.ficha { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem 1rem; margin: 1.1rem 0 .2rem; }
.ficha dt { font-size: .74rem; color: #AEB6DA; }
.ficha dd { margin: .1rem 0 0; font-size: .98rem; font-weight: 600; color: #FFFFFF; }
[data-testid="stSidebar"] hr { border-color: #34407F; }

@media (max-width: 640px) {
  .msg.usuario .columna, .msg.modelo .tarjeta { max-width: 100%; }
  .seg { font-size: .7rem; }
  .escala span.menor { display: none; }
  .st-key-hitos [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .5rem; }
  .st-key-hitos [data-testid="stColumn"] { min-width: calc(50% - .25rem) !important; flex: 1 1 calc(50% - .25rem) !important; }
}
</style>
"""
st.markdown(ESTILOS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar(ruta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, info = cargar_checkpoint(ruta, device)
    tok = Tokenizer.from_file(str(RAIZ / "data" / "tokenizer" / "tokenizer.json"))
    return modelo, tok, device, info


@st.cache_resource(show_spinner=False)
def cargar_normalizador():
    corpus = RAIZ / "data" / "clean" / "corpus.txt"
    return Normalizador(corpus) if corpus.exists() else None


def escapar(texto):
    return html.escape(texto).replace("\n", "<br>")


def html_usuario(texto, interpretada=None):
    nota = (f'<div class="interpretada">Interpretada como <b>{escapar(interpretada)}</b></div>'
            if interpretada and interpretada != texto else "")
    return f'<div class="msg usuario"><div class="columna"><div class="globo">{escapar(texto)}</div>{nota}</div></div>'


def html_modelo(texto, tipo, prompt="", meta="", escribiendo=False):
    quien = "Asistente" if tipo == "sft" else "Modelo base (continúa tu texto)"
    cuerpo = (f"<strong>{escapar(prompt)}</strong>" if tipo == "base" else "") + escapar(texto)
    cursor = '<span class="cursor"></span>' if escribiendo else ""
    pie = f'<div class="meta">{meta}</div>' if meta else ""
    return (f'<div class="msg modelo {tipo}"><div class="tarjeta"><div class="quien">{quien}</div>'
            f'<div class="texto">{cuerpo}{cursor}</div>{pie}</div></div>')


def html_franja():
    fin = max(2026, datetime.now().year)
    total = fin - 1950
    segmentos = []
    for ini, f, nombre, tipo, color in EPOCAS:
        f = f or fin
        ancho = (f - ini) / total * 100
        etiqueta = nombre if ancho >= 14 else ("❄" if tipo == "invierno" else "")
        fondo = f";background:{color}" if color else ""
        periodo = f"{ini}–{'hoy' if f == fin else f}"
        segmentos.append(f'<div class="seg {tipo}" style="flex:{f - ini}{fondo}" '
                         f'title="{nombre}, {periodo}">{etiqueta}</div>')
    marcas = []
    for anio in range(1950, fin, 10):
        clase = ' class="ini"' if anio == 1950 else (' class="menor"' if (anio - 1950) % 20 else "")
        marcas.append(f'<span{clase} style="left:{(anio - 1950) / total * 100:.2f}%">{anio}</span>')
    marcas.append('<span class="fin" style="left:100%">hoy</span>')
    return (f'<div class="franja" role="img" aria-label="Épocas de la historia de la IA desde 1950 hasta hoy">'
            f'{"".join(segmentos)}</div><div class="escala">{"".join(marcas)}</div>'
            '<div class="leyenda"><span><i class="sw-auge"></i>Épocas de auge</span>'
            '<span><i class="sw-invierno"></i>Inviernos de la IA</span></div>')


def preguntar(texto):
    st.session_state.pendiente = texto


def elegir_concepto():
    if st.session_state.concepto:
        st.session_state.pendiente = st.session_state.concepto
    st.session_state.concepto = None


# ---------------- Barra lateral ----------------
disponibles = {k: v for k, v in MODELOS.items() if v.exists()}
with st.sidebar:
    st.markdown('<div class="marca-titulo">Mini LLM</div>'
                '<div class="marca-sub">Historia de la IA en español</div>', unsafe_allow_html=True)
    if not disponibles:
        st.error("No encontré modelos entrenados. Ejecuta primero src\\train.py (Paso 5) y src\\sft.py (Paso 7).")
        st.stop()
    nombre = st.radio("Modelo", list(disponibles), captions=[DESCRIPCION_MODELOS[k] for k in disponibles])

modelo, tok, device, info = cargar(str(disponibles[nombre]))
cfg_modelo = info["config"]["modelo"]
tipo = "sft" if disponibles[nombre].name == "best_sft.pt" else "base"

with st.sidebar:
    filas = [
        ("Parámetros", f"{modelo.num_parametros() / 1e6:.1f} M"),
        ("Se ejecuta en", "GPU (CUDA)" if device == "cuda" else "CPU"),
        ("Vocabulario", f"{cfg_modelo['vocab_size']:,}".replace(",", " ")),
        ("Contexto", f"{cfg_modelo['context']} tokens"),
    ]
    if "paso" in info:
        filas.append(("Checkpoint", f"paso {info['paso']}"))
    if "valid_loss" in info:
        filas.append(("Pérdida (validación)", f"{info['valid_loss']:.3f}"))
    st.markdown('<dl class="ficha">' + "".join(f"<div><dt>{a}</dt><dd>{b}</dd></div>" for a, b in filas)
                + "</dl>", unsafe_allow_html=True)
    st.divider()
    normalizar_activo = st.toggle("Corregir mayúsculas y tildes", value=True,
                                  help="Convierte «QUE PASO EN 1950» en «¿Qué pasó en 1950?», la forma en que "
                                       "el modelo aprendió las preguntas.")
    temperatura = st.slider("Creatividad (temperatura)", 0.1, 1.5, 0.2, 0.1,
                            help="Baja: respuestas más fieles a lo aprendido. Alta: más variadas, pero inventa más.")
    max_tokens = st.slider("Largo máximo (tokens)", 20, 200, 120, 10)
    top_k = st.slider("Opciones por paso (top-k)", 5, 100, 40, 5,
                      help="En cada paso el modelo elige solo entre las k piezas más probables.")
    repeticion = st.slider("Evitar repeticiones", 1.0, 1.5, 1.1, 0.05,
                           help="Valores mayores castigan repetir palabras ya usadas.")
    if st.button("Nueva conversación", icon=":material/add_comment:", use_container_width=True):
        st.session_state.mensajes = []

# ---------------- Encabezado y franja del tiempo ----------------
st.markdown(
    '<h1 class="titulo">Pregúntale a un LLM entrenado desde cero</h1>'
    f'<p class="bajada">Un modelo de {modelo.num_parametros() / 1e6:.1f} millones de parámetros que aprendió la '
    'historia de la inteligencia artificial con medio millón de tokens de textos en español, '
    'entrenado en una GTX 1650 Ti.</p>' + html_franja(),
    unsafe_allow_html=True,
)

with st.container(key="hitos"):
    st.markdown('<p class="guia">Toca un hito para preguntar qué pasó ese año</p>', unsafe_allow_html=True)
    for fila in (HITOS[:4], HITOS[4:]):
        for col, (anio, etiqueta, pregunta_hito) in zip(st.columns(4), fila):
            col.button(f"**{anio}** {etiqueta}", key=f"hito_{anio}", on_click=preguntar, args=(pregunta_hito,),
                       use_container_width=True)

with st.container(key="conceptos"):
    st.markdown('<p class="guia">O pregunta por un concepto</p>', unsafe_allow_html=True)
    st.pills("Conceptos", CONCEPTOS, key="concepto", on_change=elegir_concepto, label_visibility="collapsed")

st.markdown('<p class="nota">Cada pregunta se responde por separado. El modelo puede equivocarse: '
            'confirma los datos importantes en fuentes confiables.</p>', unsafe_allow_html=True)

# ---------------- Conversación ----------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    if m["rol"] == "user":
        st.markdown(html_usuario(m["texto"], m.get("interpretada")), unsafe_allow_html=True)
    else:
        st.markdown(html_modelo(m["texto"], m["tipo"], m.get("prompt", ""), m.get("meta", "")),
                    unsafe_allow_html=True)

pregunta = st.chat_input("Pregunta sobre la historia de la IA") or st.session_state.pop("pendiente", None)
if pregunta:
    normalizador = cargar_normalizador()
    entrada = normalizador(pregunta) if (normalizar_activo and normalizador) else pregunta
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta, "interpretada": entrada})
    st.markdown(html_usuario(pregunta, entrada), unsafe_allow_html=True)

    if tipo == "sft":  # <s> <|usuario|> pregunta <|asistente|>  -> el modelo escribe la respuesta
        ids = ids_prompt(tok, entrada)
    else:              # el modelo base solo continúa el texto
        ids = [tok.token_to_id("<s>")] + tok.encode(entrada).ids
    texto_prompt = tok.decode(ids)

    espacio = st.empty()
    espacio.markdown(html_modelo("", tipo, entrada, escribiendo=True), unsafe_allow_html=True)
    generados, texto, t0 = [], "", time.perf_counter()
    for t in tokens_generados(modelo, ids, device, tok.token_to_id("</s>"), max_tokens=max_tokens,
                              temperature=temperatura, top_k=top_k, top_p=0.9, repeticion=repeticion):
        generados.append(t)
        texto = tok.decode(generados) if tipo == "sft" else tok.decode(ids + generados)[len(texto_prompt):]
        espacio.markdown(html_modelo(texto, tipo, entrada, escribiendo=True), unsafe_allow_html=True)
    segundos = time.perf_counter() - t0

    if not texto.strip():
        texto = "El modelo no generó texto. Sube un poco la creatividad e inténtalo de nuevo."
    meta = f"{len(generados)} tokens en {segundos:.1f} s"
    espacio.markdown(html_modelo(texto, tipo, entrada, meta), unsafe_allow_html=True)
    st.session_state.mensajes.append({"rol": "assistant", "texto": texto, "tipo": tipo,
                                      "prompt": entrada, "meta": meta})
