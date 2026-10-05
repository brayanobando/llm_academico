"""
Paso 4 (lámina 5/8) - El Transformer completo (tipo GPT, solo decodificador)

    Tokens (IDs) -> Embeddings + Posición -> [Atención + MLP] × N -> LayerNorm -> Logits (vocab)

Cada salida es un vector de 16000 puntajes (logits): uno por pieza del vocabulario.
El más alto es la pieza que el modelo cree que viene después.
"""
import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    context: int = 256
    d_model: int = 320
    n_heads: int = 5
    n_layers: int = 7
    d_ff: int = 1280
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # "sinusoidal" (Transformer 2017) o "learned" (GPT-2)
    pre_norm: bool = True             # True: LayerNorm antes de cada subcapa (GPT-2, LLaMA)
    tie_weights: bool = True          # la capa de salida reutiliza la matriz de embeddings
    flash: bool = True

    @classmethod
    def from_dict(cls, d):
        validos = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in validos})


class PosicionSinusoidal(nn.Module):
    """Codificación posicional fija con senos y cosenos (las "ondas" de la lámina).
    Cada posición recibe un vector único; el modelo lo suma al embedding para saber el orden."""

    def __init__(self, context, d_model):
        super().__init__()
        pe = torch.zeros(context, d_model)
        pos = torch.arange(context, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, T):
        return self.pe[:T]


class MLP(nn.Module):
    """Red feed-forward de cada bloque: procesa cada token por separado."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class BloqueTransformer(nn.Module):
    """Multi-Head Attention -> Add & Norm -> MLP -> Add & Norm  (se repite × N)"""

    def __init__(self, cfg):
        super().__init__()
        self.pre_norm = cfg.pre_norm
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.context, cfg.dropout, cfg.flash)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        if self.pre_norm:  # Norm -> subcapa -> Add (más estable al entrenar)
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:              # subcapa -> Add -> Norm (orden exacto del diagrama y del paper de 2017)
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # ID -> vector de tamaño d_model
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        else:
            self.pos_emb = PosicionSinusoidal(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.bloques = nn.ModuleList([BloqueTransformer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.pre_norm else nn.Identity()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # vector -> logits
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_pesos)
        # Las proyecciones que vuelven a la "vía residual" empiezan más pequeñas (como GPT-2)
        for nombre, p in self.named_parameters():
            if nombre.endswith("out_proj.weight") or nombre.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_pesos(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())  # los pesos compartidos se cuentan una vez

    def forward(self, idx, targets=None, ignore_index=-100):
        """idx: (B, T) IDs de entrada. targets: (B, T) IDs esperados (la entrada corrida una posición)."""
        B, T = idx.shape
        assert T <= self.cfg.context, f"La secuencia ({T}) supera el contexto ({self.cfg.context})"

        if self.cfg.pos_encoding == "learned":
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x = self.tok_emb(idx) + pos
        else:
            x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model) + self.pos_emb(T)
        x = self.drop(x)
        for bloque in self.bloques:
            x = bloque(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Entropía cruzada: qué tan "sorprendido" está el modelo con la pieza correcta
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Genera token por token: predice, elige uno al azar según las probabilidades y lo agrega."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits.float(), dim=-1)
            siguiente = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, siguiente], dim=1)
            if eos_id is not None and bool((siguiente == eos_id).all()):
                break
        return idx


def cargar_config(ruta):
    import yaml
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def crear_modelo(cfg_dict):
    return MiniGPT(GPTConfig.from_dict(cfg_dict))


def cargar_checkpoint(ruta, device="cpu"):
    """Carga un modelo entrenado desde best.pt / last.pt, o desde model.safetensors (+ config.json).
    Devuelve (modelo en modo evaluación, info del checkpoint)."""
    import json
    from pathlib import Path

    ruta = Path(ruta)
    if ruta.suffix == ".safetensors":
        from safetensors.torch import load_model
        cfg = json.loads((ruta.parent / "config.json").read_text(encoding="utf-8"))
        modelo = crear_modelo(cfg["modelo"])
        load_model(modelo, str(ruta))
        info = {"config": cfg}
    else:
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        modelo = crear_modelo(ck["config"]["modelo"])
        modelo.load_state_dict(ck["modelo"])
        info = {k: v for k, v in ck.items() if k not in ("modelo", "optimizador", "scaler")}
    return modelo.to(device).eval(), info
