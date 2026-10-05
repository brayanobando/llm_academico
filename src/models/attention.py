"""
Paso 4 (lámina 5/8) - Atención multi-cabeza causal
Aquí el modelo decide "a qué palabras mirar" (Parte 1, lámina 5/8: Q, K y V).

    Q (query) = ¿qué estoy buscando?      -> la pregunta del token actual
    K (key)   = ¿qué información tengo?   -> la "etiqueta" de cada token
    V (value) = ¿qué contenido aporto?    -> lo que se combina si el token es relevante

    Atención(Q, K, V) = softmax(Q · Kᵀ / √d_head) · V

Es "causal": cada token solo puede mirar a los anteriores y a sí mismo, nunca al futuro,
porque el modelo aprende a predecir la siguiente palabra.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context, dropout=0.0, flash=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        # Proyecciones lineales que convierten cada embedding en su Q, K y V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # une la salida de todas las cabezas
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # flash=True usa la versión optimizada de PyTorch (mismo cálculo, más rápida y con menos memoria)
        self.flash = flash and hasattr(F, "scaled_dot_product_attention")

        # Máscara causal: matriz triangular inferior (1 = puede mirar, 0 = futuro prohibido)
        mascara = torch.tril(torch.ones(context, context, dtype=torch.bool))
        self.register_buffer("mascara", mascara.view(1, 1, context, context), persistent=False)

    def _separar_cabezas(self, t, B, T):
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, devolver_pesos=False):
        B, T, C = x.shape  # lote, longitud de la secuencia, d_model
        q = self._separar_cabezas(self.q_proj(x), B, T)
        k = self._separar_cabezas(self.k_proj(x), B, T)
        v = self._separar_cabezas(self.v_proj(x), B, T)

        if self.flash and not devolver_pesos:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
            pesos = None
        else:
            # Versión "a mano", paso por paso (la misma fórmula de arriba)
            puntajes = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)        # (B, h, T, T)
            puntajes = puntajes.masked_fill(~self.mascara[:, :, :T, :T], float("-inf"))
            pesos = F.softmax(puntajes, dim=-1)                                  # cada fila suma 1
            y = self.attn_dropout(pesos) @ v                                     # (B, h, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenar las cabezas
        y = self.resid_dropout(self.out_proj(y))
        return (y, pesos) if devolver_pesos else y
