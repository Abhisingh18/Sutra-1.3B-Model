"""Grouped-Query Attention.

This is where tokens exchange information. Every token builds a Query ("what am
I looking for"), a Key ("what do I offer") and a Value ("here is my actual
content"). Queries are matched against Keys, and each token pulls in a weighted
mixture of the Values it matched.

GQA: we keep 32 query heads but only 4 key/value heads, with 8 query heads
sharing each KV head. Quality is nearly identical to full multi-head attention,
but the KV cache during generation shrinks 8x -- which is the single biggest
memory saving available at inference time.

Causal masking means a token may only attend to itself and earlier tokens.
Without it the model could peek at the answer and would learn nothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads.

    [batch, n_kv_heads, seq, head_dim] -> [batch, n_kv_heads * n_rep, seq, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, s, d)
            .reshape(b, n_kv * n_rep, s, d))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.dropout = cfg.dropout

        # No biases anywhere -- they cost parameters and buy nothing at this scale.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, offset: int = 0):
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Inject position information by rotation.
        q, k = apply_rope(q, k, cos, sin, offset=offset)

        # During generation, prepend the keys/values we already computed.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # Share each KV head across its group of query heads.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # PyTorch dispatches this to FlashAttention when the shapes and dtype
        # allow it, which keeps the seq x seq matrix out of HBM entirely.
        # is_causal handles masking for us during training; during single-token
        # generation there is nothing to mask, so we turn it off.
        is_causal = kv_cache is None or s > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out), new_cache
