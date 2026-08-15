"""Multi-head Latent Attention (MLA), as introduced in DeepSeek-V2.

Standard attention caches full K and V for every head: at 16 heads x 64 dims
that is 2048 numbers per token per layer. MLA instead projects the input down
into a single narrow latent vector (256 dims here), caches *that*, and expands
it back to per-head K and V on the fly. The cache shrinks ~3.5x, and unlike GQA
this costs almost nothing in quality because no head is forced to share.

THE ROPE PROBLEM -- this is the part that trips people up.

The obvious design fails. If you compress K into a latent and then apply RoPE to
the reconstructed keys, the rotation depends on absolute position, so the cached
latent is no longer position-agnostic and cannot be reused. RoPE and low-rank
compression simply do not commute.

DeepSeek's fix is to split every head in two:

    qk_nope_head_dim (64)  compressed content, NO position
    qk_rope_head_dim (32)  position-carrying, cached separately, uncompressed

The nope half rides through the latent; the rope half bypasses compression
entirely and is cached raw. Attention scores are computed over the concatenation
of both. The rope half is shared across all heads (MQA-style), so it stays tiny.

Cache per token per layer = kv_lora_rank + qk_rope_head_dim = 256 + 32 = 288
versus 2048 for full multi-head attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import apply_rope


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim

        d = cfg.d_model

        # ---- query path ----------------------------------------------------
        # At d_model=1024 the query projection is small enough that compressing
        # it (q_lora_rank) costs more in complexity than it saves. Larger models
        # do compress it; we keep the option but default to off.
        if cfg.q_lora_rank:
            self.q_a_proj = nn.Linear(d, cfg.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
            self.q_b_proj = nn.Linear(cfg.q_lora_rank,
                                      self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(d, self.n_heads * self.qk_head_dim, bias=False)

        # ---- key/value path ------------------------------------------------
        # One projection produces BOTH the compressed latent and the decoupled
        # RoPE key. Only this output is cached.
        self.kv_a_proj = nn.Linear(d, cfg.kv_lora_rank + self.qk_rope_head_dim,
                                   bias=False)
        # Normalising the latent keeps its scale stable; without it, training
        # is noticeably less well-behaved.
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)

        # Expands the latent back into per-head nope-keys and values.
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, d, bias=False)

        # Scale uses the FULL qk head dim (nope + rope), since scores are
        # computed over the concatenation of both halves.
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(self, x, cos, sin, kv_cache=None, offset: int = 0):
        b, s, _ = x.shape

        # ---- queries -------------------------------------------------------
        if self.cfg.q_lora_rank:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.view(b, s, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- keys/values: compress -----------------------------------------
        kv_a = self.kv_a_proj(x)
        c_kv, k_rope = torch.split(
            kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_norm(c_kv)

        # k_rope is shared by all heads -- one set of position keys, not 16.
        k_rope = k_rope.view(b, s, 1, self.qk_rope_head_dim).transpose(1, 2)

        # ---- cache: store the COMPRESSED form, which is the whole point -----
        new_cache = None
        if kv_cache is not None:
            if kv_cache[0] is not None:
                past_ckv, past_krope = kv_cache
                c_kv = torch.cat([past_ckv, c_kv], dim=1)
                k_rope = torch.cat([past_krope, k_rope], dim=2)
            new_cache = (c_kv, k_rope)

        kv_len = c_kv.shape[1]

        # ---- keys/values: expand -------------------------------------------
        kv = self.kv_b_proj(c_kv)
        kv = kv.view(b, kv_len, self.n_heads,
                     self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ---- position: applied ONLY to the rope halves ----------------------
        # Queries sit at [offset, offset+s); keys span the whole cached range.
        q_rope, _ = apply_rope(q_rope, q_rope, cos, sin, offset=offset)
        _, k_rope = apply_rope(k_rope, k_rope, cos, sin, offset=0)

        # Broadcast the shared rope keys across heads.
        k_rope = k_rope.expand(b, self.n_heads, kv_len, self.qk_rope_head_dim)

        # ---- attention over [nope | rope] -----------------------------------
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        is_causal = kv_cache is None or s > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out), new_cache
