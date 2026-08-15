"""One transformer block.

The entire model is this block, repeated 22 times. Two things happen:

    x = x + Attention(RMSNorm(x))     # tokens talk to each other
    x = x + FFN(RMSNorm(x))           # each token thinks on its own

Two design details carry more weight than they appear to:

*Residual*: each block *adds* to the stream rather than replacing it, so every
layer only has to learn a small correction. Without this, gradients cannot
travel back through 22 layers and training simply fails.

*Pre-norm*: normalisation happens on the way *into* each sub-block, not after
it. Post-norm transformers diverge once they get deep. Pre-norm is what keeps
a multi-week training run alive.
"""

import torch.nn as nn

from .norm import RMSNorm
from .attention import Attention
from .ffn import SwiGLU


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None, offset: int = 0):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache, offset)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
