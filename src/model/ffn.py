"""SwiGLU feed-forward network.

After attention has moved information between tokens, the FFN processes each
token independently. It projects 2048 -> 5632, does nonlinear work in that
wider space, and projects back down.

This block holds roughly 3.7x more parameters than attention does, and it is
where most of the model's actual *knowledge* is stored. Attention routes
information; the FFN remembers things.

SwiGLU uses three matrices instead of the classic two. One path (`gate`) passes
through SiLU and decides *how much* signal to let through; the other (`up`)
carries *what* the signal is. They are multiplied elementwise. Because there are
three matrices, the hidden width is set to ~2.7x d_model rather than 4x, so the
parameter count stays comparable to a classic FFN.
"""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
