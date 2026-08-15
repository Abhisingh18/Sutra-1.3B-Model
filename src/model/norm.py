"""RMSNorm — root mean square layer normalisation.

LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm
skips the mean subtraction entirely and just divides by the root-mean-square.
Quality is the same and it is roughly 10% faster, which is why every modern
LLM uses it.

This is the component that keeps a 22-layer stack numerically stable for
weeks of training. It is worth understanding rather than treating as boilerplate.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # One learnable scale per channel. Initialised to 1.0 so the layer
        # starts as a pure normalisation and learns to deviate from there.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(mean(x^2)) — computed in fp32 even under bf16 autocast, because
        # squaring bf16 values close to zero underflows and produces NaNs.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight
