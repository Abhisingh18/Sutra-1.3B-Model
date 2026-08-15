"""RoPE — Rotary Position Embeddings.

The model has no inherent notion of word order. RoPE injects it by *rotating*
each query and key vector by an angle proportional to its position. Because a
dot product between two rotated vectors depends only on the difference of their
angles, attention ends up seeing *relative* distance rather than absolute
position. That is what makes context extension possible later: raise
`rope_theta`, fine-tune briefly, and the same weights handle a longer window.

Each pair of adjacent channels (2i, 2i+1) is treated as a point in a 2D plane
and rotated. Low-index pairs rotate fast (they encode nearby positions), high
index pairs rotate slowly (they encode long-range position).
"""

import torch


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10_000.0,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables once, reuse for every forward pass.

    Returns two tensors of shape [max_seq_len, head_dim].
    """
    # Frequency for each channel pair: theta^(-2i/d)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    # [seq, head_dim/2] — the angle for every (position, channel-pair)
    angles = torch.outer(positions, inv_freq)

    # Duplicate so the table lines up with rotate_half's layout below.
    emb = torch.cat([angles, angles], dim=-1)          # [seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate: [a, b] -> [-b, a]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0):
    """Rotate q and k in place-ish.

    q: [batch, n_heads, seq, head_dim]
    k: [batch, n_kv_heads, seq, head_dim]
    `offset` is the position of the first token — non-zero during cached
    generation, where we feed one token at a time but it sits at position N.
    """
    seq = q.shape[-2]
    c = cos[offset:offset + seq].unsqueeze(0).unsqueeze(0)   # [1, 1, seq, hd]
    s = sin[offset:offset + seq].unsqueeze(0).unsqueeze(0)

    q_out = (q * c) + (rotate_half(q) * s)
    k_out = (k * c) + (rotate_half(k) * s)
    return q_out.type_as(q), k_out.type_as(k)
