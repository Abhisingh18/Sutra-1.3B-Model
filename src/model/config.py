"""Model configuration for Sutra-1.3B.

Everything about the architecture lives here. Nothing else in the codebase
should hardcode a dimension.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelConfig:
    # ---- vocabulary -------------------------------------------------------
    # 48K: English primary + Devanagari + chat/reasoning special tokens
    # + 32 reserved slots + 4096 slots reserved for future audio tokens.
    vocab_size: int = 48_000

    # ---- shape ------------------------------------------------------------
    n_layers: int = 22
    d_model: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 4          # GQA: 8 query heads share one KV head
    ffn_hidden: int = 5632       # SwiGLU, ~2.75 * d_model, multiple of 256

    # ---- positions --------------------------------------------------------
    max_seq_len: int = 4096      # extended to 8192 in a later phase
    rope_theta: float = 10_000.0

    # ---- regularisation / numerics ---------------------------------------
    norm_eps: float = 1e-5
    dropout: float = 0.0         # no dropout: we are data-limited, not
                                 # overfitting-limited, during pretraining

    # ---- weights ----------------------------------------------------------
    tie_embeddings: bool = False
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        return self.n_heads // self.n_kv_heads

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide by n_kv_heads"

    # ---- bookkeeping ------------------------------------------------------
    def param_count(self) -> dict:
        """Exact parameter count, broken down by component."""
        d, h, kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim

        embed = self.vocab_size * d
        attn = (d * d) + 2 * (d * kv * hd) + (d * d)      # q, k, v, o
        ffn = 3 * d * self.ffn_hidden                      # gate, up, down
        norms = 2 * d                                      # two RMSNorms per block
        per_layer = attn + ffn + norms

        head = 0 if self.tie_embeddings else self.vocab_size * d
        total = embed + self.n_layers * per_layer + d + head

        return {
            "embedding": embed,
            "per_layer_attention": attn,
            "per_layer_ffn": ffn,
            "all_layers": self.n_layers * per_layer,
            "lm_head": head,
            "total": total,
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))


# Smaller configs for validating the whole pipeline before committing GPU-months.
DEBUG = ModelConfig(vocab_size=48_000, n_layers=4, d_model=256, n_heads=8,
                    n_kv_heads=2, ffn_hidden=768, max_seq_len=512)

SMALL_150M = ModelConfig(vocab_size=48_000, n_layers=12, d_model=768, n_heads=12,
                         n_kv_heads=4, ffn_hidden=2048, max_seq_len=2048)

MAIN_1B = ModelConfig()


if __name__ == "__main__":
    for name, cfg in [("DEBUG", DEBUG), ("SMALL_150M", SMALL_150M), ("MAIN_1B", MAIN_1B)]:
        c = cfg.param_count()
        print(f"{name:12s} {c['total']/1e6:9.1f}M params  "
              f"(embed {c['embedding']/1e6:.1f}M, layers {c['all_layers']/1e6:.1f}M)")
