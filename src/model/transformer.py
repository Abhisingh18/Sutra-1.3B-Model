"""The full model: embedding -> 22 blocks -> norm -> lm_head."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .block import Block
from .rope import build_rope_cache


class AbhiLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE tables are constants, not parameters. persistent=False keeps them
        # out of the checkpoint -- they are cheap to rebuild and this lets us
        # change max_seq_len later without breaking checkpoint loading.
        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        # Residual-scaled init: every block adds into the residual stream, so
        # with 22 blocks the stream's variance would grow ~22x without this.
        # Scaling the two output projections by 1/sqrt(2 * n_layers) keeps the
        # stream's magnitude stable at initialisation.
        scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids, targets=None, kv_caches=None, offset: int = 0):
        """
        input_ids : [batch, seq]
        targets   : [batch, seq] or None. -100 marks positions to ignore,
                    which is how SFT masks out the user's turns.
        """
        x = self.embed(input_ids)

        cos, sin = self.rope_cos, self.rope_sin
        new_caches = [] if kv_caches is not None else None

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, c = block(x, cos, sin, cache, offset)
            if new_caches is not None:
                new_caches.append(c)

        x = self.final_norm(x)

        if targets is None:
            # Generation: only the last position matters, so skip the rest of
            # the lm_head. At 48K vocab this is a large saving per step.
            logits = self.lm_head(x[:, -1:, :])
            return logits, new_caches

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    # ---- helpers ----------------------------------------------------------

    def num_params(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        if trainable_only:
            ps = (p for p in ps if p.requires_grad)
        return sum(p.numel() for p in ps)

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token, used to compute MFU during training."""
        c = self.cfg
        n = self.num_params()
        # 6ND for the dense matmuls, plus the attention score/value matmuls
        # which the parameter count does not capture.
        attn_flops = 12 * c.n_layers * c.d_model * c.max_seq_len
        return 6 * n + attn_flops

    def configure_optimizer(self, lr, weight_decay, betas, device_type="cuda"):
        """AdamW with decay applied only to matrices, not to norms/embeddings.

        Weight-decaying 1-D parameters (RMSNorm scales) actively hurts; this
        split is standard practice and worth keeping.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # Fused AdamW is meaningfully faster and available on all modern CUDA builds.
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8,
                 top_k=50, top_p=0.95, eos_id=None):
        """Sampling with a KV cache. Kept simple and readable, not maximally fast."""
        self.eval()
        caches = [None] * self.cfg.n_layers
        offset = 0
        cur = input_ids

        for _ in range(max_new_tokens):
            logits, caches = self.forward(cur, kv_caches=caches, offset=offset)
            offset += cur.shape[1]

            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            cur = nxt

            if eos_id is not None and (nxt == eos_id).all():
                break

        return input_ids
