"""The full 1B MoE model: MLA attention + MoE FFN, DeepSeek style."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_config import MoEModelConfig
from .norm import RMSNorm
from .rope import build_rope_cache
from .mla import MLA
from .moe import MoEFFN, Expert


class MoEBlock(nn.Module):
    """One layer. Structurally identical to a dense block -- only the FFN differs.

    `use_moe=False` gives a dense SwiGLU FFN. The first layer uses this: routing
    decisions made on raw embeddings are near-random, and letting the router
    loose there is the most reliable way to trigger expert collapse in the first
    few hundred steps.
    """

    def __init__(self, cfg, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.use_moe = use_moe
        self.ffn = (MoEFFN(cfg) if use_moe
                    else Expert(cfg.d_model, cfg.dense_intermediate))

    def forward(self, x, cos, sin, kv_cache=None, offset: int = 0):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache, offset)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class AbhiMoE(nn.Module):
    def __init__(self, cfg: MoEModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            MoEBlock(cfg, use_moe=(i >= cfg.first_k_dense))
            for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE table is built over the rope half only -- the nope half never
        # sees position at all.
        cos, sin = build_rope_cache(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        # Residual-scaled init on the two output projections of every block,
        # so the residual stream's variance does not grow with depth.
        scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scale)
            if block.use_moe:
                for e in list(block.ffn.experts) + list(block.ffn.shared_experts):
                    nn.init.normal_(e.down_proj.weight, mean=0.0, std=scale)
            else:
                nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # Sequence positions per chunk when computing the loss. See _chunked_loss.
    LOSS_CHUNK = 512

    def _chunked_loss(self, x, targets):
        """Cross-entropy without ever materialising the full logits tensor.

        The naive version computes logits for every position at once and casts
        them to fp32 for the softmax. At batch 8, seq 4096, vocab 48000 that
        single fp32 tensor is 8*4096*48000*4 = 6.3 GB, on top of a 3.1 GB bf16
        copy -- which is what OOMs a 48GB card even though the model itself only
        needs ~32 GB.

        Slicing the sequence into chunks keeps peak logits memory at
        chunk/seq of that, so 512 positions costs ~0.8 GB instead of 6.3 GB.
        The result is numerically identical: cross-entropy is a mean over
        tokens, so summing per-chunk sums and dividing by the total count at
        the end gives exactly the same value.
        """
        b, s, _ = x.shape
        flat_x = x.view(b * s, -1)
        flat_t = targets.reshape(b * s)

        total_loss = x.new_zeros((), dtype=torch.float32)
        total_count = 0
        step = self.LOSS_CHUNK * b

        for i in range(0, b * s, step):
            xc = flat_x[i:i + step]
            tc = flat_t[i:i + step]
            logits = self.lm_head(xc).float()
            n = int((tc != -100).sum())
            if n == 0:
                continue
            # reduction="sum" so chunks of unequal valid-token counts combine
            # correctly; a mean-of-means would silently weight them wrong.
            total_loss = total_loss + F.cross_entropy(
                logits, tc, ignore_index=-100, reduction="sum")
            total_count += n

        return total_loss / max(total_count, 1)

    def forward(self, input_ids, targets=None, kv_caches=None, offset: int = 0,
                return_logits: bool = False):
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
            # return_logits=True gives logits for EVERY position, which DPO
            # needs to score a whole sequence. The default returns only the
            # last position, which is all generation needs and far cheaper.
            logits = self.lm_head(x if return_logits else x[:, -1:, :])
            return logits, new_caches

        loss = self._chunked_loss(x, targets)
        # Training never uses the logits, and returning them would defeat the
        # memory saving above.
        return (self.lm_head(x) if return_logits else None), loss

    # ---- MoE bookkeeping --------------------------------------------------

    def update_router_bias(self):
        """Call once after every optimizer step. This IS the load balancing.

        Under DDP each rank only sees its own shard of the batch, so its load
        counters describe local routing, not global. Summing them across ranks
        first means every rank derives the SAME bias from the SAME global
        counts -- otherwise the ranks' routing tables drift apart and the model
        that gets checkpointed routes differently from the one that trained.
        """
        import torch.distributed as dist
        routers = [b.ffn.router for b in self.blocks if b.use_moe]
        if not routers:
            return

        if dist.is_available() and dist.is_initialized():
            # One flat all-reduce for all layers rather than 15 small ones.
            flat = torch.cat([r.load_counter for r in routers])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off = 0
            for r in routers:
                n = r.load_counter.numel()
                r.load_counter.copy_(flat[off:off + n])
                off += n

        for r in routers:
            r.update_bias(self.cfg.router_bias_update_rate)

    def router_stats(self):
        """Aggregate load health across layers. Watch `dead` and `max_frac`.

        With 20 experts, uniform load is 5% each. A max_frac above ~0.20 or any
        dead expert means routing is collapsing and the run needs attention.
        """
        stats = [b.ffn.router.load_stats() for b in self.blocks if b.use_moe]
        if not stats:
            return {}
        return {
            "max_frac": max(s["max_frac"] for s in stats),
            "min_frac": min(s["min_frac"] for s in stats),
            "dead": sum(s["dead"] for s in stats),
        }

    # ---- accounting -------------------------------------------------------

    def num_params(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        if trainable_only:
            ps = (p for p in ps if p.requires_grad)
        return sum(p.numel() for p in ps)

    def num_active_params(self) -> int:
        return self.cfg.param_count()["active"]

    def flops_per_token(self) -> float:
        return self.cfg.flops_per_token()

    def configure_optimizer(self, lr, weight_decay, betas, device_type="cuda"):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8,
                 top_k=50, top_p=0.95, eos_id=None,
                 min_new_tokens=0, repetition_penalty=1.0,
                 no_repeat_ngram_size=0):
        """Sample a continuation.

        `min_new_tokens` exists because this model ends turns far too early:
        asked "what is ai" it answers "Artificial Intelligence" and then emits
        <|end_turn|> with probability 0.83. Blocking the stop token for the
        first N steps forces it past the one-line answer into an explanation.

        `repetition_penalty` is the other half of that trade: once forced to
        keep going, a model this small falls into loops, so tokens it has
        already emitted get their scores pushed down.

        `no_repeat_ngram_size` catches what the penalty cannot. Asked "what is
        ai" this model emitted "A computer is a system of computers." five times
        in a row: every token in that sentence is common enough that a per-token
        penalty barely moves it, but the n-gram never repeats if the whole
        continuation is blocked.
        """
        self.eval()
        caches = [(None, None)] * self.cfg.n_layers
        offset = 0
        cur = input_ids
        prompt_len = input_ids.shape[1]

        for i in range(max_new_tokens):
            logits, caches = self.forward(cur, kv_caches=caches, offset=offset)
            offset += cur.shape[1]
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(logits.shape[0]):
                    seen = torch.unique(input_ids[b, prompt_len:])
                    if seen.numel():
                        s = logits[b, seen]
                        # Negative scores must be multiplied, not divided, or
                        # the "penalty" would make them more likely instead.
                        logits[b, seen] = torch.where(
                            s > 0, s / repetition_penalty, s * repetition_penalty)

            n = no_repeat_ngram_size
            if n > 1 and input_ids.shape[1] >= n:
                for b in range(logits.shape[0]):
                    seq = input_ids[b].tolist()
                    prefix = tuple(seq[-(n - 1):])
                    # Any token that already followed this exact prefix would
                    # complete a repeated n-gram, so rule it out.
                    for j in range(len(seq) - n + 1):
                        if tuple(seq[j:j + n - 1]) == prefix:
                            logits[b, seq[j + n - 1]] = float("-inf")

            # Block the stop token until the reply has some substance.
            if eos_id is not None and i < min_new_tokens:
                logits[:, eos_id] = float("-inf")

            logits = logits / max(temperature, 1e-5)

            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None and 0 < top_p < 1.0:
                # Nucleus sampling. This was accepted as an argument but never
                # applied, so every reply was really plain top-k 50 -- far too
                # loose at this scale, and where the off-topic tokens came from.
                srt, idx = torch.sort(logits, descending=True, dim=-1)
                sp = F.softmax(srt, dim=-1)
                # Keep the token that crosses p, drop everything after it.
                drop = (torch.cumsum(sp, dim=-1) - sp) > top_p
                srt = srt.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            cur = nxt

            if eos_id is not None and (nxt == eos_id).all():
                break

        return input_ids
