"""Mixture-of-Experts FFN with DeepSeek-V3 style routing.

Instead of one big FFN, each layer holds 20 small "expert" FFNs. A router picks
the top 4 for each token, so a token only pays for 4 of them. Total capacity
goes up; compute per token stays low.

Three design choices, each of which matters:

*Shared expert.* One expert runs for EVERY token, on top of the 4 routed ones.
Some knowledge -- basic grammar, common patterns -- is needed by all tokens.
Without a shared expert, every routed expert has to independently learn that
same baseline, wasting capacity. The shared expert absorbs it so routed experts
can actually specialise.

*Sigmoid scoring, not softmax.* Softmax forces expert scores to compete for a
fixed budget, which entangles them. V3 switched to sigmoid, scoring each expert
independently, then normalises the selected top-k weights.

*Bias-based load balancing, NOT an auxiliary loss.* The classic approach adds a
balancing loss to the objective, which directly fights language modelling and
costs quality. V3 instead keeps a per-expert bias that is added to routing
scores for SELECTION ONLY, and nudges it after each step: overloaded experts get
their bias lowered, underloaded ones raised. No gradient touches it. This is the
cleanest improvement in V3 and you should use it.

The failure mode to watch for is router collapse: early in training all tokens
pile onto 2-3 experts and the rest never learn anything. Log per-expert load
every few hundred steps. Healthy is roughly uniform; an expert stuck near zero
means the bias update rate is too low or layer 0 was not left dense.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single SwiGLU FFN. Identical in form to a dense FFN, just narrower."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Scores experts and selects top-k."""

    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.scoring = cfg.router_scoring
        self.norm_topk_prob = cfg.norm_topk_prob

        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.d_model))
        nn.init.normal_(self.weight, std=cfg.init_std)

        # Load-balancing bias. A buffer, not a parameter: it is updated by an
        # explicit rule after each optimizer step, never by gradient descent.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        # Running token count per expert, consumed and reset by update_bias().
        self.register_buffer("load_counter", torch.zeros(cfg.n_routed_experts))
        # Smoothed load fractions kept for logging. update_bias() zeroes the
        # counter every step, so without this the stats read as all-zero and
        # you lose the one signal that tells you routing is collapsing.
        self.register_buffer("load_ema", torch.zeros(cfg.n_routed_experts))

    def forward(self, x_flat):
        # x_flat: [n_tokens, d_model]
        logits = F.linear(x_flat.float(), self.weight.float())

        if self.scoring == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)

        # Bias affects WHICH experts are chosen, but not how much they weigh.
        # Mixing the bias into the weights would corrupt the model's output.
        _, topk_idx = torch.topk(scores + self.expert_bias, k=self.top_k, dim=-1)
        topk_weight = scores.gather(-1, topk_idx)

        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)

        if self.training:
            with torch.no_grad():
                self.load_counter += torch.bincount(
                    topk_idx.flatten(), minlength=self.n_experts).float()

        return topk_idx, topk_weight.type_as(x_flat)

    @torch.no_grad()
    def update_bias(self, rate: float):
        """Nudge routing toward balance. Call once per optimizer step."""
        total = self.load_counter.sum()
        if total == 0:
            return
        target = total / self.n_experts
        err = self.load_counter - target
        # Sign-based update, as in V3: fixed step size, direction only. This is
        # more stable than scaling by the magnitude of the imbalance.
        self.expert_bias -= rate * torch.sign(err)

        frac = self.load_counter / total
        self.load_ema.mul_(0.9).add_(frac, alpha=0.1)
        self.load_counter.zero_()

    @torch.no_grad()
    def load_stats(self):
        # Prefer the live counter (before the first update), fall back to the EMA.
        src = self.load_counter if self.load_counter.sum() > 0 else self.load_ema
        total = src.sum()
        if total == 0:
            return {"max_frac": 0.0, "min_frac": 0.0, "dead": 0}
        frac = src / total
        return {
            "max_frac": frac.max().item(),
            "min_frac": frac.min().item(),
            "dead": int((frac < 1e-4).sum().item()),
        }


class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.top_k = cfg.top_k
        self.n_routed = cfg.n_routed_experts

        self.router = Router(cfg)
        self.experts = nn.ModuleList([
            Expert(cfg.d_model, cfg.moe_intermediate)
            for _ in range(cfg.n_routed_experts)
        ])
        self.shared_experts = nn.ModuleList([
            Expert(cfg.d_model, cfg.moe_intermediate)
            for _ in range(cfg.n_shared_experts)
        ])

    def forward(self, x):
        b, s, d = x.shape
        x_flat = x.view(-1, d)
        topk_idx, topk_weight = self.router(x_flat)

        out = torch.zeros_like(x_flat)

        # Gather-compute-scatter, one expert at a time. Every expert sees a
        # contiguous batch of just its own tokens, so each call is one dense
        # matmul rather than a masked full-width one. This loop is the standard
        # single-device implementation; multi-device MoE replaces it with
        # all-to-all, which we do not need here since all 21 experts fit
        # comfortably on one GPU.
        flat_idx = topk_idx.view(-1)
        flat_weight = topk_weight.view(-1)
        token_idx = torch.arange(x_flat.shape[0], device=x.device
                                 ).repeat_interleave(self.top_k)

        for e in range(self.n_routed):
            sel = (flat_idx == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = token_idx[sel]
            y = self.experts[e](x_flat[tok])
            out.index_add_(0, tok, y * flat_weight[sel].unsqueeze(-1))

        # Shared experts run unconditionally on every token.
        for se in self.shared_experts:
            out = out + se(x_flat)

        return out.view(b, s, d)
