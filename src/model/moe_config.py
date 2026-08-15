"""1B DeepSeek-inspired MoE configuration.

Design target: ~1.0B total parameters, ~0.39B active per token.

Understand the tradeoff before committing to this. MoE buys capacity at fixed
FLOPs -- you get more parameters for the same compute. But a 1B-*total* MoE is
NOT stronger than a 1B-*dense* model; it is roughly as strong as a ~0.5B dense
model while running as fast as a 0.39B one. The win here is training speed
(~4 weeks instead of ~7 for 100B tokens), not quality.

MoE also needs more tokens than dense to reach its potential, because each
expert only sees the fraction of tokens routed to it. Budget accordingly.

Two structural choices are load-bearing and come straight from DeepSeek:
  * layer 0 stays dense -- routing on raw embeddings collapses early in training
  * load balancing is bias-based, not auxiliary-loss-based (see MoEConfig below)
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class MoEModelConfig:
    # ---- vocabulary -------------------------------------------------------
    vocab_size: int = 48_000

    # ---- shape ------------------------------------------------------------
    n_layers: int = 26
    d_model: int = 1024
    max_seq_len: int = 4096

    # ---- MLA (Multi-head Latent Attention) --------------------------------
    # K and V are compressed into a single `kv_lora_rank`-wide latent vector and
    # only that is cached, instead of full per-head K and V.
    #
    # The nope/rope split exists because RoPE does not commute with the
    # low-rank compression: you cannot rotate a compressed key and still
    # reconstruct it. So each head is split -- `qk_nope_head_dim` dimensions
    # carry compressed content with no position, and `qk_rope_head_dim`
    # dimensions carry position and are cached separately, uncompressed.
    # This is the subtlest part of MLA and the easiest place to get it wrong.
    n_heads: int = 16
    kv_lora_rank: int = 256
    q_lora_rank: int | None = None      # None = project Q directly; at d_model
                                        # 1024 the extra compression is not worth it
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # ---- MoE --------------------------------------------------------------
    n_routed_experts: int = 20
    n_shared_experts: int = 1           # always active; absorbs the knowledge
                                        # every token needs, so routed experts
                                        # are free to specialise
    top_k: int = 4
    moe_intermediate: int = 512         # per-expert FFN width
    dense_intermediate: int = 2816      # width of the dense layers (2.75 * d_model)
    first_k_dense: int = 1              # first N layers use a dense FFN

    # Bias-based (auxiliary-loss-free) load balancing, DeepSeek-V3 style.
    # Each expert carries a bias added to its routing score. After every step,
    # overloaded experts get their bias nudged down and underloaded ones up.
    # No gradient flows through this -- it only shifts routing. The older
    # auxiliary-loss approach fights the language-modelling objective and
    # measurably costs quality; do not use it.
    router_bias_update_rate: float = 1e-3
    router_scoring: str = "sigmoid"      # V3 uses sigmoid, not softmax
    norm_topk_prob: bool = True

    # ---- numerics ---------------------------------------------------------
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = False
    init_std: float = 0.02

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __post_init__(self):
        assert self.top_k <= self.n_routed_experts
        assert self.first_k_dense < self.n_layers

    # ------------------------------------------------------------------
    # parameter accounting
    # ------------------------------------------------------------------
    def attn_params(self) -> int:
        d, h = self.d_model, self.n_heads
        if self.q_lora_rank:
            q = d * self.q_lora_rank + self.q_lora_rank * h * self.qk_head_dim
        else:
            q = d * h * self.qk_head_dim
        kv_a = d * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_b = self.kv_lora_rank * h * (self.qk_nope_head_dim + self.v_head_dim)
        o = h * self.v_head_dim * d
        # RMSNorm applied to the compressed latent before expansion.
        kv_a_norm = self.kv_lora_rank
        q_norm = self.q_lora_rank if self.q_lora_rank else 0
        return q + kv_a + kv_b + o + kv_a_norm + q_norm

    def dense_ffn_params(self) -> int:
        return 3 * self.d_model * self.dense_intermediate

    def expert_params(self) -> int:
        return 3 * self.d_model * self.moe_intermediate

    def moe_ffn_params(self) -> int:
        experts = (self.n_routed_experts + self.n_shared_experts) * self.expert_params()
        router = self.d_model * self.n_routed_experts
        return experts + router

    def param_count(self) -> dict:
        embed = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model

        attn = self.attn_params()
        n_dense = self.first_k_dense
        n_moe = self.n_layers - n_dense

        dense_layers = n_dense * (attn + self.dense_ffn_params() + 2 * self.d_model)
        moe_layers = n_moe * (attn + self.moe_ffn_params() + 2 * self.d_model)

        total = embed + dense_layers + moe_layers + self.d_model + head

        # Active: shared experts always fire, plus top_k routed ones.
        active_moe_ffn = (self.n_shared_experts + self.top_k) * self.expert_params()
        active = (embed + head + self.d_model
                  + n_dense * (attn + self.dense_ffn_params())
                  + n_moe * (attn + active_moe_ffn))

        return {
            "attn_per_layer": attn,
            "moe_ffn_per_layer": self.moe_ffn_params(),
            "dense_layers": dense_layers,
            "moe_layers": moe_layers,
            "embedding": embed,
            "lm_head": head,
            "total": total,
            "active": active,
            "sparsity": total / active,
        }

    def flops_per_token(self) -> float:
        """Uses ACTIVE parameters -- that is the whole point of MoE."""
        active = self.param_count()["active"]
        attn_flops = 12 * self.n_layers * self.d_model * self.max_seq_len
        return 6 * active + attn_flops

    def training_memory_gb(self, micro_batch: int = 8,
                           activation_checkpointing: bool = True) -> dict:
        """Per-GPU training memory under DDP + bf16 autocast.

        This is the check that decides how many experts you can afford. With
        MoE, adding experts costs MEMORY but not COMPUTE -- so on a box with
        spare VRAM, experts are close to free quality. This function tells you
        where the ceiling is.

        AdamW keeps fp32 master weights plus two moments; together with fp32
        gradients that is 16 bytes per parameter, and it scales with TOTAL
        parameters, not active ones.
        """
        total = self.param_count()["total"]
        optimizer_gb = total * 16 / 1e9          # master + grads + m + v
        weights_bf16_gb = total * 2 / 1e9

        # Activations. With checkpointing only block boundaries are kept, so
        # cost is ~= layers * batch * seq * d_model * 2 bytes, plus one block's
        # worth of internals recomputed during backward.
        act = self.n_layers * micro_batch * self.max_seq_len * self.d_model * 2
        if not activation_checkpointing:
            act *= 8                              # rough: all intermediates kept
        activations_gb = act / 1e9

        # Routing scatter/gather buffers, logits, NCCL buffers, fragmentation.
        overhead_gb = 3.0 + micro_batch * self.max_seq_len * self.vocab_size * 2 / 1e9

        total_gb = optimizer_gb + weights_bf16_gb + activations_gb + overhead_gb
        return {
            "optimizer": optimizer_gb,
            "weights_bf16": weights_bf16_gb,
            "activations": activations_gb,
            "overhead": overhead_gb,
            "total": total_gb,
        }

    def kv_cache_bytes_per_token(self) -> int:
        """MLA caches only the latent + the decoupled RoPE part."""
        return self.n_layers * (self.kv_lora_rank + self.qk_rope_head_dim) * 2

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


MOE_1B = MoEModelConfig()

# Sized for a 7-day end-to-end budget on 5 GPUs (~5 days of pretraining).
#
# The reasoning: that budget buys ~5.5e19 FLOPs. Spent on MOE_1B it yields only
# ~15B tokens, i.e. ~15 tokens per parameter -- well under Chinchilla's ~20 and
# badly undertrained, so most of the model's capacity would go unused. Spent on
# this smaller config it yields ~22B tokens, ~39 tokens per parameter. The
# smaller model trained properly beats the larger model trained briefly, and it
# is also cheaper to serve. Modern small models (Llama 3.2, Qwen 3) are
# deliberately over-trained this way.
MOE_600M = MoEModelConfig(
    n_layers=16,
    d_model=1024,
    n_routed_experts=16,
    n_shared_experts=1,
    top_k=4,
    moe_intermediate=512,
    first_k_dense=1,
)

# THE ONE TO TRAIN.
#
# Same compute as MOE_600M -- identical active parameters, identical 51K tok/s,
# identical 5 days for 22B tokens -- but 2.3x the total capacity, because on
# this box VRAM is the resource we have spare (5 x 48GB = 240GB) and compute is
# the one we do not. Adding experts costs memory, not FLOPs. That is the entire
# reason to use MoE here.
#
# Why 48 experts and not 96: two ceilings.
#   memory   96 experts needs ~51GB/GPU, over the 48GB limit.
#   data     each expert only sees tokens * top_k / n_experts. At 22B tokens
#            and 48 experts that is ~1.8B tokens per expert, which is enough to
#            train one. At 96 it drops to ~0.9B and experts start to undertrain,
#            so the extra capacity would not be paid for.
# 48 sits just inside both.
MOE_MAIN = MoEModelConfig(
    n_layers=16,
    d_model=1024,
    n_routed_experts=48,
    n_shared_experts=1,
    top_k=4,
    moe_intermediate=512,
    first_k_dense=1,
)

# Scaled-down twin for validating the pipeline. Keep the same expert/routing
# structure -- routing bugs do not show up if you only test with 2 experts.
MOE_DEBUG = MoEModelConfig(
    n_layers=4, d_model=256, n_heads=4,
    kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
    n_routed_experts=8, top_k=2, moe_intermediate=128,
    dense_intermediate=704, max_seq_len=512,
)


def report(cfg: MoEModelConfig, name: str):
    c = cfg.param_count()
    print(f"=== {name} ===")
    print(f"  layers            : {cfg.n_layers} "
          f"({cfg.first_k_dense} dense + {cfg.n_layers - cfg.first_k_dense} MoE)")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  experts           : {cfg.n_routed_experts} routed + "
          f"{cfg.n_shared_experts} shared, top-{cfg.top_k}")
    print()
    print(f"  attn / layer      : {c['attn_per_layer']/1e6:8.3f}M")
    print(f"  moe ffn / layer   : {c['moe_ffn_per_layer']/1e6:8.3f}M")
    print(f"  embedding         : {c['embedding']/1e6:8.3f}M")
    print(f"  lm_head           : {c['lm_head']/1e6:8.3f}M")
    print()
    print(f"  TOTAL params      : {c['total']/1e9:8.3f}B")
    print(f"  ACTIVE params     : {c['active']/1e9:8.3f}B")
    print(f"  sparsity          : {c['sparsity']:8.2f}x")
    print()
    print(f"  FLOPs/token       : {cfg.flops_per_token()/1e9:8.2f}G")
    print(f"  KV cache/token    : {cfg.kv_cache_bytes_per_token()} bytes "
          f"({cfg.kv_cache_bytes_per_token()*4096/1e6:.1f}MB at 4096 ctx)")


if __name__ == "__main__":
    report(MOE_1B, "MOE_1B")
    print()
    report(MOE_DEBUG, "MOE_DEBUG")

    # Compare against the dense model to make the tradeoff explicit.
    print("\n=== dense vs MoE ===")
    from .config import MAIN_1B
    dense_p = MAIN_1B.param_count()["total"]
    moe = MOE_1B.param_count()
    dense_flops = 6 * dense_p
    moe_flops = 6 * moe["active"]
    print(f"  dense total/active: {dense_p/1e9:.3f}B / {dense_p/1e9:.3f}B")
    print(f"  moe   total/active: {moe['total']/1e9:.3f}B / {moe['active']/1e9:.3f}B")
    print(f"  compute ratio     : {dense_flops/moe_flops:.2f}x cheaper per token")
