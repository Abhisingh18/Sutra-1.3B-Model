"""Transformers-facing config for Sutra-1.3B.

This exists so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
works. The architecture is unchanged -- this is a thin translation layer between
`PretrainedConfig` and the `MoEModelConfig` dataclass the model actually takes.
"""

from transformers import PretrainedConfig


class SutraConfig(PretrainedConfig):
    model_type = "sutra-moe"

    def __init__(
        self,
        vocab_size: int = 48_000,
        n_layers: int = 16,
        d_model: int = 1024,
        max_seq_len: int = 4096,
        n_heads: int = 16,
        kv_lora_rank: int = 256,
        q_lora_rank=None,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        n_routed_experts: int = 48,
        n_shared_experts: int = 1,
        top_k: int = 4,
        moe_intermediate: int = 512,
        dense_intermediate: int = 2816,
        first_k_dense: int = 1,
        router_bias_update_rate: float = 1e-3,
        router_scoring: str = "sigmoid",
        norm_topk_prob: bool = True,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
        init_std: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.moe_intermediate = moe_intermediate
        self.dense_intermediate = dense_intermediate
        self.first_k_dense = first_k_dense
        self.router_bias_update_rate = router_bias_update_rate
        self.router_scoring = router_scoring
        self.norm_topk_prob = norm_topk_prob
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.init_std = init_std

        # Aliases under the names transformers and its ecosystem look for.
        # Tooling reaches for config.num_hidden_layers and config.hidden_size
        # regardless of what a model calls them internally.
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_heads
        self.intermediate_size = moe_intermediate
        self.max_position_embeddings = max_seq_len

        super().__init__(**kwargs)
