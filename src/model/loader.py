"""Load a checkpoint without needing to know which architecture wrote it.

Checkpoints store their own `model_config` dict. MoE configs carry keys the
dense config does not, so we can tell them apart and rebuild the right class.
This keeps sft.py / dpo.py / chat.py architecture-agnostic.
"""

import torch

from .config import ModelConfig
from .transformer import AbhiLLM
from .moe_config import MoEModelConfig
from .moe_transformer import AbhiMoE


def is_moe_config(d: dict) -> bool:
    return "n_routed_experts" in d


def build_from_config_dict(d: dict):
    if is_moe_config(d):
        cfg = MoEModelConfig(**d)
        return AbhiMoE(cfg), cfg
    cfg = ModelConfig(**d)
    return AbhiLLM(cfg), cfg


def load_checkpoint(path: str, device="cpu", strict: bool = True):
    """Returns (model, config, raw_checkpoint).

    Accepts both formats. A .pt is a pickle carrying its own `model_config`,
    which is what the training scripts write. A .safetensors holds only tensors,
    so the config has to come from config.json beside it -- that is the format
    published on the Hub, because safetensors needs no pickle to load and is
    the filename pattern the Hub counts downloads for.
    """
    if path.endswith(".safetensors"):
        import json
        import os
        from safetensors.torch import load_file

        cfg_path = os.path.join(os.path.dirname(path), "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        # config.json carries display fields the dataclass does not accept.
        fields = set(MoEModelConfig.__dataclass_fields__) | \
            set(ModelConfig.__dataclass_fields__)
        d = {k: v for k, v in raw.items() if k in fields}
        model, cfg = build_from_config_dict(d)
        model.load_state_dict(load_file(path), strict=strict)
        return model.to(device), cfg, {"model_config": d}

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, cfg = build_from_config_dict(ck["model_config"])
    model.load_state_dict(ck["model"], strict=strict)
    return model.to(device), cfg, ck


def describe(model, cfg) -> str:
    if isinstance(model, AbhiMoE):
        return (f"MoE {model.num_params()/1e9:.3f}B total / "
                f"{model.num_active_params()/1e9:.3f}B active "
                f"({cfg.n_routed_experts}+{cfg.n_shared_experts} experts, "
                f"top-{cfg.top_k}, MLA)")
    return f"dense {model.num_params()/1e9:.3f}B (GQA)"
