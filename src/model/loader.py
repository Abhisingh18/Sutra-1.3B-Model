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
    # A quantized checkpoint holds int8 weights and scales where the built
    # model has float Linears, so the modules have to be swapped BEFORE the
    # load -- otherwise every quantized layer reports as an unexpected key and
    # the model silently keeps its random initialisation.
    if "quantization" in ck:
        from ..quantize import quantize_model
        quantize_model(model, skip=tuple(ck["quantization"].get("skipped", ())))
    model.load_state_dict(ck["model"], strict=strict)
    return model.to(device), cfg, ck


def _is_quantized(model) -> bool:
    return any(type(m).__name__ == "QuantLinear" for m in model.modules())


def describe(model, cfg) -> str:
    # num_params() walks parameters, and a quantized weight is a BUFFER. Left
    # alone this reports a 1.32B model as 0.099B -- the count silently drops
    # every layer that was quantized, which is nearly all of them.
    tag = " int8" if _is_quantized(model) else ""
    if isinstance(model, AbhiMoE):
        n = model.num_params() + sum(
            b.numel() for name, b in model.named_buffers()
            if name.endswith("qweight"))
        return (f"MoE {n/1e9:.3f}B total / "
                f"{model.num_active_params()/1e9:.3f}B active "
                f"({cfg.n_routed_experts}+{cfg.n_shared_experts} experts, "
                f"top-{cfg.top_k}, MLA{tag})")
    return f"dense {model.num_params()/1e9:.3f}B (GQA{tag})"
