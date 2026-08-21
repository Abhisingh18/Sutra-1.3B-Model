"""INT8 weight-only quantization for the trained checkpoint.

What this is, and what it is not. DeepSeek-V3's FP8 work is a TRAINING
technique -- the forward and backward run in FP8 with fine-grained scaling,
which buys memory and speed across the run. That opportunity closes when
training ends. This is the other kind: post-training quantization, applied to
finished weights to make them cheaper to serve.

Weight-only, symmetric, per-output-channel. Each row of a weight matrix gets
its own scale, because rows differ in range by orders of magnitude and one
scale for the whole tensor throws away most of the resolution:

    scale[i] = max(|W[i, :]|) / 127
    q[i, :]  = round(W[i, :] / scale[i])      -> int8

Activations stay bf16. That is the deliberate choice: quantizing activations
too would buy speed on hardware with int8 kernels, but it is where accuracy
actually goes, and this model has none to spare.

So be clear about the trade being made: this saves MEMORY, not time. The
weight is dequantized back to bf16 before the matmul, so arithmetic runs at
the same speed it did before -- roughly 2x less to store and load, the same
FLOPs to compute.
"""

import argparse
import dataclasses
import os

import torch
import torch.nn as nn


class QuantLinear(nn.Module):
    """A Linear whose weight is stored int8 and dequantized on use."""

    def __init__(self, in_features, out_features, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.register_buffer(
            "qweight", torch.zeros((out_features, in_features),
                                   dtype=torch.int8, device=device))
        self.register_buffer(
            "scale", torch.zeros((out_features, 1), dtype=dtype, device=device))

    @classmethod
    def from_linear(cls, lin):
        w = lin.weight.data
        q = cls(lin.in_features, lin.out_features,
                device=w.device, dtype=w.dtype)
        # Symmetric per-row: no zero point, so an exact zero stays exact.
        amax = w.abs().amax(dim=1, keepdim=True).float()
        # A row of all zeros would divide by zero and produce NaN weights --
        # silently, and only for that row.
        scale = (amax / 127.0).clamp(min=1e-12)
        q.qweight.copy_(torch.round(w.float() / scale).clamp(-127, 127).to(torch.int8))
        q.scale.copy_(scale.to(w.dtype))
        return q

    def forward(self, x):
        w = self.qweight.to(self.dtype) * self.scale
        return torch.nn.functional.linear(x, w)


def quantize_model(model, skip=("lm_head",)):
    """Replace every nn.Linear with a QuantLinear, in place.

    `lm_head` is skipped by default. It maps into 48,000 vocabulary logits
    that are compared against each other, so error there lands directly on
    which token gets sampled -- and at 49M parameters it is 4% of the model,
    a poor trade for the risk.
    """
    replaced = 0

    def walk(module, prefix=""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, nn.Linear):
                if any(s in path for s in skip):
                    continue
                setattr(module, name, QuantLinear.from_linear(child))
                replaced += 1
            else:
                walk(child, path + ".")

    walk(model)
    return replaced


def state_dict_bytes(sd):
    return sum(t.numel() * t.element_size()
               for t in sd.values() if torch.is_tensor(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dpo/dpo_epoch_0.pt")
    ap.add_argument("--out", default="checkpoints/dpo/dpo_epoch_0.int8.pt")
    ap.add_argument("--keep-lm-head", action="store_true", default=True)
    ap.add_argument("--quantize-lm-head", dest="keep_lm_head",
                    action="store_false")
    args = ap.parse_args()

    from src.model.loader import load_checkpoint

    # CPU on purpose: quantizing needs no GPU, and the serving card must stay
    # free for the model that is answering users right now.
    model, mcfg, _ = load_checkpoint(args.ckpt, "cpu")
    model = model.to(torch.bfloat16).eval()

    before = state_dict_bytes(model.state_dict())
    skip = ("lm_head",) if args.keep_lm_head else ()
    n = quantize_model(model, skip=skip)
    after = state_dict_bytes(model.state_dict())

    # "model_config" is the key the loader reads -- writing anything else
    # produces a file that saves fine and cannot be loaded.
    torch.save({"model": model.state_dict(),
                "model_config": dataclasses.asdict(mcfg),
                "quantization": {"scheme": "int8-weight-only",
                                 "granularity": "per-output-channel",
                                 "skipped": list(skip)}},
               args.out)

    print(f"quantized {n} linear layers")
    print(f"  before : {before / 1e9:6.2f} GB")
    print(f"  after  : {after / 1e9:6.2f} GB  ({before / after:.2f}x smaller)")
    print(f"  written: {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB on disk)")


if __name__ == "__main__":
    main()
