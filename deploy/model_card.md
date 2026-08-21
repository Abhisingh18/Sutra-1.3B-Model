---
license: apache-2.0
language:
  - en
  - hi
library_name: pytorch
pipeline_tag: text-generation
tags:
  - mixture-of-experts
  - moe
  - mla
  - multi-head-latent-attention
  - deepseek
  - from-scratch
  - pretraining
  - custom-architecture
  - pytorch
  - sparse
  - hindi
  - indic
---

# Sutra-1.3B

A 1.32B-parameter **Mixture-of-Experts** language model **pretrained from
scratch** in pure PyTorch, with **Multi-head Latent Attention (MLA)** and
DeepSeek-style **auxiliary-loss-free expert routing** — implemented from the
papers, not adapted from an existing codebase.

Own BPE tokenizer, own streaming data pipeline, own training loop. No
pretrained weights, no `transformers` Trainer, no reference implementation
anywhere in the stack.

**If you are here to learn how a MoE model is trained end to end**, the code is
the point: [github.com/Abhisingh18/Sutra-1.3B-Model](https://github.com/Abhisingh18/Sutra-1.3B-Model)
carries the tokenizer, data prep, pretraining, SFT, DPO, evaluation and serving
— plus the six silent bugs that cost the most time, written up in full.

## Try it without installing anything

**[sutra-1-3-b-model-15co.vercel.app](https://sutra-1-3-b-model-15co.vercel.app)** — a live chat demo running these
weights, with web search and document upload on top.

The inference widget on this page stays empty on purpose: Inference Providers
serve standard `transformers` architectures, and this is a custom MoE with
Multi-head Latent Attention. Same reason `AutoModelForCausalLM` does not work
here — the architecture code ships with the weights, not with `transformers`.

## Quick start

```python
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Abhisingh-18/Sutra-1.3B-Chat", trust_remote_code=True,
    dtype=torch.bfloat16).eval()
tok = Tokenizer.from_file("tokenizer.json")

prompt = ("<|begin_of_text|><|user|>\nWhat is the capital of France?"
          "<|end_turn|>\n<|assistant|>\n")
ids = torch.tensor([tok.encode(prompt).ids])
out = model.generate(ids, max_new_tokens=64, min_new_tokens=12, do_sample=False)
print(tok.decode(out[0].tolist()[ids.shape[1]:]))
```

`trust_remote_code=True` is required: the architecture (MoE + Multi-head Latent
Attention) ships with the weights rather than with `transformers`. The remote
code is verified to produce logits identical to the reference implementation to
7e-06 in fp32.

Or without transformers at all:

```bash
pip install torch tokenizers huggingface_hub
wget https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat/resolve/main/inference.py
python inference.py "Explain photosynthesis in three sentences."
```

That downloads the weights (5.3 GB) and the architecture code and runs. It
works on CPU — because only 0.28B parameters are active per token, CPU
generation runs at about **10 tokens/second on 2 cores**.

Interactive:

```bash
python inference.py
```

In Python:

```python
from inference import build, generate
model, mcfg, tok, device = build()
print(generate(model, mcfg, tok, device, "What is machine learning?"))
```

> **Note:** `min_new_tokens` matters. Left to itself this model often emits
> `<|end_turn|>` immediately and returns an empty string.

## Repository layout

```
├── model.safetensors        default weights — the DPO stage, 5.3 GB
├── config.json              architecture
├── tokenizer.json           48k BPE, English + Devanagari
├── inference.py             run the model
├── src/                     model code (custom arch — see note above)
└── checkpoints/             every training stage, for comparison
    ├── base_pretrained.pt
    ├── sft_epoch_0.pt
    ├── sft_epoch_1.pt
    ├── sft_epoch_2.pt
    ├── dpo_epoch_0.pt
    └── dpo_epoch_0.int8.pt   quantized — 1.42 GB
```

The root holds what you need to run the model. `checkpoints/` is the archive of
each stage, so you can hear the difference each one made instead of taking it
on faith.

| Checkpoint | Stage | Worth loading for |
|---|---|---|
| `base_pretrained.pt` | 18B tokens, no fine-tuning | Text *continuation*. It continues your prompt rather than answering it — the clearest demonstration of what SFT actually does |
| `sft_epoch_0.pt` | SFT, 1 epoch | Comparison |
| `sft_epoch_1.pt` | SFT, 2 epochs | Comparison |
| `sft_epoch_2.pt` | SFT, 3 epochs | Best held-out loss of the three (1.7033) — no overfitting |
| `dpo_epoch_0.pt` | DPO on top of SFT | Same weights as `model.safetensors` |
| `dpo_epoch_0.int8.pt` | INT8 quantized DPO | Half the memory at no measurable cost — see below |

Optimizer state is stripped from all of them, so each is 5.3 GB rather than
15.8 GB. They are for inference, not for resuming training.

To load a different stage:

```bash
SUTRA_CKPT=checkpoints/base_pretrained.pt python inference.py "The capital of France is"
```

## Architecture

| | |
|---|---|
| Parameters | 1.32B total / **0.28B active** (4.7x sparsity) |
| Experts | 48 routed + 1 shared, top-4 |
| Routing | sigmoid scoring, bias-based load balancing |
| Attention | **MLA** (Multi-head Latent Attention), kv_lora_rank 256 |
| Layers | 16 (layer 0 dense, 1-15 MoE) |
| d_model | 1024 |
| Context | 4096 |
| Vocab | 48,000 (English + Devanagari) |

## Training

| Stage | Data | Compute |
|---|---|---|
| Pretraining | 18B tokens (English, Hindi, code, math) | 4x RTX 6000 Ada, 4d 9h |
| SFT | 200K conversations | 18h |
| DPO | 100K preference pairs | 6h |

Pretraining held-out perplexity **15.00**; SFT held-out perplexity **5.49**.

Every dataset behind all three stages — with its exact config, split, text field
and token share, plus the prep script — is published as
**[Sutra-1.3B-Data](https://huggingface.co/datasets/Abhisingh-18/Sutra-1.3B-Data)**,
so the corpus can be rebuilt from the original sources.

## Evaluation

Log-likelihood scoring, 500 examples per task, length-normalised accuracy.

| Task | Random | Base | SFT | DPO |
|---|---|---|---|---|
| HellaSwag | 25.0 | 38.4 | 39.8 | **40.4** |
| ARC-easy | 25.0 | 45.0 | 44.8 | **45.0** |
| PIQA | 50.0 | 62.6 | 65.4 | **65.6** |
| WinoGrande | 50.0 | 50.6 | 49.0 | 49.0 |

Two things worth reading honestly here. ARC-easy and PIQA sit well above chance,
so the model learned real commonsense and not just fluent grammar. WinoGrande
sits *at* chance, which is the clearest signal of what 0.28B active parameters
cannot buy: the pronoun-resolution reasoning that task measures never appeared.

DPO's held-out preference accuracy came out at **47.5%** against a 50% baseline,
so the alignment stage did not generalise — the 66% reported during training was
measured on training batches. The SFT and DPO checkpoints perform about equally.

## INT8 quantized weights

`checkpoints/dpo_epoch_0.int8.pt` is the same model at **1.42 GB instead of
2.64 GB** (1.62 GB resident on GPU). Measured on all four tasks, 500 examples
each:

| Task | bf16 | INT8 | Δ |
|---|---|---|---|
| HellaSwag | 40.4 | 40.2 | −0.2 |
| ARC-easy | 45.0 | 45.0 | 0.0 |
| PIQA | 65.6 | 65.2 | −0.4 |
| WinoGrande | 49.0 | 48.8 | −0.2 |

At 500 examples, 0.2 points is a single example — the difference is noise.

Weight-only, symmetric, per-output-channel: each row of each weight matrix
carries its own scale, because rows differ in range by orders of magnitude.
Activations stay bf16, and `lm_head` is left unquantized. **This saves memory,
not time** — the weight is dequantized before the matmul, so arithmetic runs
at the same speed.

```bash
SUTRA_CKPT=checkpoints/dpo_epoch_0.int8.pt python inference.py "Explain gravity."
```

Not to be confused with DeepSeek's FP8 work, which is a *training* technique.
This is post-training quantization, applied to finished weights.

## Limitations

Trained on 18B tokens — roughly **500x less** than comparable 1B models such as
Llama 3.2 1B (9T tokens). Concretely:

- Writes fluent English and follows formatting instructions well
- Does **not** reliably recall facts, and states wrong ones confidently
- Does **not** do multi-step reasoning or write working code
- Sensitive to phrasing — a typo or a terse prompt derails it, where a larger
  model would recover

Pair it with retrieval for anything knowledge-dependent.

## License

Apache 2.0.
