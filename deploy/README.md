---
title: Sutra-1.3B Chat
emoji: 🧵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: A 1.32B MoE language model trained from scratch on 18B tokens
---

# Sutra-1.3B

A Mixture-of-Experts language model built from first principles in PyTorch —
tokenizer, data pipeline, pretraining, SFT and DPO, with no pretrained weights
and no `transformers` Trainer anywhere in the stack.

| | |
|---|---|
| Parameters | 1.32B total / 0.28B active |
| Experts | 48 routed + 1 shared, top-4 |
| Attention | Multi-head Latent Attention, kv_lora_rank 256 |
| Layers / d_model | 16 / 1024 |
| Context | 4096 |
| Vocab | 48,000 (English + Devanagari) |
| Training | 18B tokens, 4x RTX 6000 Ada, ~4.5 days |

## Results

| Task | Random | Base | SFT | DPO |
|---|---|---|---|---|
| HellaSwag | 25.0 | 38.4 | 39.8 | **40.4** |
| ARC-easy | 25.0 | 45.0 | 44.8 | **45.0** |
| PIQA | 50.0 | 62.6 | 65.4 | **65.6** |
| WinoGrande | 50.0 | 50.6 | 49.0 | 49.0 |

Pretraining held-out perplexity 15.00; SFT held-out perplexity 5.49.

WinoGrande sitting at chance is the honest signal of what 0.28B active
parameters buy: the model has learned language and a good deal of commonsense,
but not the pronoun-resolution reasoning that task measures.

## Limitations

Trained on 18B tokens — roughly 500x less than comparable 1B models. It writes
fluently and follows instructions, but does not reliably recall facts, reason
in multiple steps, or produce working code. Pair it with retrieval for anything
knowledge-dependent.
