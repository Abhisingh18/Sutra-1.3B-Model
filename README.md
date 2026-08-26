# Sutra-1.3B — a 1.32B parameter MoE LLM trained from scratch

Pretraining, chat fine-tuning and preference alignment, written from first
principles in PyTorch. No HuggingFace `Trainer`, no pretrained weight.

Target hardware: 4x RTX 6000 Ada (48GB), PCIe, no NVLink.

**Live demo:** [sutra-1-3-b-model-15co.vercel.app](https://sutra-1-3-b-model-15co.vercel.app)
**Weights:** [huggingface.co/Abhisingh-18/Sutra-1.3B-Chat](https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat)
**Data recipe:** [huggingface.co/datasets/Abhisingh-18/Sutra-1.3B-Data](https://huggingface.co/datasets/Abhisingh-18/Sutra-1.3B-Data) — every source, config, split and share

```bash
pip install torch tokenizers huggingface_hub
wget https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat/resolve/main/inference.py
python inference.py "Explain photosynthesis in three sentences."
```

Runs on CPU at ~10 tokens/second — only 0.28B of the 1.32B parameters are
active per tokens.

## Results

Log-likelihood scoring, 500 examples per task, length-normalised accuracy
(`python -m src.eval --compare`).

| Task | Random | Base | SFT | DPO |
|---|---|---|---|---|
| HellaSwag | 25.0 | 38.4 | 39.8 | **40.4** |
| ARC-easy | 25.0 | 45.0 | 44.8 | **45.0** |
| PIQA | 50.0 | 62.6 | 65.4 | **65.6** |
| WinoGrande | 50.0 | 50.6 | 49.0 | 49.0 |

Pretraining held-out perplexity **15.00**, SFT held-out perplexity **5.49**.

Read these honestly. ARC-easy and PIQA sit well clear of chance, so the model
learned real commonsense rather than only fluent grammar. WinoGrande sits *at*
chance — the pronoun-resolution reasoning it measures never arrived, which is
the sharpest available statement of what 0.28B active parameters do not buy.

DPO's held-out preference accuracy came out at **47.5%** against a 50% baseline:
the alignment stage did not generalise, and the 66% its training loop printed
was measured on training batches. SFT and DPO perform about equally.

---

## What this produces

A decoder-only transformer, pretrained on 18B tokens of English, Hindi, code
and math, then turned into a chat model via SFT and aligned with DPO.

**Be clear about what 1B buys you.** The model will write fluent English and
usable Hindi, follow instructions, summarise, extract structured data, and
handle domain tasks well after fine-tuning. It will *not* reliably remember
facts, do multi-step reasoning, or write real code — those need 10-100x more
parameters and compute. Pair it with retrieval (RAG) for anything
knowledge-dependent. Built and used with that in mind, it is a genuinely useful
model; measured against ChatGPT, it will disappoint.

---

## Architecture

Two architectures are implemented. `use_moe` in `src/train_config.py` selects
one; MoE is the default.

### MoE — DeepSeek-inspired (default)

| | |
|---|---|
| Total / active params | **1.32B / 0.28B** (4.71x sparsity) |
| Layers | 16 (layer 0 dense, 1-15 MoE) |
| d_model | 1024 |
| Attention | **MLA**, 16 heads, kv_lora_rank 256 |
| Head split | 64 nope + 32 rope, v_head_dim 64 |
| Experts | 48 routed + 1 shared, top-4, width 512 |
| Routing | sigmoid scoring, bias-based balancing (V3 style) |
| Context | 4096 |
| Vocab | 48,000 |

```
tokens → embed → [ RMSNorm → MLA        → +
                   RMSNorm → MoE-FFN    → + ] × 26 → RMSNorm → lm_head
                              │
                        Router (top-4 of 20)
                              + 1 shared expert
```

### Dense — Llama-style

1.16B params, 22 layers, d_model 2048, GQA (32 query / 4 KV), SwiGLU 5632.

### Which to use

| | Dense 1.16B | MoE 1.02B |
|---|---|---|
| Active params | 1.16B | 0.39B |
| Compute/token | 1.0x | **2.97x cheaper** |
| 100B tokens | ~7 weeks | **~4 weeks** |
| Quality | ~1.1B dense | ~0.5B dense |

MoE trades quality for speed here. It is the right choice if you want a working
model quickly and cheap iteration; dense is the right choice if you want the
best model this hardware can produce. **A third option, not yet configured:**
keep active params at 0.39B but raise total to ~2.5B by adding experts. Training
stays at ~4 weeks and the result beats the dense 1.16B. Memory is the only cost,
and 48GB cards have room. This is technically the best use of the hardware.

Nothing in either architecture is novel, and that is deliberate. The largest
risk in a multi-week run is instability, and novelty buys risk without buying
measurable quality at this scale.

---

## Pipeline

| Stage | Script | Time (4 GPUs) | Output |
|---|---|---|---|
| 0. Tokenizer | `src/tokenizer/train_tokenizer.py` | 2-3 hours | `tokenizer/tokenizer.json` |
| 1. Data prep | `src/data/prepare.py` | 3-5 days | ~200GB of token shards |
| 2. Pretrain | `src/train.py` | **~7 weeks** | base model |
| 3. SFT | `src/sft.py` | ~1 day | chat model |
| 4. DPO | `src/dpo.py` | ~1 day | aligned model |

Stage 2 is 95% of the compute. Stages 3 and 4 are 100% of the reason it feels
like a chat assistant.

---

## Running it

```bash
pip install -r requirements.txt

# 0. Tokenizer — run ONCE. Everything downstream depends on it.
python -m src.tokenizer.train_tokenizer --output tokenizer/ --sample-gb 20

# 1. Data — streams from HuggingFace, writes only uint16 tokens
python -m src.data.prepare --out data/tokens --tokens 100e9 --workers 64

# 2. Pretrain — auto-resumes from the newest checkpoint, no flags needed
torchrun --standalone --nproc_per_node=4 -m src.train

# 3. Chat fine-tuning
torchrun --standalone --nproc_per_node=4 -m src.sft --base checkpoints/final.pt

# 4. Alignment
torchrun --standalone --nproc_per_node=4 -m src.dpo --sft checkpoints/sft/sft_epoch_2.pt

# Talk to it
python -m src.chat --ckpt checkpoints/dpo/dpo_epoch_0.pt
```

### Do a dry run first

Before committing two months of GPU time, run the whole pipeline end to end on
the 150M config in `src/model/config.py`. A few days of small-scale runs will
surface the bugs that would otherwise cost you weeks. This is the single highest
-value thing you can do before starting.

---

## Three decisions that cannot be undone

**Special tokens must exist before pretraining.** `src/tokenizer/special_tokens.py`
reserves the chat tokens, reasoning tokens, 32 spare slots, and 4096 audio slots
for a future speech front-end. Added later, their embeddings start from noise
while everything else has seen 100B tokens, and they never catch up. The whole
reservation costs 0.7% of the model.

**Vocab size.** 48K supports English + Devanagari + specials. Changing it means
retokenizing the corpus and retraining from zero.

**The data mixture.** `src/data/mixture.py` is the single place data proportions
live. At 1B scale you cannot maximise English fluency, Hindi fluency, and
reasoning simultaneously — the current split favours English, with Hindi and
math/code as real but secondary. Move the weights if your priorities differ, but
decide before Stage 1, not after.

---

## Expect to be interrupted

A seven-week run will not complete uninterrupted. The training loop is built
around that:

- Checkpoints every 2000 steps (~40 min), written atomically so a crash mid-save
  cannot corrupt them
- Restart picks up the newest checkpoint automatically — same step, same
  optimizer state, same data order
- Batches are a deterministic function of `(step, rank)`, so resume replays
  exactly the data the crashed run would have seen
- A loss-spike guard rolls back to the last checkpoint if loss jumps 1.5x above
  its running average for 3 consecutive steps

Watch `grad_norm` in the logs. Steady is healthy; climbing means a spike is
coming; NaN means the run is already dead and you should roll back.

## Watch the router (MoE only)

Logged every 200 steps:

```
router: load 3.8%-6.1% (uniform 5.0%), dead 0
```

With 20 experts, uniform load is 5% each. The failure mode is **router
collapse**: tokens pile onto a few experts and the rest never learn, so you
paid for capacity you do not get. `max_frac` above ~20%, or any dead expert,
means routing is collapsing — the log flags this as `COLLAPSE RISK`.

Two things prevent it, and both are already in place: layer 0 is dense (routing
on raw embeddings is near-random and collapses early), and load balancing is
bias-based rather than an auxiliary loss. If collapse still appears, raise
`router_bias_update_rate` in `src/model/moe_config.py`.

---

## Layout

```
src/
  model/
    config.py          dense architecture config + parameter accounting
    norm.py            RMSNorm
    rope.py            rotary position embeddings
    attention.py       grouped-query attention + KV cache
    ffn.py             SwiGLU
    block.py           one dense transformer block
    transformer.py     full dense model, optimizer setup, generation
    moe_config.py      MoE config + exact parameter/FLOP accounting
    mla.py             multi-head latent attention (decoupled RoPE)
    moe.py             experts, router, bias-based load balancing
    moe_transformer.py full MoE model
    loader.py          architecture-agnostic checkpoint loading
  tokenizer/
    special_tokens.py  chat template + reserved tokens  ← read this first
    train_tokenizer.py BPE training
  data/
    mixture.py         data proportions
    prepare.py         streaming tokenizer → shards
    dataloader.py      memmap loader
  train_config.py      pretraining hyperparameters + batch arithmetic
  train.py             pretraining loop
  sft.py               chat fine-tuning
  dpo.py               preference alignment
  chat.py              interactive inference
```

Each model file is standalone and commented to be read in order:
`norm.py` → `rope.py` → `attention.py` → `ffn.py` → `block.py` → `transformer.py`.
