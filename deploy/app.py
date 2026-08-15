"""Gradio chat UI for Sutra-1.3B, sized for a free HuggingFace Space.

The Space runs on 2 vCPU with no GPU, so every choice here trades quality for
latency. The MoE architecture helps more than it does on GPU: only 0.28B of the
1.32B parameters are active per token, so this is really a 0.28B model's compute
cost on CPU.

Weights are pulled from the Hub at startup rather than committed to the Space,
because Spaces are git repos and a 5 GB checkpoint in git history makes the
Space slow to clone forever after.
"""

import os
import sys
import time

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.loader import load_checkpoint            # noqa: E402
from src.tokenizer.special_tokens import (              # noqa: E402
    chat_template, END_TURN,
)

MODEL_REPO = os.environ.get("MODEL_REPO", "REPLACE_ME/Sutra-1.3B-Chat")
CKPT_FILE = os.environ.get("CKPT_FILE", "dpo_epoch_0.pt")

SYSTEM = ("You are a helpful assistant. Answer the question directly and "
          "clearly.")

print("downloading weights...", flush=True)
from huggingface_hub import hf_hub_download              # noqa: E402

ckpt_path = hf_hub_download(MODEL_REPO, CKPT_FILE)
tok_path = hf_hub_download(MODEL_REPO, "tokenizer.json")

from tokenizers import Tokenizer                         # noqa: E402

tok = Tokenizer.from_file(tok_path)
END_TURN_ID = tok.token_to_id(END_TURN)

print("loading model...", flush=True)
# float32, not bfloat16: CPU bf16 matmul falls back to a slow path in torch,
# and 1.32B params in fp32 is 5.3 GB, which fits the 16 GB free tier.
model, mcfg, _ = load_checkpoint(ckpt_path, "cpu")
model.eval().to(torch.float32)
print(f"ready: {mcfg.param_count()['total']/1e9:.2f}B total, "
      f"{mcfg.param_count()['active']/1e9:.2f}B active", flush=True)


def respond(message, chat_history, max_tokens, temperature):
    """Yield the reply token by token.

    Measured at ~10 tok/s on 2 vCPU, so an 80-token reply lands in about eight
    seconds. Streaming still matters: eight seconds of blank chat box reads as
    broken, whereas eight seconds of text appearing reads as thinking.
    """
    if not message.strip():
        yield ""
        return

    prompt = chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": message}],
        add_generation_prompt=True,
    )
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids])

    caches = [(None, None)] * mcfg.n_layers
    offset, cur = 0, ids
    produced = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(int(max_tokens)):
            logits, caches = model.forward(cur, kv_caches=caches, offset=offset)
            offset += cur.shape[1]
            lg = logits[:, -1, :].float()

            # Same three guards as src/chat.py -- without them this model
            # answers "what is ai" with "AI" and stops.
            if produced:
                seen = torch.unique(torch.tensor(produced))
                s = lg[0, seen]
                lg[0, seen] = torch.where(s > 0, s / 1.15, s * 1.15)
            if i < 24:
                lg[:, END_TURN_ID] = float("-inf")

            lg = lg / max(temperature, 1e-5)
            kth = torch.topk(lg, 40)[0][..., -1, None]
            lg = lg.masked_fill(lg < kth, float("-inf"))
            srt, idx = torch.sort(lg, descending=True, dim=-1)
            sp = torch.softmax(srt, dim=-1)
            srt = srt.masked_fill((torch.cumsum(sp, -1) - sp) > 0.9, float("-inf"))
            lg = torch.full_like(lg, float("-inf")).scatter(-1, idx, srt)

            nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1)
            tid = nxt.item()
            if tid == END_TURN_ID:
                break
            produced.append(tid)
            cur = nxt
            yield tok.decode(produced)

    dt = time.time() - t0
    text = tok.decode(produced)
    yield f"{text}\n\n*{len(produced)} tokens in {dt:.0f}s*"


DESCRIPTION = """
# Sutra-1.3B

A 1.32B-parameter Mixture-of-Experts language model **trained from scratch** —
no pretrained weights, no HuggingFace `Trainer`. Pretrained on 18B tokens, then
chat-tuned with SFT and aligned with DPO.

| | |
|---|---|
| Parameters | 1.32B total / **0.28B active** (48 experts, top-4) |
| Attention | Multi-head Latent Attention (MLA) |
| Training | 18B tokens, 4x RTX 6000 Ada, ~4.5 days |
| Benchmarks | HellaSwag 40.4 · ARC-easy 45.0 · PIQA 65.6 |

**This Space runs on a free CPU.** Measured at ~10 tokens/second on 2 vCPU,
so a reply takes about 10 seconds; text streams in as it is generated. Only
0.28B of the 1.32B parameters are active per token, which is what makes CPU
inference practical at all.

**What to expect.** It writes fluent English, follows formatting instructions,
and handles simple explanations. It does *not* reliably know facts, do
multi-step reasoning, or write working code — 18B training tokens is roughly
500x less than Llama 3.2 1B saw. Ask it to write or explain, not to recall.
"""

with gr.Blocks(title="Sutra-1.3B") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Accordion("Settings", open=False):
        max_tokens = gr.Slider(16, 200, value=80, step=8, label="Max new tokens")
        temperature = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Temperature")
    gr.ChatInterface(
        respond,
        additional_inputs=[max_tokens, temperature],
        examples=[
            ["Write a short email to my manager asking for two days of leave."],
            ["Explain photosynthesis in three sentences."],
            ["What is machine learning?"],
            ["List five healthy breakfast foods."],
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
