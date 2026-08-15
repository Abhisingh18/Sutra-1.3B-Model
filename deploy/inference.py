"""Run Sutra-1.3B. Self-contained -- downloads everything it needs.

    pip install torch tokenizers huggingface_hub
    python inference.py                       # interactive chat
    python inference.py "What is machine learning?"

Works on CPU. Only 0.28B of the 1.32B parameters are active per token, so CPU
generation runs at roughly 10 tokens/second on 2 cores.

This is a custom architecture -- MoE with Multi-head Latent Attention, written
from scratch -- so `transformers` cannot load it. The model code comes down
from this same repo.
"""

import os
import sys

import torch
from huggingface_hub import hf_hub_download, snapshot_download

REPO = os.environ.get("SUTRA_REPO", "Abhisingh-18/Sutra-1.3B-Chat")

# The architecture modules live in this repo under src/. Pull them and put them
# on the path so `from src.model...` resolves.
# Only what is needed to run: the code, the config, the tokenizer and ONE set
# of weights. Without the filter this pulls every checkpoint in the repo -- 26 GB
# instead of 5.3.
local = snapshot_download(REPO, allow_patterns=[
    "src/*", "config.json", "tokenizer.json", "model.safetensors"])
sys.path.insert(0, local)

from src.model.loader import load_checkpoint                      # noqa: E402
from src.tokenizer.special_tokens import chat_template, END_TURN  # noqa: E402
from tokenizers import Tokenizer                                  # noqa: E402

SYSTEM = "You are a helpful assistant. Answer the question directly and clearly."


def build():
    tok = Tokenizer.from_file(os.path.join(local, "tokenizer.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.environ.get("SUTRA_CKPT", "model.safetensors")
    model, mcfg, _ = load_checkpoint(os.path.join(local, ckpt), device)
    model.eval()
    model = model.to(torch.bfloat16 if device == "cuda" else torch.float32)
    return model, mcfg, tok, device


def generate(model, mcfg, tok, device, question, max_new_tokens=120,
             temperature=0.5):
    """Three guards, all necessary.

    Without min_new_tokens this model answers "what is ai" with "AI" and stops:
    it emits <|end_turn|> after three tokens with probability 0.83. The
    repetition penalty is what keeps it from looping once forced to continue,
    and top-p keeps the tail of a 48k vocab from leaking in.
    """
    end_id = tok.token_to_id(END_TURN)
    prompt = chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": question}], add_generation_prompt=True)
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids],
                       device=device)

    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=40, top_p=0.9, min_new_tokens=24, repetition_penalty=1.15,
            eos_id=end_id)
    return tok.decode(out[0, ids.shape[1]:].tolist()).split(END_TURN)[0].strip()


def main():
    model, mcfg, tok, device = build()
    pc = mcfg.param_count()
    print(f"Sutra-1.3B on {device}: {pc['total']/1e9:.2f}B total, "
          f"{pc['active']/1e9:.2f}B active\n")

    if len(sys.argv) > 1:
        print(generate(model, mcfg, tok, device, " ".join(sys.argv[1:])))
        return

    print("type 'exit' to quit\n")
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q == "exit":
            break
        print(generate(model, mcfg, tok, device, q) + "\n")


if __name__ == "__main__":
    main()
