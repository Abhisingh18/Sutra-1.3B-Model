"""Custom handler for a HuggingFace Inference Endpoint.

HF Endpoints' DEFAULT handler builds an AutoTokenizer + AutoModelForCausalLM
pipeline itself, and that fails here: the repo ships a raw tokenizers.json
(the format src/chat.py and deploy/server.py both use), not the
tokenizer_config.json / vocab layout AutoTokenizer expects. A custom handler
sidesteps that by loading the tokenizer the same way this project always has.

The generation settings below (top_k=40, top_p=0.9, a 24-token floor before
<|end_turn|> is allowed, no-repeat on the last 3 tokens) are not arbitrary --
they are copied from deploy/server.py's generate_tokens(), which is what the
live chat page runs. A different sampling setup here would make the endpoint
answer differently than the same model does everywhere else.

HF loads this file automatically when it is named handler.py at the repo
root and the endpoint's "Custom handler" logic is enabled -- no extra
configuration beyond that.

Local test, without provisioning an endpoint:
    python -m deploy.hf_endpoint_handler
"""

import os

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

SYSTEM = "You are a helpful assistant. Answer the question directly and clearly."
MIN_NEW_TOKENS = 24    # see server.py: <|end_turn|> fires 3 tokens in ~83% of
                       # the time unblocked, so every early reply is a fragment


class EndpointHandler:
    def __init__(self, path: str = ""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.end_id = self.tok.token_to_id("<|end_turn|>")

        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device).eval()

    def _prompt(self, message: str) -> str:
        return (f"<|begin_of_text|><|system|>\n{SYSTEM}<|end_turn|>\n"
                f"<|user|>\n{message}<|end_turn|>\n<|assistant|>\n")

    def __call__(self, data: dict) -> list[dict]:
        inputs = data.get("inputs", "")
        params = data.get("parameters") or {}
        max_new = int(params.get("max_new_tokens", 256))
        temperature = float(params.get("temperature", 0.5))

        ids = torch.tensor([self.tok.encode(self._prompt(inputs)).ids],
                           device=self.device)

        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new,
                min_new_tokens=min(MIN_NEW_TOKENS, max_new),
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_k=40,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=self.end_id,
                eos_token_id=self.end_id,
            )

        reply = self.tok.decode(out[0].tolist()[ids.shape[1]:])
        return [{"generated_text": reply}]


if __name__ == "__main__":
    # Exercises exactly what an endpoint would run, against the checkpoint
    # already staged for the Hub -- catches a broken handler before it costs
    # an hour of rented GPU time to find out on HF's infrastructure instead.
    import sys

    repo_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    h = EndpointHandler(repo_dir)
    for msg in ["What is the capital of France?", "Write a haiku about rain."]:
        print(f"\n> {msg}")
        print(h({"inputs": msg, "parameters": {"max_new_tokens": 60}})[0]["generated_text"])
