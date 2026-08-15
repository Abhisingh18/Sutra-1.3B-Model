"""HTTP inference server for the Vercel frontend.

    pip install fastapi uvicorn sse-starlette
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
      python -m deploy.server --ckpt checkpoints/dpo/dpo_epoch_0.pt

Then expose it (no account needed, no port forwarding):

    cloudflared tunnel --url http://localhost:8000

That prints a public https URL. Put it in the frontend as NEXT_PUBLIC_API_URL.

Tokens stream over Server-Sent Events. At ~15 tok/s a full reply takes several
seconds, and a chat UI that waits for the whole thing before rendering reads as
broken -- the same reason src/chat.py and the Gradio app stream.
"""

import argparse
import asyncio
import json
import os

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model.loader import load_checkpoint, describe
from src.tokenizer.special_tokens import chat_template, END_TURN

SYSTEM = "You are a helpful assistant. Answer the question directly and clearly."

app = FastAPI(title="Sutra-1.3B")

# The frontend is served from a different origin (vercel.app), so the browser
# will not read the response without this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {}


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.5
    rag: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("desc", "not loaded"),
            "rag": STATE.get("retriever") is not None}


def generate_tokens(req: ChatRequest):
    """Yield decoded text incrementally.

    Guards match src/chat.py exactly. Without min_new_tokens the model answers
    "what is ai" with "AI" and stops -- it emits <|end_turn|> after three tokens
    with probability 0.83.
    """
    model, tok, mcfg, device = (STATE["model"], STATE["tok"],
                                STATE["mcfg"], STATE["device"])
    end_id = tok.token_to_id(END_TURN)

    question = req.message
    if req.rag and STATE.get("retriever") is not None:
        from src.rag.retrieve import build_prompt
        hits = STATE["retriever"].search(req.message, k=3)
        question = build_prompt(req.message, hits)

    prompt = chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": question}], add_generation_prompt=True)
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids],
                       device=device)

    caches = [(None, None)] * mcfg.n_layers
    offset, cur, produced = 0, ids, []
    sent = 0

    with torch.no_grad():
        for i in range(req.max_tokens):
            logits, caches = model.forward(cur, kv_caches=caches, offset=offset)
            offset += cur.shape[1]
            lg = logits[:, -1, :].float()

            if produced:
                seen = torch.tensor(produced, device=device).unique()
                s = lg[0, seen]
                lg[0, seen] = torch.where(s > 0, s / 1.25, s * 1.25)
            # Block any 4-gram that already appeared. Without this the model
            # loops: "A computer is a system of computers." five times over.
            if len(produced) >= 3:
                seq = ids[0].tolist() + produced
                pre = tuple(seq[-3:])
                for j in range(len(seq) - 3):
                    if tuple(seq[j:j + 3]) == pre:
                        lg[0, seq[j + 3]] = float("-inf")

            if i < 24:
                lg[:, end_id] = float("-inf")

            lg = lg / max(req.temperature, 1e-5)
            kth = torch.topk(lg, 40)[0][..., -1, None]
            lg = lg.masked_fill(lg < kth, float("-inf"))
            srt, idx = torch.sort(lg, descending=True, dim=-1)
            sp = torch.softmax(srt, dim=-1)
            srt = srt.masked_fill((torch.cumsum(sp, -1) - sp) > 0.9, float("-inf"))
            lg = torch.full_like(lg, float("-inf")).scatter(-1, idx, srt)

            nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1)
            tid = nxt.item()
            if tid == end_id:
                break
            produced.append(tid)
            cur = nxt

            # Decode the whole prefix each step and emit only the delta: BPE
            # pieces do not map to characters one-to-one, so decoding a single
            # token in isolation mangles multi-byte and word-boundary output.
            text = tok.decode(produced)
            if len(text) > sent:
                yield text[sent:]
                sent = len(text)


@app.post("/chat")
async def chat(req: ChatRequest):
    async def stream():
        loop = asyncio.get_event_loop()
        gen = generate_tokens(req)
        while True:
            # Generation is blocking and CPU/GPU bound; run it off the event
            # loop so one request cannot freeze the server for everyone else.
            chunk = await loop.run_in_executor(None, lambda: next(gen, None))
            if chunk is None:
                break
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dpo/dpo_epoch_0.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--rag-index", help="enable retrieval from this index dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    model = model.to(torch.bfloat16 if device == "cuda" else torch.float32)

    STATE.update(model=model, mcfg=mcfg, device=device,
                 tok=Tokenizer.from_file(args.tokenizer),
                 desc=describe(model, mcfg), retriever=None)

    if args.rag_index:
        from src.rag.retrieve import Retriever
        STATE["retriever"] = Retriever(args.rag_index, device=device)
        print(f"rag: {len(STATE['retriever'].chunks):,} chunks")

    print(f"{STATE['desc']} on {device}")
    print(f"listening on http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
