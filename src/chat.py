"""Interactive chat with a trained model.

    python -m src.chat --ckpt checkpoints/dpo/dpo_epoch_0.pt

Also works on a base (pre-SFT) checkpoint via --raw, which is worth doing once:
seeing the base model continue your text instead of answering it makes the
purpose of SFT concrete in a way no explanation does.
"""

import argparse

import torch

from .model.loader import load_checkpoint, describe
from .tokenizer.special_tokens import chat_template, END_TURN, EOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    # The bare "You are a helpful assistant." makes this model open replies with
    # "I don't have the capability to do this." -- a refusal pattern it picked up
    # from the hh-rlhf half of the DPO data. Asking for a direct answer in the
    # system prompt removes it.
    ap.add_argument("--system",
                    default="You are a helpful assistant. Answer the question "
                            "directly and clearly.")
    # 0.5, not 0.7: at 0.28B active params the tail of the distribution is
    # mostly noise, and sampling from it is what produced off-topic replies.
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    # The model answers "what is ai" with "Artificial Intelligence" and stops.
    # Blocking <|end_turn|> for the first 32 tokens forces it to elaborate.
    ap.add_argument("--min-new-tokens", type=int, default=32)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    # Turns of history to carry. 0 by default: at this scale the model copies
    # its own previous answer verbatim rather than answering the new question,
    # and the repetition penalty cannot see prompt tokens to stop it.
    ap.add_argument("--history-turns", type=int, default=0)
    ap.add_argument("--rag", metavar="INDEX_DIR",
                    help="retrieve context from an index built by src.rag.ingest")
    ap.add_argument("--rag-k", type=int, default=3)
    ap.add_argument("--show-sources", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="plain text completion, no chat template")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()

    if device == "cuda":
        model = model.to(torch.bfloat16)

    print(f"loaded {describe(model, mcfg)} on {device}")
    print("type 'exit' to quit, 'reset' to clear history\n")

    retriever = None
    if args.rag:
        from .rag.retrieve import Retriever, build_prompt
        retriever = Retriever(args.rag, device=device)
        print(f"rag: {len(retriever.chunks):,} chunks from {args.rag}")

    end_turn_id = tok.token_to_id(END_TURN)
    eos_id = tok.token_to_id(EOS)
    history = []

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "exit":
            break
        if user == "reset":
            history = []
            print("(history cleared)\n")
            continue

        if args.raw:
            prompt = user
        else:
            question = user
            if retriever is not None:
                hits = retriever.search(user, k=args.rag_k)
                if args.show_sources:
                    for score, _, src in hits:
                        print(f"  [ctx {score:.2f}] {src}")
                    if not hits:
                        print("  [ctx] nothing above the relevance floor")
                question = build_prompt(user, hits)

            msgs = [{"role": "system", "content": args.system}]
            msgs += history
            msgs.append({"role": "user", "content": question})
            prompt = chat_template(msgs, add_generation_prompt=True)

        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids],
                           device=device)

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_new_tokens=0 if args.raw else args.min_new_tokens,
                repetition_penalty=args.repetition_penalty,
                eos_id=end_turn_id if not args.raw else eos_id,
            )

        reply = tok.decode(out[0, ids.shape[1]:].tolist())
        reply = reply.split(END_TURN)[0].strip()
        print(f"{reply}\n")

        if not args.raw:
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})
            history = history[-2 * args.history_turns:] if args.history_turns else []


if __name__ == "__main__":
    main()
