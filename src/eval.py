"""Multiple-choice benchmarks for any checkpoint in the pipeline.

    python -m src.eval --ckpt checkpoints/final.pt
    python -m src.eval --ckpt checkpoints/dpo/dpo_epoch_0.pt --tasks hellaswag,piqa
    python -m src.eval --compare          # base vs sft vs dpo, side by side

Scoring is log-likelihood, not generation: each candidate answer is appended to
the context and scored, and the highest-scoring one is the model's pick. That
is what makes these numbers comparable to published results, and it works on a
base model that cannot follow instructions at all.

Two accuracies are reported because they disagree in a useful way:

    acc       sum of token log-probs -- biased toward SHORT answers
    acc_norm  divided by answer length in characters -- the fairer number,
              and the one usually quoted for HellaSwag and ARC

Random baseline is 1/n_choices: 25% for HellaSwag and ARC, 50% for PIQA and
WinoGrande. A score at baseline means the model has learned nothing the task
measures -- expected for MMLU-style knowledge at this scale, not for PIQA.

No training happens here. Checkpoints are read, never written.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from .model.loader import load_checkpoint, describe


# Each loader returns a list of {"context": str, "choices": [str], "gold": int}.
# Keeping one shape for every task is what lets the scorer stay this small.

def load_hellaswag(n):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if row["label"] == "":
            continue
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize() if row["ctx_b"] else row["ctx_a"]
        out.append({"context": row["activity_label"] + ": " + ctx,
                    "choices": [" " + e.strip() for e in row["endings"]],
                    "gold": int(row["label"])})
    return out


def load_arc_easy(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        out.append({"context": "Question: " + row["question"] + "\nAnswer:",
                    "choices": [" " + t for t in row["choices"]["text"]],
                    "gold": labels.index(row["answerKey"])})
    return out


def load_piqa(n):
    from datasets import load_dataset
    # Not ybisk/piqa: that repo ships a loading script, which datasets>=4 no
    # longer executes. baber/piqa is the same data as parquet.
    ds = load_dataset("baber/piqa", split="validation", streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        out.append({"context": "Question: " + row["goal"] + "\nAnswer:",
                    "choices": [" " + row["sol1"], " " + row["sol2"]],
                    "gold": int(row["label"])})
    return out


def load_winogrande(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                      streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if row["answer"] not in ("1", "2"):
            continue
        # WinoGrande is scored by substituting each option into the blank and
        # comparing the likelihood of the SHARED remainder, so the context
        # differs per choice. Modelled here as full-sentence scoring, which is
        # the simpler variant and slightly easier than the official one.
        idx = row["sentence"].index("_")
        prefix, suffix = row["sentence"][:idx], row["sentence"][idx + 1:]
        out.append({"context": prefix.rstrip(),
                    "choices": [" " + o + suffix for o in
                                (row["option1"], row["option2"])],
                    "gold": int(row["answer"]) - 1})
    return out


TASKS = {
    "hellaswag": load_hellaswag,
    "arc_easy": load_arc_easy,
    "piqa": load_piqa,
    "winogrande": load_winogrande,
}


@torch.no_grad()
def score_choice(model, tok, context, choice, device):
    """Total log-prob of `choice` given `context`, and its character length."""
    ctx_ids = tok.encode(context, add_special_tokens=False).ids
    full_ids = tok.encode(context + choice, add_special_tokens=False).ids
    if len(full_ids) <= len(ctx_ids):
        return -1e9, 1
    x = torch.tensor([full_ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x, return_logits=True)
    logp = F.log_softmax(logits.float(), dim=-1)
    # Position i predicts token i+1, so the answer's tokens are scored by the
    # logits one step to their left.
    total = 0.0
    for i in range(len(ctx_ids), len(full_ids)):
        total += logp[0, i - 1, full_ids[i]].item()
    return total, max(len(choice), 1)


def run_task(model, tok, examples, device):
    hits = hits_norm = 0
    for ex in examples:
        scored = [score_choice(model, tok, ex["context"], c, device)
                  for c in ex["choices"]]
        raw = [s for s, _ in scored]
        norm = [s / L for s, L in scored]
        hits += int(max(range(len(raw)), key=lambda i: raw[i]) == ex["gold"])
        hits_norm += int(max(range(len(norm)), key=lambda i: norm[i]) == ex["gold"])
    n = len(examples)
    return {"n": n, "acc": hits / n, "acc_norm": hits_norm / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--compare", action="store_true",
                    help="evaluate base, SFT and DPO together")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande")
    ap.add_argument("--limit", type=int, default=500,
                    help="examples per task; 500 gives +-2% noise, 1000 +-1.5%")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    if not args.ckpt and not args.compare:
        raise SystemExit("pass --ckpt or --compare")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in task_names:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t}; choose from {list(TASKS)}")

    print("loading task data...", flush=True)
    data = {t: TASKS[t](args.limit) for t in task_names}
    for t, ex in data.items():
        print(f"  {t}: {len(ex)} examples, {len(ex[0]['choices'])} choices")

    if args.compare:
        ckpts = [("base", "checkpoints/final.pt"),
                 ("sft", "checkpoints/sft/sft_epoch_2.pt"),
                 ("dpo", "checkpoints/dpo/dpo_epoch_0.pt")]
        ckpts = [(n, p) for n, p in ckpts if os.path.exists(p)]
    else:
        ckpts = [(os.path.basename(args.ckpt), args.ckpt)]

    results = {}
    for name, path in ckpts:
        model, mcfg, _ = load_checkpoint(path, device)
        model.eval()
        if device == "cuda":
            model = model.to(torch.bfloat16)
        print(f"\n=== {name}: {path} ===", flush=True)
        results[name] = {}
        for t in task_names:
            r = run_task(model, tok, data[t], device)
            results[name][t] = r
            print(f"  {t:<12} acc {r['acc']*100:5.1f}%   "
                  f"acc_norm {r['acc_norm']*100:5.1f}%   (n={r['n']})", flush=True)
        del model
        torch.cuda.empty_cache()

    baselines = {"hellaswag": 25.0, "arc_easy": 25.0,
                 "piqa": 50.0, "winogrande": 50.0}
    print("\n" + "=" * 62)
    print(f"{'task':<13}{'random':>8}" + "".join(f"{n[:10]:>12}" for n, _ in ckpts))
    print("-" * 62)
    for t in task_names:
        row = f"{t:<13}{baselines.get(t, 0):>7.1f}%"
        for name, _ in ckpts:
            row += f"{results[name][t]['acc_norm']*100:>11.1f}%"
        print(row)
    print("=" * 62)
    print("(acc_norm, length-normalised)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
