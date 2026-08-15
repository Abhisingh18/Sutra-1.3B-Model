"""Held-out metrics for the SFT and DPO stages.

    python -m src.eval_posttrain --stage sft
    python -m src.eval_posttrain --stage dpo

Neither training script has any validation, so the numbers printed during those
runs are training loss on a single micro-batch. This computes the missing
numbers on data the models never saw:

    sft   loss and perplexity on held-out conversations, per checkpoint, which
          is what tells you whether epoch 3 overfit relative to epoch 1
    dpo   preference accuracy and reward margin on held-out pairs, which is the
          honest version of the 0.66 the training loop reported

The held-out split is taken from the TAIL of each source, while training took
the head, so there is no overlap as long as the loaders stay deterministic.
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F

from .model.loader import load_checkpoint
from .sft import SFTDataset, collate as sft_collate, load_sft_data
from .dpo import (PreferenceDataset, collate as dpo_collate,
                  load_preference_data, sequence_logprob, dpo_loss)


@torch.no_grad()
def eval_sft(ckpts, tok, pad_id, convs, device, batch_size=4, max_batches=100):
    ds = SFTDataset(convs, tok, 4096)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: sft_collate(b, pad_id))
    rows = []
    for path in ckpts:
        model, _, _ = load_checkpoint(path, device)
        model.eval().to(torch.bfloat16)
        tot, n = 0.0, 0
        for i, (x, y) in enumerate(dl):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            tot += loss.item()
            n += 1
        loss = tot / max(n, 1)
        rows.append((os.path.basename(path), loss, torch.tensor(loss).exp().item()))
        print(f"  {rows[-1][0]:<20} loss {loss:.4f}   ppl {rows[-1][2]:7.2f}",
              flush=True)
        del model
        torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def eval_dpo(policy_path, ref_path, tok, pad_id, pairs, device,
             beta=0.1, batch_size=2, max_batches=200):
    ds = PreferenceDataset(pairs, tok)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: dpo_collate(b, pad_id))
    policy, _, _ = load_checkpoint(policy_path, device)
    ref, _, _ = load_checkpoint(ref_path, device)
    policy.eval().to(torch.bfloat16)
    ref.eval().to(torch.bfloat16)

    n_correct = n_seen = 0
    margins, losses = [], []
    for i, (cx, cy, rx, ry) in enumerate(dl):
        if i >= max_batches:
            break
        cx, cy, rx, ry = (t.to(device) for t in (cx, cy, rx, ry))
        x = torch.cat([cx, rx], 0)
        y = torch.cat([cy, ry], 0)
        pol_c, pol_r = sequence_logprob(policy, x, y).chunk(2, 0)
        ref_c, ref_r = sequence_logprob(ref, x, y).chunk(2, 0)
        loss, acc = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        losses.append(loss.item())
        n_correct += acc.item() * cx.shape[0]
        n_seen += cx.shape[0]
        margins.append((pol_c - pol_r).mean().item())
    return {"accuracy": n_correct / max(n_seen, 1),
            "loss": sum(losses) / max(len(losses), 1),
            "margin": sum(margins) / max(len(margins), 1),
            "n_pairs": n_seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "dpo"], required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--held-out", type=int, default=400,
                    help="examples reserved from the tail of the data")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<|pad|>")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.stage == "sft":
        print("loading SFT data (tail held out)...", flush=True)
        convs = load_sft_data()[-args.held_out:]
        print(f"{len(convs)} held-out conversations\n")
        ckpts = sorted(glob.glob("checkpoints/sft/sft_epoch_*.pt"))
        if not ckpts:
            raise SystemExit("no SFT checkpoints found")
        rows = eval_sft(ckpts, tok, pad_id, convs, device)
        best = min(rows, key=lambda r: r[1])
        print(f"\nbest held-out loss: {best[0]} ({best[1]:.4f})")
        # Training loss always falls with more epochs; held-out loss is the only
        # thing that can tell you an epoch made the model worse.
        if best[0] != os.path.basename(ckpts[-1]):
            print("NOTE: the last epoch is NOT the best -- later epochs overfit.")
    else:
        print("loading preference data (tail held out)...", flush=True)
        pairs = load_preference_data()[-args.held_out:]
        print(f"{len(pairs)} held-out pairs\n")
        r = eval_dpo("checkpoints/dpo/dpo_epoch_0.pt",
                     "checkpoints/sft/sft_epoch_2.pt",
                     tok, pad_id, pairs, device)
        print(f"  held-out accuracy : {r['accuracy']*100:.1f}%  "
              f"(random = 50.0%)")
        print(f"  held-out loss     : {r['loss']:.4f}")
        print(f"  mean margin       : {r['margin']:.2f}")
        print(f"  pairs scored      : {r['n_pairs']}")


if __name__ == "__main__":
    main()
