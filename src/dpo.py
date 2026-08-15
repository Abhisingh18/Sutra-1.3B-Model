"""Stage 3: Direct Preference Optimization -- alignment.

    torchrun --standalone --nproc_per_node=4 -m src.dpo --sft checkpoints/sft/sft_epoch_2.pt

SFT taught the model what an answer looks like. DPO teaches it which of two
plausible answers is *better*: more helpful, less evasive, less padded with
filler. This is the stage that produces the "polished" feel people associate
with ChatGPT.

ChatGPT originally used RLHF/PPO, which requires training a separate reward
model and then running reinforcement learning against it -- expensive and
fiddly. DPO reaches comparable quality with a single loss function and no reward
model. Use DPO.

The loss, in words: raise the log-probability of the chosen answer, lower the
rejected one, but penalise drifting too far from the frozen reference model.
That last clause is the whole game. Without it the model discovers that longer,
more list-shaped answers score well and starts answering every question with
bullet points regardless of what was asked. That failure is called reward
hacking and you will recognise it immediately when you see it.
"""

import argparse
import glob
import math
import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing,
)

from .model.loader import load_checkpoint, describe
from .model.block import Block
from .model.moe_transformer import MoEBlock
from .tokenizer.special_tokens import BOS, USER, ASSISTANT, END_TURN

IGNORE = -100


def sequence_logprob(model, x, y):
    """Sum of log-probabilities of the answer tokens under `model`."""
    # return_logits=True is required here: the default path returns only the
    # LAST position's logits (all that generation needs), but DPO has to score
    # every token of both candidate answers.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x, return_logits=True)
    logits = logits.float()
    logp = F.log_softmax(logits, dim=-1)

    mask = (y != IGNORE)
    safe_y = y.masked_fill(~mask, 0)
    token_logp = logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected,
             ref_chosen, ref_rejected, beta=0.1):
    """The DPO objective.

    beta controls how tightly the model is held to the reference. 0.1 is the
    standard value. Larger beta = more conservative, keeps SFT quality but
    learns less; smaller beta = more aggressive, risks reward hacking.
    """
    # How much more the policy prefers chosen-over-rejected than the reference does.
    pi_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = pi_logratio - ref_logratio

    loss = -F.logsigmoid(beta * logits).mean()

    # Fraction of pairs ranked correctly. This is the metric to watch -- it
    # should climb from ~0.5 to 0.65-0.75. If it hits 0.95 you are overfitting
    # and the model is probably gaming length rather than learning quality.
    accuracy = (logits > 0).float().mean()
    return loss, accuracy


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_len=2048):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def _encode(self, prompt, answer):
        # Must match the SFT chat template byte for byte. A mismatch here is a
        # silent failure: training appears fine, deployed quality is worse.
        p = f"{BOS}{USER}\n{prompt}{END_TURN}\n{ASSISTANT}\n"
        p_ids = self.tok.encode(p, add_special_tokens=False).ids
        a_ids = self.tok.encode(answer + END_TURN, add_special_tokens=False).ids

        ids = (p_ids + a_ids)[:self.max_len]
        labels = ([IGNORE] * len(p_ids) + a_ids)[:self.max_len]
        return ids, labels

    def __getitem__(self, i):
        p = self.pairs[i]
        ci, cl = self._encode(p["prompt"], p["chosen"])
        ri, rl = self._encode(p["prompt"], p["rejected"])
        return {"chosen_ids": ci, "chosen_labels": cl,
                "rejected_ids": ri, "rejected_labels": rl}


def collate(batch, pad_id):
    # Chosen and rejected are padded to ONE common length, not to their own.
    # That is what lets the training loop concatenate them into a single
    # forward pass, which DDP requires -- see the comment in main().
    n = max(max(len(b["chosen_ids"]), len(b["rejected_ids"])) for b in batch)

    def pad(seqs, fill):
        out = torch.full((len(seqs), n), fill, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out

    cx = pad([b["chosen_ids"] for b in batch], pad_id)
    cy = pad([b["chosen_labels"] for b in batch], IGNORE)
    rx = pad([b["rejected_ids"] for b in batch], pad_id)
    ry = pad([b["rejected_labels"] for b in batch], IGNORE)
    return cx[:, :-1], cy[:, 1:], rx[:, :-1], ry[:, 1:]


def load_preference_data(max_pairs=100_000):
    from datasets import load_dataset
    pairs = []
    for name, split in [("HuggingFaceH4/ultrafeedback_binarized", "train_prefs"),
                        ("Anthropic/hh-rlhf", "train")]:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            n = 0
            for row in ds:
                if "chosen" in row and isinstance(row["chosen"], list):
                    prompt = row["prompt"]
                    chosen = row["chosen"][-1]["content"]
                    rejected = row["rejected"][-1]["content"]
                elif isinstance(row.get("chosen"), str):
                    # hh-rlhf stores full transcripts; take the last turn.
                    chosen, rejected = row["chosen"], row["rejected"]
                    prompt = chosen.split("Assistant:")[0].replace("Human:", "").strip()
                    chosen = chosen.split("Assistant:")[-1].strip()
                    rejected = rejected.split("Assistant:")[-1].strip()
                else:
                    continue
                if prompt and chosen and rejected and chosen != rejected:
                    pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                    n += 1
                if n >= max_pairs // 2:
                    break
            print(f"  {name}: {n} pairs")
        except Exception as e:
            print(f"  ! {name} failed: {e}")
    return pairs[:max_pairs]


def save_ckpt(path, raw_policy, optimizer, step, epoch, batch_idx, mcfg, args):
    """Atomic checkpoint, same contract as pretraining.

    Optimizer state is included because Adam's moments are worth ~100 steps of
    warmup -- resuming without them puts a visible kink in the loss curve.

    `batch_idx` is what makes a mid-epoch resume possible: the sampler is
    deterministic given (seed, epoch), so replaying the epoch and skipping the
    first `batch_idx` batches reproduces exactly the data ordering the crashed
    run would have seen.
    """
    tmp = path + ".tmp"
    torch.save({
        "model": raw_policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_config": mcfg.__dict__,
        "args": vars(args),
    }, tmp)
    # Either fully written or absent -- a crash mid-save must not leave a
    # corrupt file that breaks the next resume.
    os.replace(tmp, path)


def find_latest_ckpt(out_dir):
    files = glob.glob(os.path.join(out_dir, "dpo_step_*.pt"))
    if not files:
        return None
    return max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))


def prune_ckpts(out_dir, keep_n):
    files = sorted(glob.glob(os.path.join(out_dir, "dpo_step_*.pt")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    for f in files[:-keep_n]:
        os.remove(f)


def init_wandb(args, mcfg, world):
    """Separate run from pretraining, resumable like it.

    The id lives in <out>/wandb_run_id.txt, NOT in checkpoints/, so a DPO
    restart never attaches to the pretraining run and overwrite its history.
    Returns None if wandb is disabled or missing -- training must not depend on
    logging being available.
    """
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without it", flush=True)
        return None

    os.makedirs(args.out, exist_ok=True)
    id_path = os.path.join(args.out, "wandb_run_id.txt")
    if os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
        resume_mode = "must"
    else:
        run_id = wandb.util.generate_id()
        with open(id_path, "w") as f:
            f.write(run_id)
        resume_mode = "allow"

    run = wandb.init(
        project=args.wandb_project, name=args.wandb_name,
        id=run_id, resume=resume_mode,
        config={
            "stage": "dpo", "base": args.sft, "beta": args.beta,
            "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "world_size": world,
            "total_params": mcfg.param_count()["total"],
            "active_params": mcfg.param_count()["active"],
        },
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    print(f"wandb: run {run_id} -> {run.url}", flush=True)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    # DPO uses an even smaller LR than SFT -- it is a gentle nudge, not retraining.
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="optimizer steps between checkpoints (~40 min)")
    ap.add_argument("--keep-last-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", default="abhi-llm")
    ap.add_argument("--wandb-name", default="Sutra-1.3B-DPO")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    main_proc = rank == 0

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<|pad|>")

    # Two copies: one trains, one stays frozen as the reference.
    policy, mcfg, _ = load_checkpoint(args.sft, device)
    ref, _, _ = load_checkpoint(args.sft, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    if main_proc:
        print(f"policy + frozen reference: {describe(policy, mcfg)} from {args.sft}")

    # Same requirement as pretraining and SFT. Only the policy needs it -- the
    # reference runs under no_grad, so it stores no activations to begin with.
    apply_activation_checkpointing(
        policy, checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda m: isinstance(m, (Block, MoEBlock)),
    )

    raw_policy = policy
    if world > 1:
        policy = DDP(
            policy, device_ids=[local_rank], gradient_as_bucket_view=True,
            # REQUIRED for MoE, exactly as in src/train.py. Some experts receive
            # zero tokens in a micro-batch, so their buckets never become ready
            # and the collectives desync across ranks.
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    pairs = load_preference_data()
    if main_proc:
        print(f"{len(pairs):,} preference pairs")

    ds = PreferenceDataset(pairs, tok)
    # An explicit seed is what makes resume reproducible: DistributedSampler
    # derives its permutation from (seed, epoch), so a restart replays exactly
    # the ordering the crashed run had.
    sampler = (torch.utils.data.distributed.DistributedSampler(ds, seed=args.seed)
               if world > 1 else None)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        collate_fn=lambda b: collate(b, pad_id), num_workers=4, drop_last=True)

    opt = raw_policy.configure_optimizer(args.lr, 0.0, (0.9, 0.95))
    os.makedirs(args.out, exist_ok=True)

    # ---- resume ------------------------------------------------------------
    step, start_epoch, skip_batches = 0, 0, 0
    latest = find_latest_ckpt(args.out)
    if latest:
        ck = torch.load(latest, map_location=device, weights_only=False)
        raw_policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        start_epoch = ck["epoch"]
        skip_batches = ck["batch_idx"] + 1
        if main_proc:
            print(f"resuming from {latest}: epoch {start_epoch}, step {step}, "
                  f"skipping {skip_batches} batches", flush=True)
    elif main_proc:
        print("starting fresh", flush=True)

    wandb_run = init_wandb(args, mcfg, world) if main_proc else None
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        policy.train()
        opt.zero_grad(set_to_none=True)

        for i, (cx, cy, rx, ry) in enumerate(dl):
            # Fast-forward to where the crashed run stopped. Only the first
            # resumed epoch skips; later epochs start from 0 as normal.
            if i < skip_batches:
                continue

            cx, cy, rx, ry = cx.to(device), cy.to(device), rx.to(device), ry.to(device)

            # ONE forward over [chosen; rejected] stacked on the batch axis,
            # not two separate ones. DDP marks each parameter ready when its
            # gradient arrives, so two forwards feeding a single backward makes
            # it mark the same parameter twice and abort with
            #   Expected to mark a variable ready only once
            # Concatenating is also how the reference DPO implementations do it.
            x = torch.cat([cx, rx], dim=0)
            y = torch.cat([cy, ry], dim=0)

            pol_c, pol_r = sequence_logprob(policy, x, y).chunk(2, dim=0)
            with torch.no_grad():
                ref_c, ref_r = sequence_logprob(ref, x, y).chunk(2, dim=0)

            loss, acc = dpo_loss(pol_c, pol_r, ref_c, ref_r, args.beta)
            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(raw_policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % 20 == 0 and main_proc:
                    margin = (pol_c - pol_r).mean().item()
                    print(f"step {step} | loss {loss.item():.4f} | "
                          f"acc {acc.item():.3f} | margin {margin:.2f} | "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)
                    if wandb_run is not None:
                        wandb_run.log({
                            "step": step,
                            "dpo/loss": loss.item(),
                            # The metric to watch: fraction of pairs the policy
                            # ranks correctly. Should climb from ~0.5 and settle
                            # around 0.65-0.75. Near 1.0 means reward hacking.
                            "dpo/accuracy": acc.item(),
                            # Gap between chosen and rejected logprobs. Growing
                            # without bound is the other reward-hacking signal.
                            "dpo/margin": margin,
                            "dpo/epoch": epoch,
                        })

                if step % args.ckpt_every == 0 and main_proc:
                    p = os.path.join(args.out, f"dpo_step_{step:07d}.pt")
                    save_ckpt(p, raw_policy, opt, step, epoch, i, mcfg, args)
                    prune_ckpts(args.out, args.keep_last_n)
                    print(f"  saved {p}", flush=True)

        # Only the epoch we resumed into skips batches; the rest start at 0.
        skip_batches = 0

        if main_proc:
            path = os.path.join(args.out, f"dpo_epoch_{epoch}.pt")
            tmp = path + ".tmp"
            # Inference weights only -- no optimizer state, so this stays small.
            # Written atomically for the same reason as the step checkpoints.
            torch.save({"model": raw_policy.state_dict(),
                        "model_config": mcfg.__dict__}, tmp)
            os.replace(tmp, path)
            print(f"saved {path}")

    if wandb_run is not None:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
