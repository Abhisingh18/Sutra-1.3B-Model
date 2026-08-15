"""Stage 2: Supervised fine-tuning -- turn the base model into a chat model.

    torchrun --standalone --nproc_per_node=4 -m src.sft --base checkpoints/final.pt

The base model out of pretraining does not answer questions; it continues text.
Ask it "What is the capital of India?" and it will happily write three more
questions. SFT is what teaches it that after <|assistant|> it should *respond*.

The single most important detail here is loss masking. Loss is computed ONLY on
assistant tokens. Training the model to predict the user's turns teaches it to
imitate users, which shows up as a model that asks you questions instead of
answering them.

This stage is cheap -- roughly a day on 4 GPUs -- and it is where the model
suddenly starts to feel like a product.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing,
)

from .model.loader import load_checkpoint, describe
from .model.block import Block
from .model.moe_transformer import MoEBlock
from .tokenizer.special_tokens import (
    BOS, EOS, SYSTEM, USER, ASSISTANT, END_TURN,
)

IGNORE = -100


class SFTDataset(torch.utils.data.Dataset):
    """Renders conversations to the chat template and builds masked labels."""

    def __init__(self, conversations, tokenizer, max_len=4096):
        self.convs = conversations
        self.tok = tokenizer
        self.max_len = max_len
        self.role_token = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

    def __len__(self):
        return len(self.convs)

    def __getitem__(self, i):
        messages = self.convs[i]
        ids, labels = [], []

        def add(text, supervised):
            enc = self.tok.encode(text, add_special_tokens=False).ids
            ids.extend(enc)
            labels.extend(enc if supervised else [IGNORE] * len(enc))

        add(BOS, False)
        for m in messages:
            # Role tag and the newline after it are structure, not content:
            # the model should not be scored on producing them.
            add(f"{self.role_token[m['role']]}\n", False)
            # Assistant content is the only thing we actually train on. The
            # closing <|end_turn|> IS supervised -- that is how the model learns
            # to stop, and a model that never stops is unusable.
            supervised = (m["role"] == "assistant")
            add(m["content"], supervised)
            add(END_TURN, supervised)
            add("\n", False)

        ids, labels = ids[:self.max_len], labels[:self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    y = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        x[i, :L] = torch.tensor(b["input_ids"])
        y[i, :L] = torch.tensor(b["labels"])
    # Shift for next-token prediction.
    return x[:, :-1], y[:, 1:]


def load_sft_data(max_samples=200_000):
    """Open instruction datasets. All permissively licensed.

    Quality beats quantity here by a wide margin. 50K carefully chosen
    conversations produce a better chat model than 500K scraped ones -- this is
    the most consistently reproduced result in the post-training literature.
    """
    from datasets import load_dataset
    convs = []

    sources = [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", 100_000),
        ("teknium/OpenHermes-2.5", "train", 60_000),
        ("allenai/tulu-3-sft-mixture", "train", 40_000),
    ]

    for name, split, n in sources:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            count = 0
            for row in ds:
                msgs = row.get("messages") or row.get("conversations")
                if not msgs:
                    continue
                norm = []
                for m in msgs:
                    role = m.get("role") or m.get("from", "")
                    role = {"human": "user", "gpt": "assistant"}.get(role, role)
                    content = m.get("content") or m.get("value", "")
                    if role in ("system", "user", "assistant") and content:
                        norm.append({"role": role, "content": content})
                # Must end on an assistant turn, or there is nothing to train on.
                if len(norm) >= 2 and norm[-1]["role"] == "assistant":
                    convs.append(norm)
                    count += 1
                if count >= n:
                    break
            print(f"  {name}: {count} conversations")
        except Exception as e:
            print(f"  ! {name} failed: {e}")

    return convs[:max_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pretrained checkpoint")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    # 2e-5, not 3e-4. SFT on a pretrained model needs a learning rate roughly
    # 10x smaller than pretraining, or it erases what pretraining learned.
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=4096)
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

    model, mcfg, _ = load_checkpoint(args.base, device)
    if main_proc:
        print(f"loaded base: {describe(model, mcfg)} from {args.base}")

    # Non-negotiable for this MoE, same as pretraining. Without it the expert
    # loop keeps every intermediate for all 48 experts across 15 layers and
    # peaks near 47 GB, which OOMs a 48 GB card before the first step finishes.
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda m: isinstance(m, (Block, MoEBlock)),
    )

    raw_model = model
    if world > 1:
        model = DDP(
            model, device_ids=[local_rank], gradient_as_bucket_view=True,
            # REQUIRED for MoE, exactly as in src/train.py. Top-4 routing over
            # 48 experts leaves some experts with zero tokens in a micro-batch,
            # so their parameters get no gradient and their DDP buckets never
            # become ready. Without this, rank 3 dies immediately with
            #   Expected to have finished reduction in the prior iteration ...
            find_unused_parameters=True,
            # Router bias and load counters are updated by an explicit rule on
            # every rank, not by autograd. Broadcasting rank 0's copies would be
            # both wasted bandwidth and wrong.
            broadcast_buffers=False,
        )

    if main_proc:
        print("loading SFT data...")
    convs = load_sft_data()
    if main_proc:
        print(f"{len(convs):,} conversations")

    ds = SFTDataset(convs, tok, args.max_len)
    sampler = (torch.utils.data.distributed.DistributedSampler(ds)
               if world > 1 else None)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        collate_fn=lambda b: collate(b, pad_id), num_workers=4, drop_last=True,
    )

    opt = raw_model.configure_optimizer(args.lr, 0.0, (0.9, 0.95))
    steps_per_epoch = len(dl) // args.grad_accum
    total = steps_per_epoch * args.epochs
    warmup = int(0.03 * total)

    os.makedirs(args.out, exist_ok=True)
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss = loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0:
                lr = (args.lr * (step + 1) / warmup if step < warmup else
                      args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                                    max(total - warmup, 1))))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                opt.step()
                # Keep balancing during SFT too. Instruction data has a very
                # different distribution from pretraining text, so routing does
                # shift here and will drift out of balance if left alone.
                if hasattr(raw_model, "update_router_bias"):
                    raw_model.update_router_bias()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % 20 == 0 and main_proc:
                    print(f"epoch {epoch} step {step}/{total} | "
                          f"loss {loss.item()*args.grad_accum:.4f} | lr {lr:.2e} | "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)

        if main_proc:
            path = os.path.join(args.out, f"sft_epoch_{epoch}.pt")
            torch.save({"model": raw_model.state_dict(),
                        "model_config": mcfg.__dict__}, path)
            print(f"saved {path}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
