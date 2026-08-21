"""Pretraining loop.

    torchrun --standalone --nproc_per_node=3 -m src.train

Designed around one assumption: this run will be interrupted. Power cuts, driver
resets, someone else claiming a GPU, an OOM at step 30,000. So every restart is
exact -- same step, same optimizer state, same data order -- and the script
picks up the newest checkpoint automatically with no flags.
"""

import math
import os
import time
import glob
import shutil
from collections import deque

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing,
)

from .model import AbhiLLM, MAIN_1B, AbhiMoE, MOE_1B
from .model.moe_config import MOE_600M, MOE_MAIN
from .model.block import Block
from .model.moe_transformer import MoEBlock
from .train_config import TrainConfig
from .data.dataloader import TokenDataset


# ---------------------------------------------------------------------------
# distributed setup
# ---------------------------------------------------------------------------

# Which cards this project may touch. The set has changed as the box filled and
# emptied: 1-4 once held someone else's job and were off limits, and 7-10 were
# free. That reversed -- 7-10 are now busy and 1-4 are not.
#
# GPU 0 is off limits by instruction. GPU 2 runs the inference server behind the
# website, so a training run there would OOM the thing people are using. GPU 5
# is a smaller card and not part of this pool.
#
# This is enforced rather than documented, because "remember to set the env var"
# fails exactly once and the cost of that failure lands on someone else.
ALLOWED_GPUS = {"1", "3", "4"}


# ---------------------------------------------------------------------------
# wandb: one continuous run across every restart
# ---------------------------------------------------------------------------

def init_wandb(cfg, mcfg, world, start_step, resumed):
    """Attach to the same wandb run every time, so crashes do not fork the chart.

    The run id is written to disk on first launch and reused forever after.
    Without this, each restart creates a fresh run and you end up with five
    disconnected loss curves for what is really one training run.

    Metrics are logged with an explicit `step=`, so after a resume the curve
    continues from the step it died at rather than restarting at zero.
    """
    if not cfg.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without it", flush=True)
        return None

    id_path = os.path.join(cfg.out_dir, "wandb_run_id.txt")
    if os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
        # "must" fails loudly if the id is gone, rather than silently opening a
        # second run and splitting the history.
        resume_mode = "must"
    else:
        run_id = wandb.util.generate_id()
        with open(id_path, "w") as f:
            f.write(run_id)
        resume_mode = "allow"

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        id=run_id,
        resume=resume_mode,
        config={
            **{f"train/{k}": v for k, v in cfg.__dict__.items()},
            **{f"model/{k}": v for k, v in mcfg.__dict__.items()},
            "world_size": world,
            "total_params": mcfg.param_count()["total"],
            "active_params": mcfg.param_count()["active"],
            "global_batch_tokens": cfg.global_batch_tokens(world),
        },
    )
    # Panels keyed off the training step, not wandb's internal counter.
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    print(f"wandb: {'resumed' if resumed else 'started'} run {run_id} "
          f"-> {run.url}", flush=True)
    return run


def enforce_gpu_allowlist():
    # CUDA_DEVICE_ORDER must be PCI_BUS_ID. The default is FASTEST_FIRST, which
    # reorders devices by capability, so CUDA_VISIBLE_DEVICES indices do NOT
    # match nvidia-smi indices. On this machine that silently shifted a
    # "6,7,8,9" request onto physical GPUs 7,8,9,10 -- harmless by luck, but the
    # same shift could just as easily land on 1-4 and kill someone else's job.
    # It also made GPU 10 look broken during testing when the probe was in fact
    # hitting an occupied card.
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit(
            "CUDA_DEVICE_ORDER must be set to PCI_BUS_ID, or device indices do "
            "not mean what nvidia-smi says they mean.\nLaunch with:\n"
            "  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,4 \\\n"
            "  torchrun --standalone --nproc_per_node=3 -m src.train"
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is not set.\n"
            "GPU 0 is off limits and GPU 2 serves the website.\n"
            "Launch with:\n"
            "  CUDA_VISIBLE_DEVICES=1,3,4 torchrun --standalone "
            "--nproc_per_node=3 -m src.train"
        )
    requested = {d.strip() for d in visible.split(",") if d.strip()}
    forbidden = requested - ALLOWED_GPUS
    if forbidden:
        raise SystemExit(
            f"REFUSING TO START: CUDA_VISIBLE_DEVICES={visible} includes "
            f"GPU(s) {sorted(forbidden)}, which are off limits.\n"
            f"Only {sorted(ALLOWED_GPUS)} may be used by this project."
        )


def setup_dist():
    enforce_gpu_allowlist()
    if "RANK" not in os.environ:
        return 0, 0, 1, torch.device("cuda:0")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world, torch.device(f"cuda:{local_rank}")


def is_main(rank):
    return rank == 0


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def lr_at(step, cfg: TrainConfig, total_steps: int) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few hundred steps; taking full-size steps then is the classic way to blow up
    a run in its first hour.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(path, raw_model, optimizer, step, tokens_seen, cfg, mcfg, best_val):
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": mcfg.__dict__,
        "train_config": cfg.__dict__,
        "best_val": best_val,
    }, tmp)
    # Atomic rename: a checkpoint is either fully written or absent. Without
    # this, a crash mid-save leaves a corrupt file that breaks the next resume.
    os.replace(tmp, path)


def find_latest_ckpt(out_dir):
    files = glob.glob(os.path.join(out_dir, "ckpt_step_*.pt"))
    if not files:
        return None
    return max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))


def prune_ckpts(out_dir, keep_n):
    files = sorted(glob.glob(os.path.join(out_dir, "ckpt_step_*.pt")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    for f in files[:-keep_n]:
        os.remove(f)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_ds, cfg, device, n_batches):
    model.eval()
    losses = []
    for i in range(n_batches):
        x, y = val_ds.get_batch(cfg.micro_batch_size, step=i, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg = TrainConfig()
    if cfg.use_moe:
        mcfg = {"main": MOE_MAIN, "600m": MOE_600M, "1b": MOE_1B}[cfg.model_size]
    else:
        mcfg = MAIN_1B
    rank, local_rank, world, device = setup_dist()

    torch.manual_seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.out_dir, exist_ok=True)
    total_steps = cfg.total_steps(world)
    gbt = cfg.global_batch_tokens(world)

    log(rank, f"=== {cfg.run_name} ===")
    log(rank, f"world size      : {world}")
    log(rank, f"global batch    : {gbt/1e6:.2f}M tokens")
    log(rank, f"total steps     : {total_steps:,}")
    log(rank, f"total tokens    : {cfg.total_tokens/1e9:.0f}B")

    # ---- model ------------------------------------------------------------
    model = (AbhiMoE(mcfg) if cfg.use_moe else AbhiLLM(mcfg)).to(device)
    if cfg.use_moe:
        log(rank, f"architecture    : MoE ({mcfg.n_routed_experts} routed + "
                  f"{mcfg.n_shared_experts} shared, top-{mcfg.top_k}), MLA attention")
        log(rank, f"parameters      : {model.num_params()/1e9:.3f}B total, "
                  f"{model.num_active_params()/1e9:.3f}B active")
    else:
        log(rank, f"architecture    : dense, GQA attention")
        log(rank, f"parameters      : {model.num_params()/1e9:.3f}B")

    if cfg.activation_checkpointing:
        # Recompute activations in the backward pass instead of storing them.
        # Costs ~30% more compute, saves enough memory to roughly double the
        # micro-batch, which nets out clearly positive here.
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, (Block, MoEBlock)),
        )

    raw_model = model
    if cfg.compile_model:
        # MoE routing is data-dependent, so compile would recapture graphs
        # constantly. Marking it dynamic avoids that thrash.
        model = torch.compile(model, dynamic=True if cfg.use_moe else None)
    if world > 1:
        model = DDP(
            model, device_ids=[local_rank], gradient_as_bucket_view=True,
            # REQUIRED for MoE. Top-4 routing over 48 experts means some experts
            # receive zero tokens in a given micro-batch, so their parameters
            # get no gradient and their DDP buckets never become ready. Ranks
            # skip *different* experts, so the collectives desync and the run
            # dies with
            #   Watchdog caught collective operation timeout: ALLREDUCE ...
            # after the 10-minute NCCL timeout. find_unused_parameters makes DDP
            # walk the autograd graph each step to discover which parameters
            # actually participated. It costs a few percent throughput and is
            # not optional here.
            find_unused_parameters=True,
            # Buffers (router bias, load counters) are updated by an explicit
            # rule on every rank, not by autograd. Broadcasting them from rank 0
            # each forward is both wasted bandwidth and wrong -- rank 0's token
            # counts are not the global ones.
            broadcast_buffers=False,
        )

    optimizer = raw_model.configure_optimizer(
        cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))

    # ---- resume -----------------------------------------------------------
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    latest = find_latest_ckpt(cfg.out_dir)
    if latest:
        log(rank, f"resuming from   : {latest}")
        ck = torch.load(latest, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        tokens_seen = ck["tokens_seen"]
        best_val = ck.get("best_val", float("inf"))
        log(rank, f"                  step {start_step:,}, "
                  f"{tokens_seen/1e9:.2f}B tokens already seen")
    else:
        log(rank, "starting fresh")

    # ---- wandb ------------------------------------------------------------
    # Rank 0 only: every rank logging would produce five copies of every point.
    wandb_run = init_wandb(cfg, mcfg, world, start_step, bool(latest)) if is_main(rank) else None

    # ---- data -------------------------------------------------------------
    train_ds = TokenDataset(cfg.data_dir, cfg.seq_len, rank, world, cfg.seed, "train")
    val_ds = TokenDataset(cfg.data_dir, cfg.seq_len, rank, world, cfg.seed, "val")
    log(rank, f"train data      : {train_ds.stats()}")

    # ---- loop -------------------------------------------------------------
    model.train()
    loss_history = deque(maxlen=100)
    spike_count = 0
    t0 = time.time()
    flops_per_tok = raw_model.flops_per_token()

    for step in range(start_step, total_steps):
        lr = lr_at(step, cfg, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum_steps):
            x, y = train_ds.get_batch(cfg.micro_batch_size, step, device)

            # Only sync gradients on the final micro-step. Without this, DDP
            # all-reduces on every micro-step and throughput collapses --
            # especially here, where the GPUs talk over PCIe rather than NVLink.
            if world > 1:
                model.require_backward_grad_sync = (micro == cfg.grad_accum_steps - 1)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Load balancing: shifts routing only, no gradient involved. Must run
        # every step or experts drift into collapse.
        if cfg.use_moe:
            raw_model.update_router_bias()

        tokens_seen += gbt

        # ---- loss spike guard --------------------------------------------
        if len(loss_history) == loss_history.maxlen:
            avg = sum(loss_history) / len(loss_history)
            if accum_loss > avg * cfg.spike_threshold:
                spike_count += 1
                log(rank, f"  ! loss spike at step {step}: "
                          f"{accum_loss:.4f} vs avg {avg:.4f} "
                          f"({spike_count}/{cfg.spike_patience})")
                if spike_count >= cfg.spike_patience:
                    log(rank, "  ! persistent spike -- reverting to last checkpoint")
                    ck_path = find_latest_ckpt(cfg.out_dir)
                    if ck_path:
                        ck = torch.load(ck_path, map_location=device, weights_only=False)
                        raw_model.load_state_dict(ck["model"])
                        optimizer.load_state_dict(ck["optimizer"])
                    spike_count = 0
                    continue
            else:
                spike_count = 0
        loss_history.append(accum_loss)

        # ---- logging -------------------------------------------------------
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_s = gbt * cfg.log_every / dt if step > start_step else 0
            mfu = (flops_per_tok * tok_per_s) / (world * 91e12) if tok_per_s else 0
            eta_h = (total_steps - step) * (dt / cfg.log_every) / 3600 if tok_per_s else 0
            log(rank, f"step {step:6d}/{total_steps} | loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | gn {grad_norm:.2f} | "
                      f"{tok_per_s/1e3:.0f}K tok/s | mfu {mfu*100:.1f}% | "
                      f"{tokens_seen/1e9:.2f}B seen | eta {eta_h/24:.1f}d")

            if wandb_run is not None:
                rs = raw_model.router_stats() if cfg.use_moe else {}
                wandb_run.log({
                    "step": step,
                    "train/loss": accum_loss,
                    "train/perplexity": math.exp(min(accum_loss, 20)),
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_seen": tokens_seen,
                    "perf/tokens_per_sec": tok_per_s,
                    "perf/mfu": mfu,
                    "perf/eta_days": eta_h / 24,
                    **({"router/max_load": rs["max_frac"],
                        "router/min_load": rs["min_frac"],
                        "router/dead_experts": rs["dead"]} if rs else {}),
                }, step=step)

        # ---- router health -------------------------------------------------
        # The single most important MoE diagnostic. With 20 experts, uniform
        # load is 5% each. max_frac climbing past ~0.20, or any dead expert,
        # means routing is collapsing -- capacity you paid for is being wasted.
        if cfg.use_moe and step > 0 and step % cfg.router_log_every == 0:
            rs = raw_model.router_stats()
            warn = "  <-- COLLAPSE RISK" if (rs["max_frac"] > 0.20 or rs["dead"]) else ""
            log(rank, f"  router: load {rs['min_frac']*100:.1f}%-{rs['max_frac']*100:.1f}% "
                      f"(uniform {100/mcfg.n_routed_experts:.1f}%), "
                      f"dead {rs['dead']}{warn}")

        # ---- eval ----------------------------------------------------------
        if step > 0 and step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_ds, cfg, device, cfg.eval_batches)
            ppl = math.exp(min(val_loss, 20))
            log(rank, f"  eval @ {step}: loss {val_loss:.4f}  ppl {ppl:.2f}")
            best_val = min(best_val, val_loss)
            if wandb_run is not None:
                wandb_run.log({"step": step, "val/loss": val_loss,
                               "val/perplexity": ppl, "val/best_loss": best_val},
                              step=step)

        # ---- checkpoint ----------------------------------------------------
        if step > 0 and step % cfg.ckpt_every == 0 and is_main(rank):
            path = os.path.join(cfg.out_dir, f"ckpt_step_{step:07d}.pt")
            save_ckpt(path, raw_model, optimizer, step, tokens_seen, cfg, mcfg, best_val)
            prune_ckpts(cfg.out_dir, cfg.keep_last_n)
            log(rank, f"  saved {path}")

        if world > 1:
            dist.barrier()

    # ---- final -------------------------------------------------------------
    if is_main(rank):
        path = os.path.join(cfg.out_dir, "final.pt")
        save_ckpt(path, raw_model, optimizer, total_steps, tokens_seen, cfg, mcfg, best_val)
        log(rank, f"done -> {path}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
