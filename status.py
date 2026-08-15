#!/usr/bin/env python3
"""Live status of the whole pipeline.

    python status.py            once
    python status.py -w         refresh every 10s
    python status.py -w -n 30   refresh every 30s

Reads only from disk and nvidia-smi, so it is safe to run at any time and
cannot disturb a running job.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data/tokens/raw")
SHARDS = os.path.join(ROOT, "data/tokens")
TARGET_TOKENS = 25e9

# Only these GPUs belong to this project; the rest are someone else's.
OURS = {"6", "7", "8", "9", "10"}


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.2f}{unit}"
    return str(int(n))


def bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def proc_running(pattern):
    try:
        out = subprocess.run(["pgrep", "-fc", pattern], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out or 0)
    except Exception:
        return 0


def section_data():
    print("=" * 72)
    print("DATA PREP")
    print("=" * 72)

    running = proc_running("src.data.prepare")
    print(f"  process       : {'RUNNING' if running else 'not running'}")

    if not os.path.isdir(RAW):
        print("  no output yet\n")
        return

    files = sorted(glob.glob(os.path.join(RAW, "*.bin")))
    partial = sorted(glob.glob(os.path.join(RAW, "*.bin.tmp")))
    total = 0

    try:
        sys.path.insert(0, ROOT)
        from src.data.mixture import MIXTURE
        from src.data.prepare import safe_name
        quotas = {safe_name(s): TARGET_TOKENS * s.weight * s.epochs for s in MIXTURE}
        order = [safe_name(s) for s in MIXTURE]
    except Exception:
        quotas, order = {}, []

    seen = {}
    for p in files + partial:
        name = os.path.basename(p).replace(".bin.tmp", "").replace(".bin", "")
        toks = os.path.getsize(p) // 2
        seen[name] = (toks, p.endswith(".tmp"))
        total += toks

    print(f"  {'source':46s} {'tokens':>9s}  {'quota':>7s}")
    print("  " + "-" * 66)
    for name in order or sorted(seen):
        toks, is_tmp = seen.get(name, (0, False))
        q = quotas.get(name, 0)
        pct = 100 * toks / q if q else 0
        mark = " <- fetching" if is_tmp else ""
        short = "  SHORT" if (not is_tmp and toks and q and pct < 80) else ""
        print(f"  {name[:46]:46s} {human(toks):>9s}  {pct:6.0f}%{mark}{short}")

    print("  " + "-" * 66)
    frac = total / TARGET_TOKENS
    print(f"  {'TOTAL':46s} {human(total):>9s}  {frac*100:6.1f}%")
    print(f"  {bar(frac)} {human(total)} / {human(TARGET_TOKENS)}")
    print(f"  on disk       : {total*2/1e9:.1f} GB")

    shards = sorted(glob.glob(os.path.join(SHARDS, "shard_*.bin")))
    if shards:
        st = sum(os.path.getsize(s) for s in shards) // 2
        print(f"  shards built  : {len(shards)} ({human(st)} tokens)")
    manifest = os.path.join(SHARDS, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        print(f"  manifest      : COMPLETE, {human(m['total_tokens'])} tokens, "
              f"{len(m['shards'])} shards")
    print()


def section_train():
    print("=" * 72)
    print("PRETRAINING")
    print("=" * 72)
    running = proc_running("src.train")
    print(f"  process       : {'RUNNING' if running else 'not running'}")

    ckpts = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "ckpt_step_*.pt")))
    if ckpts:
        latest = max(ckpts, key=lambda f: int(re.findall(r"(\d+)", f)[-1]))
        step = int(re.findall(r"(\d+)", latest)[-1])
        print(f"  checkpoints   : {len(ckpts)}, latest step {step:,}")
    else:
        print("  checkpoints   : none yet")

    log = os.path.join(ROOT, "logs_train.txt")
    if os.path.exists(log):
        with open(log) as f:
            lines = [l.rstrip() for l in f if l.startswith("step ")
                     or "router:" in l or "eval @" in l]
        for l in lines[-6:]:
            print(f"    {l}")
    print()


def section_gpu():
    print("=" * 72)
    print("GPUs")
    print("=" * 72)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,"
             "utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        print(f"  nvidia-smi unavailable: {e}\n")
        return

    for line in out.splitlines():
        idx, used, tot, util = [x.strip() for x in line.split(",")]
        ours = idx in OURS
        tag = "OURS   " if ours else "others "
        used_i, tot_i = int(used), int(tot)
        warn = ""
        if not ours and used_i > 1000:
            warn = "  (in use - do not touch)"
        print(f"  GPU {idx:>2} {tag} {used_i:6d}/{tot_i} MiB  {util:>3}%{warn}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--watch", action="store_true")
    ap.add_argument("-n", "--interval", type=int, default=10)
    args = ap.parse_args()

    while True:
        if args.watch:
            os.system("clear")
        print(f"Sutra-1.3B pipeline status   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        section_data()
        section_train()
        section_gpu()
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
