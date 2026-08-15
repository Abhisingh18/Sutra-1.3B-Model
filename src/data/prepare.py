"""Stream, tokenize and shard the pretraining corpus.

Raw text is NEVER written to disk. It is streamed from HuggingFace, tokenized,
and only uint16 token ids are stored -- turning a ~500GB raw-text problem into a
50GB tokenized one.

TWO PHASES, and the split is not cosmetic:

  Phase 1 (fetch)  one source at a time, streamed and tokenized into its own
                   .bin file.
  Phase 2 (merge)  those per-source files are interleaved into training shards.

The obvious design -- open all 11 streaming datasets at once and sample between
them -- does not survive contact with the library. Each `datasets` stream runs
its own aiohttp worker threads, and eleven of them concurrently crashes the
interpreter with

    Fatal Python error: PyGILState_Release: thread state must be current

Phase 1 keeps exactly one stream open at a time, which is stable. Phase 2 then
recovers the interleaving that matters for training -- feeding all the web data
first and all the code last would make the model forget the web data -- using
only numpy and file IO, with no threads involved at all.

Phase 1 is resumable: a source whose .bin already exists is skipped, so a
network failure five sources in costs you one source, not the whole run.

Output:
    data/tokens/raw/<source>.bin    per-source tokens (phase 1)
    data/tokens/shard_00000.bin     interleaved training shards (phase 2)
    data/tokens/manifest.json       shard list, counts, per-source stats

Usage:
    python -m src.data.prepare --out data/tokens --tokens 25e9
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

from .mixture import MIXTURE, validate

SHARD_TOKENS = 250_000_000        # ~500MB per shard as uint16
TARGET_DOC_CHARS = 2000           # for gluing sentence-level corpora
MAX_BATCH_CHARS = 4_000_000       # ~1M tokens per write, keeps quotas accurate


def safe_name(src) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_",
                  f"{src.name}_{src.config}_{src.split}").strip("_")


# ---------------------------------------------------------------------------
# phase 1: fetch + tokenize, one source at a time
# ---------------------------------------------------------------------------

def fetch_source(src, tok, eos_id, target_tokens, out_dir, batch_size=2000):
    """Stream one source until `target_tokens` are collected. Returns count."""
    from datasets import load_dataset

    path = os.path.join(out_dir, f"{safe_name(src)}.bin")
    if os.path.exists(path):
        got = os.path.getsize(path) // 2
        print(f"  [skip] {safe_name(src)}: {got/1e6:.0f}M tokens already on disk",
              flush=True)
        return got

    label = f"{src.name}:{src.config}:{src.split}"
    print(f"  [fetch] {label} -> {target_tokens/1e6:.0f}M tokens", flush=True)

    ds = load_dataset(src.name, src.config, split=src.split, streaming=True)
    ds = ds.shuffle(seed=1337, buffer_size=10_000)

    tmp = path + ".tmp"
    fh = open(tmp, "wb")
    total = 0
    docs = 0
    batch = []
    batch_chars = 0
    sent_buf = []
    t0 = time.time()

    def write(texts):
        nonlocal total
        ids = []
        for enc in tok.encode_batch(texts, add_special_tokens=False):
            ids.extend(enc.ids)
            # Every document ends with EOS. Without it, unrelated documents
            # bleed into each other across the 4096-token window and the model
            # learns transitions that do not exist.
            ids.append(eos_id)
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(fh)
        total += len(arr)

    # Some sources are plain-text dumps containing invalid UTF-8, and the
    # reader raises partway through the stream. Whatever was tokenized before
    # that point is perfectly good data, so keep it, record the shortfall, and
    # move on -- one malformed source must not abort a multi-hour run. The
    # shortfall is reported at the end so a source that yields almost nothing
    # is impossible to miss.
    error = None
    try:
        for row in ds:
            text = row.get(src.text_field)
            if not text or len(text) < src.min_chars:
                continue
            docs += 1

            if src.sentence_level:
                # Sentence-level corpora would otherwise emit an EOS every ~40
                # characters, teaching the model that text constantly ends.
                sent_buf.append(text)
                if sum(len(t) for t in sent_buf) < TARGET_DOC_CHARS:
                    continue
                text = " ".join(sent_buf)
                sent_buf = []

            batch.append(text)
            batch_chars += len(text)
            # Cap batches by CHARACTERS, not document count. Document sizes
            # vary by four orders of magnitude across these sources -- Gutenberg
            # ships whole books (~1M chars each) while Wikipedia ships
            # paragraphs. A fixed 2000-document batch means one Gutenberg batch
            # is ~225M tokens, so a 2M-token quota overshoots by 100x and that
            # source swamps the mixture. Capping by characters keeps every
            # source's quota accurate to within one small batch.
            if len(batch) >= batch_size or batch_chars >= MAX_BATCH_CHARS:
                write(batch)
                batch = []
                batch_chars = 0
                if total >= target_tokens:
                    break
                if total and total % (50_000_000) < 2_000_000:
                    rate = total / max(time.time() - t0, 1)
                    print(f"      {total/1e6:6.0f}M / {target_tokens/1e6:.0f}M "
                          f"({rate/1e3:.0f}K tok/s)", flush=True)

        if batch and total < target_tokens:
            write(batch)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        fh.close()

    os.replace(tmp, path)
    pct = 100 * total / max(target_tokens, 1)
    status = "done" if pct >= 80 else "SHORT"
    print(f"  [{status}]  {safe_name(src)}: {total/1e6:.0f}M tokens "
          f"({pct:.0f}% of quota) from {docs:,} docs "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    if error:
        print(f"      stream ended early: {error}", flush=True)
    return total


# ---------------------------------------------------------------------------
# phase 2: interleave into shards
# ---------------------------------------------------------------------------

def merge_shards(out_dir, raw_dir, sources_info, seed=1337):
    """Interleave per-source token files into training shards.

    Reads each source in blocks and emits them in a weighted-random order, so
    every shard contains the full mixture rather than one source at a time.
    Pure numpy and file IO -- no threads, no network.
    """
    rng = np.random.default_rng(seed)
    BLOCK = 1_000_000                      # tokens per draw

    arrays, weights, names = [], [], []
    for info in sources_info:
        p = os.path.join(raw_dir, info["file"])
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        arrays.append(np.memmap(p, dtype=np.uint16, mode="r"))
        weights.append(info["weight"])
        names.append(info["label"])

    if not arrays:
        raise RuntimeError("phase 1 produced no data")

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    positions = [0] * len(arrays)
    exhausted = set()

    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    buf_n = 0
    shard_idx = 0
    total = 0
    manifest = []
    t0 = time.time()

    def flush():
        nonlocal buf_n, shard_idx
        if buf_n == 0:
            return
        path = os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")
        tmp = path + ".tmp"
        buf[:buf_n].tofile(tmp)
        os.replace(tmp, path)
        manifest.append({"file": os.path.basename(path), "tokens": int(buf_n)})
        print(f"  shard {shard_idx:05d}: {buf_n/1e6:.0f}M tokens | "
              f"total {total/1e9:.2f}B", flush=True)
        shard_idx += 1
        buf_n = 0

    print(f"\nphase 2: interleaving {len(arrays)} sources into shards", flush=True)
    while len(exhausted) < len(arrays):
        live = np.array([w if i not in exhausted else 0.0
                         for i, w in enumerate(weights)])
        if live.sum() == 0:
            break
        live /= live.sum()
        i = int(rng.choice(len(arrays), p=live))

        start = positions[i]
        end = min(start + BLOCK, len(arrays[i]))
        if start >= end:
            exhausted.add(i)
            continue
        block = arrays[i][start:end]
        positions[i] = end

        pos = 0
        while pos < len(block):
            take = min(SHARD_TOKENS - buf_n, len(block) - pos)
            buf[buf_n:buf_n + take] = block[pos:pos + take]
            buf_n += take
            pos += take
            total += take
            if buf_n == SHARD_TOKENS:
                flush()

    flush()
    print(f"phase 2 done in {(time.time()-t0)/60:.1f}m", flush=True)
    return manifest, total


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--tokens", type=float, default=25e9)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--phase", choices=["all", "fetch", "merge"], default="all")
    args = ap.parse_args()

    validate()
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size()
    assert vocab <= 65_535, f"vocab {vocab} does not fit in uint16"
    eos_id = tok.token_to_id("<|end_of_text|>")
    assert eos_id is not None, "tokenizer is missing <|end_of_text|>"

    target = int(args.tokens)
    print(f"target {target/1e9:.1f}B tokens, vocab {vocab}, eos={eos_id}\n")

    sources_info = []
    if args.phase in ("all", "fetch"):
        print("phase 1: fetch + tokenize (one source at a time)")
        shortfalls = []
        for src in MIXTURE:
            quota = int(target * src.weight * src.epochs)
            got = fetch_source(src, tok, eos_id, quota, raw_dir, args.batch)
            if got < quota * 0.8:
                shortfalls.append((f"{src.name}:{src.split}", got, quota))
            sources_info.append({
                "file": f"{safe_name(src)}.bin",
                "label": f"{src.name}:{src.config}:{src.split}",
                "weight": src.weight,
                "tokens": int(got),
            })

        if shortfalls:
            print("\n  sources that did not meet quota:")
            for label, got, quota in shortfalls:
                print(f"    {label}: {got/1e6:.0f}M / {quota/1e6:.0f}M "
                      f"({100*got/max(quota,1):.0f}%)")
            print("  The mixture will be renormalised over what was actually "
                  "collected, so proportions shift toward the sources that "
                  "succeeded.")
    else:
        for src in MIXTURE:
            p = os.path.join(raw_dir, f"{safe_name(src)}.bin")
            sources_info.append({
                "file": f"{safe_name(src)}.bin",
                "label": f"{src.name}:{src.config}:{src.split}",
                "weight": src.weight,
                "tokens": os.path.getsize(p) // 2 if os.path.exists(p) else 0,
            })

    if args.phase == "fetch":
        print("\nphase 1 complete; run with --phase merge to build shards")
        sys.stdout.flush()
        os._exit(0)

    manifest, total = merge_shards(args.out, raw_dir, sources_info)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({
            "total_tokens": int(total),
            "vocab_size": vocab,
            "eos_id": eos_id,
            "shards": manifest,
            "sources": sources_info,
        }, f, indent=2)

    print(f"\ndone: {total/1e9:.2f}B tokens in {len(manifest)} shards "
          f"({total*2/1e9:.0f}GB)", flush=True)

    # Hard exit, skipping interpreter finalization. `datasets` leaves aiohttp
    # worker threads alive and the Rust tokenizer keeps its own thread pool;
    # at shutdown these race with GIL teardown and abort the process. All data
    # is already flushed and renamed into place by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
