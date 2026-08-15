"""Memory-mapped token loader.

The tokenized corpus is ~200GB, far larger than RAM. np.memmap lets us treat it
as one long array while the OS pages in only what we touch. With 1TB of RAM the
page cache will end up holding a large fraction of it, so after the first epoch
reads are essentially free.

Each rank reads a disjoint slice of every batch, so no two GPUs ever see the
same tokens in the same step.
"""

import json
import os

import numpy as np
import torch


class TokenDataset:
    def __init__(self, data_dir: str, seq_len: int, rank: int = 0, world_size: int = 1,
                 seed: int = 1337, split: str = "train", val_shards: int = 2):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)

        shards = manifest["shards"]
        # Hold out the last few shards for validation. They were produced from
        # the same interleaved stream, so they are a fair sample of the mixture.
        if split == "train":
            shards = shards[:-val_shards] if val_shards else shards
        else:
            shards = shards[-val_shards:]

        self.arrays = [
            np.memmap(os.path.join(data_dir, s["file"]), dtype=np.uint16, mode="r")
            for s in shards
        ]
        self.lengths = [len(a) for a in self.arrays]
        self.total_tokens = sum(self.lengths)
        self.split = split

        # Sequence starts are strided by seq_len, so sequences never overlap.
        self.seqs_per_shard = [l // (seq_len + 1) for l in self.lengths]
        self.total_seqs = sum(self.seqs_per_shard)

    def __len__(self):
        return self.total_seqs

    def _locate(self, idx):
        for si, n in enumerate(self.seqs_per_shard):
            if idx < n:
                return si, idx
            idx -= n
        raise IndexError

    def get_batch(self, batch_size: int, step: int, device="cuda"):
        """Deterministic given (step, rank) -- this is what makes resume exact.

        A resumed run at step N draws exactly the batch it would have drawn had
        it never crashed, so a restart does not silently re-show or skip data.
        """
        g = np.random.default_rng(self.seed + step * 100003 + self.rank)
        idxs = g.integers(0, self.total_seqs, size=batch_size)

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)

        for i, idx in enumerate(idxs):
            si, off = self._locate(int(idx))
            start = off * (self.seq_len + 1)
            chunk = self.arrays[si][start:start + self.seq_len + 1].astype(np.int64)
            # Targets are inputs shifted by one: predict token t+1 from token t.
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if device == "cuda":
            # pin + non_blocking overlaps the host->device copy with compute.
            xt = xt.pin_memory().to(device, non_blocking=True)
            yt = yt.pin_memory().to(device, non_blocking=True)
        return xt, yt

    def stats(self):
        return {
            "split": self.split,
            "shards": len(self.arrays),
            "tokens": self.total_tokens,
            "sequences": self.total_seqs,
            "gb_on_disk": self.total_tokens * 2 / 1e9,
        }
