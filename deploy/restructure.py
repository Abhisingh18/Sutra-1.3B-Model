"""Reorganise the Hub repo into a proper layout.

    python deploy/restructure.py

Target:

    README.md              model card
    config.json            architecture
    tokenizer.json
    model.safetensors      default weights (the DPO stage)
    inference.py           run it
    src/                   model code -- custom arch, transformers cannot load it
    checkpoints/           every training stage, for comparison
        base_pretrained.pt
        sft_epoch_{0,1,2}.pt
        dpo_epoch_0.pt

The root holds exactly what someone needs to RUN the model; `checkpoints/` holds
the archive. Before this, 26 GB of stage checkpoints sat next to the entry point
with nothing marking which one to use.

Copies happen server-side via CommitOperationCopy, so nothing is re-uploaded --
the 21 GB never leaves the Hub.
"""

import os

from huggingface_hub import (
    CommitOperationCopy, CommitOperationDelete, HfApi,
)

REPO = "Abhisingh-18/Sutra-1.3B-Chat"

# Files that move from the root into checkpoints/.
MOVES = [
    "base_pretrained.pt",
    "sft_epoch_0.pt",
    "sft_epoch_1.pt",
    "sft_epoch_2.pt",
    "dpo_epoch_0.pt",
]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN")
    api = HfApi(token=token)

    present = set(api.list_repo_files(REPO))
    ops = []
    for name in MOVES:
        if name not in present:
            print(f"skip {name} (not at root)")
            continue
        ops.append(CommitOperationCopy(
            src_path_in_repo=name, path_in_repo=f"checkpoints/{name}"))
        ops.append(CommitOperationDelete(path_in_repo=name))

    if not ops:
        print("nothing to move")
        return

    # Copy and delete land in ONE commit, so the repo is never in a state where
    # a checkpoint has been removed but its copy does not exist yet.
    api.create_commit(
        repo_id=REPO, repo_type="model", operations=ops,
        commit_message="Move stage checkpoints into checkpoints/",
    )
    print(f"moved {len(MOVES)} checkpoints into checkpoints/")
    for f in sorted(api.list_repo_files(REPO)):
        print(" ", f)


if __name__ == "__main__":
    main()
