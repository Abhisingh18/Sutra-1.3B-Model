"""Push the model and the Space to HuggingFace.

    export HF_TOKEN=hf_...
    python deploy/push.py --user YOUR_USERNAME

Creates two repos:

    <user>/Sutra-1.3B-Chat     model weights + tokenizer
    <user>/Sutra-1.3B-Demo     the Gradio Space

The Space gets a COPY of src/model and src/tokenizer, because the architecture
is custom and cannot be loaded by `transformers`. Only the modules the loader
actually imports are copied -- the training code has no business in a Space.
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

# Everything src.model.loader and src.tokenizer.special_tokens pull in.
NEEDED = [
    "src/__init__.py",
    "src/model/__init__.py",
    "src/model/loader.py",
    "src/model/config.py",
    "src/model/moe_config.py",
    "src/model/moe_transformer.py",
    "src/model/moe.py",
    "src/model/mla.py",
    "src/model/norm.py",
    "src/model/rope.py",
    "src/model/attention.py",
    "src/model/ffn.py",
    "src/model/block.py",
    "src/model/transformer.py",
    "src/tokenizer/__init__.py",
    "src/tokenizer/special_tokens.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model-repo", default="Sutra-1.3B-Chat")
    ap.add_argument("--space-repo", default="Sutra-1.3B-Demo")
    ap.add_argument("--ckpt", default="checkpoints/dpo/dpo_epoch_0.pt")
    ap.add_argument("--skip-model", action="store_true",
                    help="Space only; use when the weights are already up")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (huggingface.co/settings/tokens, write)")

    api = HfApi(token=token)
    model_id = f"{args.user}/{args.model_repo}"
    space_id = f"{args.user}/{args.space_repo}"

    if not args.skip_model:
        print(f"creating {model_id}")
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        print(f"uploading {args.ckpt} (5.3 GB, this takes a while)...")
        api.upload_file(path_or_fileobj=args.ckpt,
                        path_in_repo=os.path.basename(args.ckpt),
                        repo_id=model_id, repo_type="model")
        api.upload_file(path_or_fileobj="tokenizer/tokenizer.json",
                        path_in_repo="tokenizer.json",
                        repo_id=model_id, repo_type="model")
        api.upload_file(path_or_fileobj="deploy/README.md",
                        path_in_repo="README.md",
                        repo_id=model_id, repo_type="model")
        print(f"  -> https://huggingface.co/{model_id}")

    print(f"\ncreating space {space_id}")
    api.create_repo(space_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for rel in NEEDED:
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(rel, dst)
        shutil.copy("deploy/requirements.txt", os.path.join(tmp, "requirements.txt"))
        shutil.copy("deploy/README.md", os.path.join(tmp, "README.md"))

        # Point the app at the repo we just created.
        app = open("deploy/app.py").read().replace(
            "REPLACE_ME/Sutra-1.3B-Chat", model_id)
        with open(os.path.join(tmp, "app.py"), "w") as f:
            f.write(app)

        api.upload_folder(folder_path=tmp, repo_id=space_id, repo_type="space")

    print(f"  -> https://huggingface.co/spaces/{space_id}")
    print("\nThe Space builds for a few minutes, then downloads 5.3 GB of "
          "weights on first start. Watch its Logs tab.")


if __name__ == "__main__":
    main()
