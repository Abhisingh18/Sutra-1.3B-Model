"""Build a retrieval index.

    python -m src.rag.ingest --source wikipedia --docs 20000
    python -m src.rag.ingest --source dir --path ./my_docs

Writes two files to --out:

    chunks.json   the text of every chunk, in index order
    embeds.npy    float32 [n_chunks, dim], L2-normalised

Nothing here touches the language model. The embedding model is a separate
90 MB encoder; the 1.3B model is never loaded, never trained, never modified.

Normalising the embeddings at write time is what lets retrieval be a single
matrix multiply later: with unit vectors, a dot product IS cosine similarity.
"""

import argparse
import json
import os

import numpy as np

# Small, fast, and good enough. The bottleneck in this system is the 1.3B
# generator, not the retriever, so a larger encoder would buy nothing.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text, size=900, overlap=150):
    """Split on character count with overlap.

    The overlap matters: a fact that straddles a chunk boundary is otherwise
    retrievable by neither half. 150 characters is roughly a sentence and a
    half, which is enough to keep most statements intact.
    """
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def load_wikipedia(n_docs):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    docs = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        docs.append((row["title"], row["text"]))
    return docs


def load_dir(path):
    docs = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if not f.lower().endswith((".txt", ".md")):
                continue
            p = os.path.join(root, f)
            with open(p, errors="ignore") as fh:
                docs.append((f, fh.read()))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wikipedia", "dir"], default="wikipedia")
    ap.add_argument("--path", help="directory of .txt/.md files, for --source dir")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--out", default="rag_index")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.source == "dir":
        if not args.path:
            raise SystemExit("--source dir requires --path")
        docs = load_dir(args.path)
    else:
        print(f"streaming {args.docs:,} wikipedia articles...", flush=True)
        docs = load_wikipedia(args.docs)
    print(f"{len(docs):,} documents", flush=True)

    chunks, sources = [], []
    for title, text in docs:
        for c in chunk_text(text, args.chunk_size, args.overlap):
            chunks.append(c)
            sources.append(title)
    print(f"{len(chunks):,} chunks", flush=True)
    if not chunks:
        raise SystemExit("nothing to index")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device=args.device)
    embeds = enc.encode(chunks, batch_size=args.batch_size,
                        show_progress_bar=True, convert_to_numpy=True,
                        normalize_embeddings=True).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    # Written to a temp name then renamed, so an interrupted run cannot leave a
    # half-written index that loads without error but returns garbage.
    np.save(os.path.join(args.out, "embeds.npy.tmp"), embeds)
    os.replace(os.path.join(args.out, "embeds.npy.tmp.npy"),
               os.path.join(args.out, "embeds.npy"))
    with open(os.path.join(args.out, "chunks.json"), "w") as f:
        json.dump({"chunks": chunks, "sources": sources,
                   "model": EMBED_MODEL}, f)

    print(f"\nindexed {len(chunks):,} chunks -> {args.out}/")
    print(f"  embeds.npy  {embeds.nbytes/1e6:.1f} MB  shape {embeds.shape}")


if __name__ == "__main__":
    main()
