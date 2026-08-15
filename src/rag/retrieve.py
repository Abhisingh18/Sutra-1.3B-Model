"""Query the index built by src.rag.ingest.

    from src.rag.retrieve import Retriever
    r = Retriever("rag_index")
    for score, text, source in r.search("how do I boil an egg", k=3):
        ...

No FAISS. At this corpus size a dense matrix multiply against normalised
vectors is a few milliseconds and beats the cost of another dependency; the
1.3B generator dominates latency either way.
"""

import json
import os

import numpy as np


class Retriever:
    def __init__(self, index_dir, device="cuda"):
        with open(os.path.join(index_dir, "chunks.json")) as f:
            meta = json.load(f)
        self.chunks = meta["chunks"]
        self.sources = meta["sources"]
        self.embeds = np.load(os.path.join(index_dir, "embeds.npy"))

        from sentence_transformers import SentenceTransformer
        # Must be the model the index was built with -- embeddings from two
        # different encoders are not comparable, and the failure is silent:
        # retrieval simply returns irrelevant chunks.
        self.enc = SentenceTransformer(meta["model"], device=device)

    def search(self, query, k=3, min_score=0.25):
        q = self.enc.encode([query], convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
        # Both sides are unit vectors, so this dot product is cosine similarity.
        scores = self.embeds @ q[0]
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        # Below the floor the "context" is noise, and handing a 1.3B model
        # irrelevant text makes its answer worse than no context at all.
        return [(float(scores[i]), self.chunks[i], self.sources[i])
                for i in idx if scores[i] >= min_score]


def build_prompt(question, hits, max_chars=1800):
    """Fold retrieved passages into the user turn.

    Deliberately label-free. Templates that wrap the passages in "Context:" and
    "Question:" scaffolding measurably worse here: this model echoes the
    scaffolding back instead of answering, opening replies with "The relevant
    information to answer the above question is:" and then quoting. Given bare
    passages and one instruction, it answers in its own words.

    Source titles are dropped for the same reason -- a leading "[Alan Turing]"
    gets copied into the reply as if it were part of the answer.
    """
    if not hits:
        return question
    ctx, used = [], 0
    for _, text, _src in hits:
        if used + len(text) > max_chars:
            break
        ctx.append(text)
        used += len(text)
    joined = "\n".join(ctx)
    return (f"{joined}\n\n"
            f"Based on the passage above, answer in your own words: {question}")
