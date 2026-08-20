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
import re

import numpy as np

# Words that carry no topic. A query is reduced to what is left, and that is
# what the title gate matches against.
_STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by",
    "with", "about", "into", "over", "under", "and", "or", "but", "if",
    "that", "this", "these", "those", "it", "its", "as", "than", "then",
    "there", "here", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "have", "has", "had", "me", "my", "you", "your",
    "i", "we", "our", "they", "their", "he", "she", "his", "her",
    "tell", "explain", "describe", "give", "list", "name", "mean", "means",
}


def _content_words(text, min_len=3):
    """Topic-bearing words, lowercased."""
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= min_len and w not in _STOP]


def _title_matches(query, title):
    """Does this article plausibly answer this query?

    Cosine similarity alone cannot tell. Asked for the capital of Japan the
    index returned "Fuji (spacecraft)" at 0.72, and for the Roman Empire a
    Duke of Wurttemberg at 0.90 -- both far above any workable score floor,
    both useless. What separates them is not the score but whether the article
    is ABOUT what was asked, and the title carries that.

    Substring matching in both directions on purpose: "earthquakes" has to
    match "Quake (natural phenomenon)", and "Einstein" has to match "Albert
    Einstein".
    """
    q_words = _content_words(query)
    if not q_words:
        return True          # nothing to check against; do not block

    title_low = title.lower()
    query_low = " ".join(q_words)
    t_words = _content_words(title)

    # Matching in both directions: "Einstein" has to find "Albert Einstein",
    # and "earthquakes" has to find "Quake (natural phenomenon)".
    matched = sum(1 for w in q_words
                  if w in title_low or any(w in t or t in w for t in t_words))

    # One word in common is not enough. "Amazon River" shares a word with
    # "Amazon rainforest", "Superocean" with "Pacific Ocean", "History of
    # China" with "Great Wall of China" -- every one of those passed a
    # single-word test and handed the model the wrong article. Ask for most of
    # what was actually named: everything for a one- or two-word topic, half
    # for a longer one.
    need = len(q_words) if len(q_words) <= 2 else (len(q_words) + 1) // 2
    return matched >= need


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

    def search(self, query, k=3, min_score=0.25, gate=True):
        q = self.enc.encode([query], convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
        # Both sides are unit vectors, so this dot product is cosine similarity.
        scores = self.embeds @ q[0]
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        hits = [(float(scores[i]), self.chunks[i], self.sources[i])
                for i in idx if scores[i] >= min_score]

        if gate:
            kept = [h for h in hits if _title_matches(query, h[2])]
            # All or nothing. A partial set still hands the model a passage
            # about something else, and it will use it -- so when nothing
            # passes the gate, return nothing and let the model answer from
            # its own weights, which at least stays on topic.
            hits = kept

        return hits


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
