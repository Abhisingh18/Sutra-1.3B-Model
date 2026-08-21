"""Pull the answering sentence out of a passage, verbatim.

Retrieval keeps working and the model keeps failing on the same step: asked
for the self-attention formula it was handed a passage reading
"Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)) . V" and replied
"Attention (Q, K) = C_K W - C_V T".

Nothing in a prompt fixes that. What fixes it is not asking the model to
reproduce the sentence at all: find the sentence in the retrieved text that
best answers the question, and put it on screen exactly as written. Copying in
Python is exact, which is the one guarantee the model cannot offer.

The model still answers underneath. This is the checkable version above it.
"""

import re

# A sentence ends with punctuation attached to a word -- "...enriched by the
# context." The alphanumeric lookbehind is what keeps formulas whole: in
# "softmax((Q . K^T) / sqrt(d_k)) . V" the dots are multiplication operators
# with a space in front, and splitting there truncated the one sentence the
# reader came for down to "Attention(Q, K, V) = softmax((Q ."
_SPLIT = re.compile(r"(?<=[a-zA-Z0-9\)\]][.!?])\s+(?=[A-Z(\\$])")

_STOP = {
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "that", "this", "it", "its", "as", "do", "does", "did",
    "tell", "explain", "give", "show", "me", "my", "you", "your",
}


def _sentences(text, min_len=40, max_len=400):
    out = []
    for raw in _SPLIT.split(text.replace("\n", " ")):
        s = " ".join(raw.split())
        if min_len <= len(s) <= max_len:
            out.append(s)
    return out


def _words(text):
    return {w for w in re.findall(r"[a-z0-9_^]+", text.lower())
            if len(w) > 2 and w not in _STOP}


# What the question asks for, and the shape of a sentence that supplies it.
# Embedding similarity alone gets this wrong in a specific, repeatable way:
# asked for the self-attention formula it ranked "In simple words, Self
# Attention = Self + Attention" above
# "Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)) . V", because MiniLM
# reads prose far better than it reads symbols. If someone asks for a formula,
# a sentence carrying an actual expression is the better answer whatever the
# cosine says.
_INTENT = [
    (("formula", "equation", "expression"),
     re.compile(r"[=∑√]\s*\S|\bsoftmax\b|\bsqrt\b")),
    (("when", "year", "date", "born", "founded"), re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")),
    (("many", "much", "count", "number", "percent", "cost", "price"), re.compile(r"\d")),
]


def _intent_bonus(query, sentence):
    q = query.lower()
    bonus = 0.0
    for words, pattern in _INTENT:
        if any(w in q for w in words) and pattern.search(sentence):
            bonus += 0.2
    return bonus


def best_quote(query, hits, encoder=None):
    """The sentence most likely to answer `query`, or None.

    With an encoder it scores by embedding similarity; without one it falls
    back to word overlap. The fallback matters -- web results arrive with no
    index loaded behind them.
    """
    candidates = []
    for _score, text, source in hits:
        for s in _sentences(text):
            candidates.append((s, source))
    if not candidates:
        return None

    sentences = [s for s, _ in candidates]

    if encoder is not None:
        try:
            import numpy as np
            embeds = encoder.encode(sentences + [query], convert_to_numpy=True,
                                    normalize_embeddings=True)
            sims = embeds[:-1] @ embeds[-1]
            # Cosine ranks prose above symbols, so it cannot be the only vote.
            # The intent bonus is what puts the equation ahead of the sentence
            # that merely talks about the equation.
            ranked = [(float(sims[i]) + _intent_bonus(query, sentences[i]), i)
                      for i in range(len(sentences))]
            best, idx = max(ranked)
            # Floor applies to the cosine, not the bonus -- a bonus must not
            # promote a sentence that was never relevant.
            if float(sims[idx]) < 0.3:
                return None
            return {"text": candidates[idx][0], "source": candidates[idx][1],
                    "score": round(float(sims[idx]), 2)}
        except Exception:
            pass

    q_words = _words(query)
    if not q_words:
        return None
    scored = []
    for s, source in candidates:
        overlap = len(q_words & _words(s)) / len(q_words)
        scored.append((overlap + _intent_bonus(query, s), overlap, s, source))
    _rank, overlap, s, source = max(scored)
    # Below this the "answer" is a sentence that happens to share a word, which
    # is worse than showing nothing and letting the reply stand alone.
    if overlap < 0.4:
        return None
    return {"text": s, "source": source, "score": round(overlap, 2)}
