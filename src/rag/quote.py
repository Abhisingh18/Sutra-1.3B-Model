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


# Search results arrive as extracted markdown, and it shows: heading markers,
# empty table rows, backslash-escaped underscores ("d\\_k" for "d_k"), and
# "[...]" where the extractor dropped text. Left in, all of it gets quoted
# verbatim -- which is the one thing this card promises not to garble.
_ELIDED = re.compile(r"\[\.\.\.\]")
_MD_NOISE = re.compile(r"^#+\s*|\|[\s|]*\||\\\"\)")
_MD_ESCAPE = re.compile(r"\\([_*\[\]()#.|+-])")


def _clean(text):
    text = text.replace("\n", " ")
    text = _MD_NOISE.sub(" ", text)
    text = _MD_ESCAPE.sub(r"\1", text)
    return text


def _sentences(text, min_len=40, max_len=400):
    out = []
    # Split on "[...]" before anything else. It marks text the extractor
    # dropped, so what sits either side of it is unrelated -- and since no
    # sentence-ending punctuation separates them, they otherwise fuse into a
    # single "sentence" and the card quotes two halves of different thoughts.
    for segment in _ELIDED.split(_clean(text)):
        for raw in _SPLIT.split(segment):
            s = " ".join(raw.split())
            if min_len <= len(s) <= max_len and not _LATEX.search(s):
                out.append(s)
    return out


def _words(text):
    return {w for w in re.findall(r"[a-z0-9_^]+", text.lower())
            if len(w) > 2 and w not in _STOP}


# Sentences that are mostly LaTeX markup. Wikipedia stores equations as
# "{\\displaystyle E_{\\text{k}}={\\tfrac {1}{2}}mv^{2}}", which is correct and
# completely unreadable on screen. Better to quote nothing than to quote that.
_LATEX = re.compile(r"\\displaystyle|\\tfrac|\\begin\{")

# Characters that appear in written mathematics but not in an English sentence.
_MATHY = re.compile(r"[()\[\]/^_|∑√·×÷≈≤≥]|\d")


def _is_formula(sentence):
    """Does this sentence carry an expression, rather than talk about one?

    An equals sign alone does not decide it: "In simple words, Self Attention =
    Self + Attention" has one, and ranking it as a formula is exactly the
    mistake that hid the real answer. What separates them is that either side
    of a real formula is symbols, not words -- so require the structural
    characters too.
    """
    return "=" in sentence and len(_MATHY.findall(sentence)) >= 4


# What the question asks for, and the shape of a sentence that supplies it.
# Embedding similarity alone gets this wrong in a specific, repeatable way:
# asked for the self-attention formula it ranked the plain-words restatement
# above "Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)) . V", because
# MiniLM reads prose far better than it reads symbols.
_INTENT = [
    (("formula", "equation", "expression"), _is_formula),
    (("when", "year", "date", "born", "founded"),
     lambda s: re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s) is not None),
    (("many", "much", "count", "number", "percent", "cost", "price"),
     lambda s: re.search(r"\d", s) is not None),
]


def _intent_bonus(query, sentence):
    q = query.lower()
    bonus = 0.0
    for words, matches in _INTENT:
        if any(w in q for w in words) and matches(sentence):
            bonus += 0.35
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
