"""Measure whether retrieval actually finds the right passage.

    python -m src.rag.eval_retrieval --index rag_index_big

Reports recall@k against Wikipedia questions whose answer article is known:
a hit means at least one of the top-k chunks came from the right article.

This is the metric that explains the failure seen in manual testing, where
"What is the capital of Japan?" retrieved an unrelated article at similarity
0.65 and the model then answered "Kokubunji" -- worse than the Tokyo it gets
with no context at all. Retrieval quality, not the generator, is the ceiling
on a RAG system, and a confident-but-wrong retrieval is worse than an empty
one, so the miss rate matters as much as recall.
"""

import argparse

from .retrieve import Retriever


# (question, title that should be retrieved). Titles are matched case-folded
# and by prefix, so "Tokyo" matches the article "Tokyo".
PROBES = [
    ("What is the capital of Japan?", "tokyo"),
    ("Who was Albert Einstein?", "albert einstein"),
    ("Who was Alan Turing?", "alan turing"),
    ("What is a black hole?", "black hole"),
    ("What is the Eiffel Tower?", "eiffel tower"),
    ("What is photosynthesis?", "photosynthesis"),
    ("Where is the Great Wall of China?", "great wall of china"),
    ("What is the Amazon rainforest?", "amazon rainforest"),
    ("Who wrote Hamlet?", "hamlet"),
    ("What is DNA?", "dna"),
    ("What is the Moon?", "moon"),
    ("What causes earthquakes?", "earthquake"),
    ("What is the Roman Empire?", "roman empire"),
    ("What is gravity?", "gravit"),
    ("What is the human heart?", "heart"),
    ("What is the Pacific Ocean?", "pacific ocean"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="rag_index_big")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    r = Retriever(args.index)
    print(f"index: {len(r.chunks):,} chunks\n")

    hits = 0
    misses = []
    for q, want in PROBES:
        found = r.search(q, k=args.k)
        titles = [src for _, _, src in found]
        ok = any(want in t.lower() for t in titles)
        hits += ok
        top = f"{found[0][0]:.2f} {found[0][2]}" if found else "no hits"
        print(f"{'HIT ' if ok else 'MISS'}  {q:<38} top: {top}")
        if not ok:
            misses.append((q, want, top))

    n = len(PROBES)
    print(f"\nrecall@{args.k}: {hits}/{n} = {hits/n*100:.0f}%")
    if misses:
        # A miss with a high top score is the dangerous kind: the retriever is
        # confident, so the floor in Retriever.search will not filter it out and
        # the wrong passage reaches the model as if it were an answer.
        print("\nmisses (these actively degrade answers):")
        for q, want, top in misses:
            print(f"  {q}\n    wanted '{want}', got {top}")


if __name__ == "__main__":
    main()
