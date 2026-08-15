"""Compare tokenizer vocabulary sizes before committing to one.

The tokenizer is the single irreversible decision in the project: changing it
later means re-tokenizing the corpus and retraining from zero. Devanagari needs
far more merges than Latin script, so a vocabulary tuned for English silently
makes every Hindi document 2-3x more expensive in tokens -- which means the
Hindi share of the token budget buys much less Hindi *content* than it looks
like on paper.

This trains several small candidates on the same sample and reports fertility,
so the choice is made on measurements rather than intuition.

    python -m src.tokenizer.vocab_experiment --sample-gb 2
"""

import argparse
import time

from tokenizers import (Tokenizer, models, pre_tokenizers, decoders, trainers,
                        normalizers)

from .special_tokens import ALL_SPECIALS

# Fixed evaluation text, so candidates are compared on identical input.
EVAL = {
    "en": "The capital of India is New Delhi, a city of over 32 million people "
          "spread across a large metropolitan region in the northern part of "
          "the country.",
    "hi": "भारत की राजधानी नई दिल्ली है, जो देश के उत्तरी भाग में स्थित एक बड़ा "
          "महानगर है जहाँ तीन करोड़ से अधिक लोग रहते हैं।",
    "code": "def fibonacci(n):\n    if n < 2:\n        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)",
    "math": "Solve for x: 3x^2 + 12x - 15 = 0, then verify that x = 1 and x = -5.",
}


def build(vocab_size):
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=ALL_SPECIALS, min_frequency=2,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    return tok, trainer


def corpus(sample_gb, hindi_boost):
    """Same mixture as training, with Hindi weighted up by `hindi_boost`."""
    from datasets import load_dataset
    from ..data.mixture import MIXTURE

    weights = []
    for s in MIXTURE:
        w = s.weight
        if "sangraha" in s.name or "Indic" in s.name or s.config == "20231101.hi":
            w *= hindi_boost
        weights.append(w)
    total = sum(weights)
    weights = [w / total for w in weights]

    budget = sample_gb * 1e9
    texts = []
    for src, w in zip(MIXTURE, weights):
        limit = budget * w
        try:
            ds = load_dataset(src.name, src.config, split=src.split, streaming=True)
        except Exception:
            continue
        used = 0
        for row in ds:
            t = row.get(src.text_field)
            if not t or len(t) < src.min_chars:
                continue
            used += len(t)
            if used > limit:
                break
            texts.append(t)
    return texts


def held_out_text(n_docs=150):
    """Real corpus text for measurement.

    A single hand-written sentence is not a measurement -- it uses only common
    words whose merges exist at any vocabulary size, which is exactly why the
    first version of this experiment reported 0.0% difference everywhere.
    """
    from datasets import load_dataset
    out = {}
    srcs = {
        "en": ("HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text"),
        "hi": ("ai4bharat/sangraha", "verified", "hin", "text"),
        "code": ("codeparrot/codeparrot-clean", None, "train", "content"),
        "math": ("open-web-math/open-web-math", "default", "train", "text"),
    }
    for lang, (name, cfg, split, field) in srcs.items():
        ds = load_dataset(name, cfg, split=split, streaming=True)
        buf = []
        # Skip the head of the stream: those documents went into training.
        for i, row in enumerate(ds):
            if i < 3000:
                continue
            t = row.get(field)
            if t and len(t) > 500:
                buf.append(t[:3000])
            if len(buf) >= n_docs:
                break
        out[lang] = buf
    return out


def evaluate(tok, held_out):
    out = {}
    for lang, docs in held_out.items():
        chars = sum(len(d) for d in docs)
        toks = sum(len(tok.encode(d, add_special_tokens=False).ids) for d in docs)
        out[lang] = {
            "tokens": toks,
            "chars_per_tok": chars / toks,
            "roundtrip": all(tok.decode(tok.encode(d, add_special_tokens=False).ids) == d
                             for d in docs[:20]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-gb", type=float, default=2.0)
    ap.add_argument("--candidates", default="48000:2,64000:2,64000:4,80000:4",
                    help="comma list of vocab_size:hindi_boost")
    args = ap.parse_args()

    specs = []
    for c in args.candidates.split(","):
        v, b = c.split(":")
        specs.append((int(v), float(b)))

    print("fetching held-out evaluation text...")
    held_out = held_out_text()
    for k, v in held_out.items():
        print(f"  {k}: {len(v)} docs, {sum(len(d) for d in v):,} chars")

    # Cache the corpus per boost value -- streaming is the slow part.
    cache = {}
    results = []
    for vocab, boost in specs:
        if boost not in cache:
            print(f"streaming {args.sample_gb}GB sample (hindi_boost={boost})...")
            t0 = time.time()
            cache[boost] = corpus(args.sample_gb, boost)
            print(f"  {len(cache[boost]):,} docs in {time.time()-t0:.0f}s")

        print(f"training vocab={vocab} boost={boost} ...", flush=True)
        tok, trainer = build(vocab)
        t0 = time.time()
        tok.train_from_iterator(cache[boost], trainer=trainer)
        actual = tok.get_vocab_size()
        res = evaluate(tok, held_out)
        results.append((vocab, actual, boost, res, time.time() - t0))
        print(f"  -> actual vocab {actual:,} "
              f"({'SATURATED - corpus too small' if actual < vocab - 100 else 'reached target'})")

    langs = list(held_out)
    print(f"\n{'asked':>7} {'actual':>7} {'boost':>6} " +
          " ".join(f"{k:>10}" for k in langs) + "   roundtrip")
    print("-" * 76)
    for vocab, actual, boost, res, _ in results:
        cells = " ".join(f"{res[k]['chars_per_tok']:10.2f}" for k in langs)
        rt = "ok" if all(res[k]["roundtrip"] for k in langs) else "FAIL"
        print(f"{vocab:7d} {actual:7d} {boost:6.1f} {cells}   {rt}")
    print("\n(CHARS PER TOKEN on held-out real text -- higher is better)")

    base = results[0]
    print(f"\nvs baseline vocab={base[0]} boost={base[2]}:")
    for vocab, actual, boost, res, _ in results[1:]:
        d = {k: (res[k]["chars_per_tok"] / base[3][k]["chars_per_tok"] - 1) * 100
             for k in langs}
        extra = (vocab - base[0]) * 1024 * 2 / 1e6
        print(f"  vocab={vocab} boost={boost}: " +
              ", ".join(f"{k} {d[k]:+.1f}%" for k in langs) +
              f"  (+{extra:.0f}M params)")


if __name__ == "__main__":
    main()
