"""Train the BPE tokenizer.

Run this ONCE, before anything else. Every later stage depends on it, and
changing it means throwing away all tokenized data and all trained weights.

Two decisions worth understanding:

*Byte-level BPE* means no token is ever <unk>. Any byte sequence -- Devanagari,
emoji, corrupted UTF-8 -- can always be encoded. This matters a lot for Hindi.

*Digit splitting*: numbers are split into individual digits. Without this the
tokenizer learns "2023" as one token and "2024" as another, and the model has no
way to see that they are related. Every model that is any good at arithmetic
does this.

Usage:
    python -m src.tokenizer.train_tokenizer --output tokenizer/ --sample-gb 20
"""

import argparse
import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

from .special_tokens import ALL_SPECIALS, VOCAB_SIZE, N_LEARNED
from ..data.mixture import MIXTURE


# Hindi is deliberately weighted ABOVE its share of the training mix. Devanagari
# needs more merges than Latin script to encode efficiently, and an
# under-merged Hindi vocabulary silently doubles the token cost of every Hindi
# document for the entire pretraining run.
HINDI_BOOST = 2.0


def sample_weights():
    """Tokenizer sampling weights, derived from the real training mixture.

    The tokenizer sample must reflect the corpus it will be used on. Deriving
    these from MIXTURE rather than hardcoding them means the two can never drift
    apart -- which they did once already in this project.
    """
    weights = []
    for s in MIXTURE:
        w = s.weight
        if "sangraha" in s.name or "Indic" in s.name or s.config == "20231101.hi":
            w *= HINDI_BOOST
        weights.append(w)
    total = sum(weights)
    return [w / total for w in weights]


def sample_corpus(sample_gb: float = 20.0):
    """Stream a representative sample for tokenizer training.

    The tokenizer needs a representative sample, not the full corpus -- 20GB is
    plenty for learning ~44K merges.
    """
    from datasets import load_dataset

    weights = sample_weights()
    budget = sample_gb * 1e9

    for src, w in zip(MIXTURE, weights):
        limit = budget * w
        label = f"{src.name}:{src.config}:{src.split}"
        try:
            ds = load_dataset(src.name, src.config, split=src.split, streaming=True)
        except Exception as e:
            print(f"  ! skipping {label}: {str(e)[:90]}")
            continue

        print(f"  sampling {label} up to {limit/1e9:.2f}GB")
        used = 0
        for row in ds:
            text = row.get(src.text_field)
            if not text or len(text) < src.min_chars:
                continue
            used += len(text)
            if used > limit:
                break
            yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tokenizer")
    ap.add_argument("--sample-gb", type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))

    # NFC only. Do NOT lowercase or strip accents -- that destroys information
    # the model needs, and Devanagari in particular breaks under aggressive
    # normalisation.
    tok.normalizer = normalizers.NFC()

    tok.pre_tokenizer = pre_tokenizers.Sequence([
        # Split numbers into individual digits (see module docstring).
        pre_tokenizers.Digits(individual_digits=True),
        # add_prefix_space MUST be False here. With Digits() in front, every
        # digit becomes its own pre-token, and add_prefix_space=True then
        # prepends a space to each one -- so "3*x - 5" round-trips as
        # " 3 *x -  5". The model would learn to write "3 4" instead of "34"
        # and every number and code sample in the corpus would be corrupted.
        # ByteLevel already encodes spaces as a distinct byte (Ġ), so word
        # boundaries survive without the prefix hack.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()

    # BpeTrainer counts special tokens INSIDE vocab_size, so pass the full
    # target here. Passing N_LEARNED subtracts the specials twice and silently
    # gives you a 43,861-token vocabulary instead of 48,000.
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=ALL_SPECIALS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"training BPE: {N_LEARNED} learned + {len(ALL_SPECIALS)} special "
          f"= {VOCAB_SIZE} total")
    tok.train_from_iterator(sample_corpus(args.sample_gb), trainer=trainer)

    path = os.path.join(args.output, "tokenizer.json")
    tok.save(path)
    got = tok.get_vocab_size()
    print(f"saved -> {path}  (vocab {got})")

    # The model's embedding matrix is sized from VOCAB_SIZE. A mismatch here
    # means every id above `got` is dead weight, or worse, out of range.
    assert got == VOCAB_SIZE, f"vocab is {got}, expected {VOCAB_SIZE}"
    for t in ("<|end_of_text|>", "<|assistant|>", "<|think|>", "<|audio_0|>"):
        assert tok.token_to_id(t) is not None, f"special token {t} missing"

    # ---- sanity checks ----------------------------------------------------
    # Fertility = tokens per word. Below ~1.5 is good for English. Hindi will be
    # worse; if it is above ~2.5 the vocab is too small for two languages and
    # you should raise it before spending two months of GPU time.
    checks = [
        ("en", "The capital of India is New Delhi, a city of 32 million people."),
        ("hi", "भारत की राजधानी नई दिल्ली है, जो एक बड़ा शहर है।"),
        ("code", "def fibonacci(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)"),
        ("math", "Solve for x: 3x^2 + 12x - 15 = 0"),
    ]
    print("\nfertility check (tokens/word, lower is better):")
    for lang, text in checks:
        ids = tok.encode(text, add_special_tokens=False).ids
        print(f"  {lang:5s} {len(ids):4d} tokens / {len(text.split()):3d} words "
              f"= {len(ids)/len(text.split()):.2f}")

    # Exact round-trip. This is not a formality: a tokenizer that silently
    # mangles digits or whitespace corrupts every token of the corpus, and the
    # damage is only visible weeks later in the model's output.
    print("\nround-trip check:")
    failures = []
    roundtrip_cases = [t for _, t in checks] + [
        "x = 1234567",
        "The year 2024 had 365 days.",
        "  leading and trailing space  ",
        "tabs\tand\nnewlines",
        "emoji 🚀 and ünïcode",
        "मैं 25 साल का हूँ।",
    ]
    for text in roundtrip_cases:
        got = tok.decode(tok.encode(text, add_special_tokens=False).ids)
        if got != text:
            failures.append((text, got))
    if failures:
        for want, got in failures:
            print(f"  FAIL want {want!r}\n       got  {got!r}")
        raise SystemExit(f"{len(failures)} round-trip failure(s) -- do NOT use "
                         f"this tokenizer; fix the pre-tokenizer first.")
    print(f"  all {len(roundtrip_cases)} cases exact")


if __name__ == "__main__":
    main()
