"""The pretraining data mixture.

Weights are token shares of the final corpus, not byte shares of the download.

EVERY entry below was verified against the HuggingFace datasets-server API:
dataset exists, config name is real, split name is real, and the text field name
is correct. Do not edit these by guesswork -- HF config/split naming is
inconsistent across datasets and a wrong string fails only once the streaming
run is already hours in. Re-verify with:

    python -m src.data.verify_sources

The mix balances three goals competing for a fixed 100B token budget: fluent
English, working Hindi, and enough math/code for reasoning to be trainable
later. You cannot max out all three at 1B scale.
"""

from dataclasses import dataclass


@dataclass
class Source:
    name: str              # HuggingFace dataset id
    config: str | None     # HF "config"/"subset" name
    split: str             # HF split name -- for several Indic sets the
                           # LANGUAGE lives here, not in the config
    text_field: str        # column holding the raw text
    weight: float          # share of the final token budget
    epochs: float = 1.0    # >1 = deliberately repeated (small, dense sources only)
    note: str = ""
    # Some corpora ship one SENTENCE per row rather than one document. Emitting
    # those as individual documents would append an EOS every ~40 characters,
    # teaching the model that text ends constantly. Rows from these sources are
    # concatenated into document-sized chunks before EOS is added.
    sentence_level: bool = False
    min_chars: int = 100   # rows shorter than this are dropped as noise


MIXTURE = [
    # ---- English web: the backbone --------------------------------------
    Source("HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text", 0.57,
           note="Common Crawl, educational-quality filtered"),

    # ---- Indic -----------------------------------------------------------
    # Note the shape: config is 'verified', the LANGUAGE is the split.
    # Was 0.09, with the remaining 0.03 coming from IndicCorpV2. That source is
    # a plain-text dump containing invalid UTF-8; the reader aborts on the very
    # first bad byte and it produced ZERO tokens in a full run. Its share moved
    # here instead. Sangraha is human-verified and higher quality anyway, so
    # this is a better mixture than the original, not just a repair.
    Source("ai4bharat/sangraha", "verified", "hin", "text", 0.12,
           note="AI4Bharat, human-verified Hindi"),

    # ---- code ------------------------------------------------------------
    # Getting open, multi-language code is harder than it looks. Everything
    # good is either gated or script-based:
    #   bigcode/the-stack-dedup, the-stack-v2, starcoderdata   gated
    #   codeparrot/github-code-clean, bigcode/commitpackft     script-based
    # So these two are what actually stream today. Note the different fields.
    #
    # UPGRADE PATH: run `huggingface-cli login` (starcoderdata is gated=auto,
    # so approval is instant), then replace both entries with:
    #   Source("bigcode/starcoderdata", "python",     "train", "content", 0.06)
    #   Source("bigcode/starcoderdata", "javascript", "train", "content", 0.03)
    #   Source("bigcode/starcoderdata", "c",          "train", "content", 0.03)
    # That is genuinely better code data -- multi-language and well filtered.
    Source("codeparrot/codeparrot-clean", None, "train", "content", 0.08,
           note="Python, deduplicated GitHub"),
    Source("vikp/starcoder_filtered", None, "train", "code", 0.04,
           note="Jupyter notebooks -- narrower than general code, but open"),

    # ---- math: the foundation for any later reasoning work ---------------
    Source("open-web-math/open-web-math", "default", "train", "text", 0.06),
    Source("HuggingFaceTB/finemath", "finemath-4plus", "train", "text", 0.03),

    # ---- long-form: where coherence across paragraphs comes from ---------
    # RedPajama-Data-1T is script-based and no longer loadable under
    # datasets>=3.0, so arXiv and books come from these instead.
    Source("common-pile/arxiv_papers", "default", "train", "text", 0.03),
    Source("manu/project_gutenberg", "default", "en", "text", 0.03,
           note="public-domain books"),

    # ---- reference: small, dense, worth repeating -------------------------
    Source("wikimedia/wikipedia", "20231101.en", "train", "text", 0.03, epochs=2.0),
    Source("wikimedia/wikipedia", "20231101.hi", "train", "text", 0.01, epochs=3.0),
]

# Deliberately excluded:
#   bigcode/the-stack-dedup            gated; github-code-clean covers it
#   togethercomputer/RedPajama-Data-1T script-based, broken on datasets>=3.0
#   EleutherAI/proof-pile-2            script-based, same problem
#   HuggingFaceH4/stack-exchange-*     stores question/answers columns, not
#                                      flat text; would need a custom formatter.
#                                      Q&A shape is picked up at SFT anyway.

# Sized to the 7-day budget: 5 days of pretraining on 5 GPUs buys ~22B tokens.
# A little headroom on top so the loader never runs dry near the end of the run.
TOTAL_TOKENS = 25_000_000_000


def validate():
    total = sum(s.weight for s in MIXTURE)
    assert abs(total - 1.0) < 1e-6, f"weights sum to {total}, not 1.0"
    return True


def summary():
    validate()
    print(f"{'source':38s} {'config':16s} {'split':10s} {'field':7s} {'share':>6s} {'tok':>7s}")
    print("-" * 92)
    for s in MIXTURE:
        print(f"{s.name:38s} {str(s.config):16s} {s.split:10s} {s.text_field:7s} "
              f"{s.weight*100:5.1f}% {s.weight*TOTAL_TOKENS/1e9:6.1f}B")
    print("-" * 92)
    print(f"{'TOTAL':38s} {'':16s} {'':10s} {'':7s} {100.0:5.1f}% "
          f"{TOTAL_TOKENS/1e9:6.1f}B")
    print(f"\ntokenized on disk (uint16): {TOTAL_TOKENS*2/1e9:.0f} GB")

    by_kind = {
        "English web": 0.57, "Hindi": 0.12, "code": 0.12,
        "math": 0.09, "long-form": 0.06, "wikipedia": 0.04,
    }
    print("\nby category:")
    for k, v in by_kind.items():
        print(f"  {k:14s} {v*100:5.1f}%")


if __name__ == "__main__":
    summary()
