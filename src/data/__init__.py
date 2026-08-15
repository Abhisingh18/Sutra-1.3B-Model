"""Data package.

`TokenDataset` is exposed lazily because it pulls in torch, while the mixture
and verification tools deliberately need nothing beyond the standard library.
That lets you validate data sources on a machine (or in an env) that has no
torch installed yet.
"""

from .mixture import MIXTURE, TOTAL_TOKENS, Source, validate

__all__ = ["MIXTURE", "TOTAL_TOKENS", "Source", "validate", "TokenDataset"]


def __getattr__(name):
    if name == "TokenDataset":
        from .dataloader import TokenDataset
        return TokenDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
