"""TENOR: valid evaluation of long-running agents from early surrogates and late outcomes."""
from .confseq import asymptotic_cs, betting_cs, betting_cs_range, bounds_at

__version__ = "0.1.0"
__all__ = [
    "asymptotic_cs",
    "betting_cs",
    "betting_cs_range",
    "bounds_at",
]
