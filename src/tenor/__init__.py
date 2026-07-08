"""TENOR: valid evaluation of long-running agents from early surrogates and late outcomes."""
from .censoring import MaturationModel, rectifier_terms
from .confseq import asymptotic_cs, betting_cs, betting_cs_range, bounds_at
from .index import CONTROLLED, INDEPENDENT, SCREEN_INDEPENDENT, SIGNALS, SurrogateIndex, outcome_r2

__version__ = "0.1.0"
__all__ = [
    "MaturationModel",
    "rectifier_terms",
    "asymptotic_cs",
    "betting_cs",
    "betting_cs_range",
    "bounds_at",
    "CONTROLLED",
    "INDEPENDENT",
    "SCREEN_INDEPENDENT",
    "SIGNALS",
    "SurrogateIndex",
    "outcome_r2",
]
