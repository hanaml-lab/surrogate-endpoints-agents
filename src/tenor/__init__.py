"""TENOR: valid evaluation of long-running agents from early surrogates and late outcomes.

Reference implementation of the report "TENOR: Valid Evaluation of Long-Running Agents from Early Surrogates and
Late Outcomes" (Agarwal, hanaML, 2026).
"""
from .censoring import MaturationModel, rectifier_terms
from .confseq import asymptotic_cs, betting_cs, betting_cs_range, bounds_at
from .gate import BLOCK, SHIP, WAIT, GateTrace, ReleaseGate, Streams, combine, decide, outcome_only, proxy_only
from .index import CONTROLLED, INDEPENDENT, SCREEN_INDEPENDENT, SIGNALS, SurrogateIndex, outcome_r2
from .validate import PastChange, Validation, huber_slope, sign_agreement, trial_r2, validate

__version__ = "0.1.0"
__all__ = [
    "MaturationModel", "rectifier_terms", "asymptotic_cs", "betting_cs", "betting_cs_range", "bounds_at", "BLOCK", "SHIP",
    "WAIT", "GateTrace", "ReleaseGate", "Streams", "combine", "decide", "outcome_only", "proxy_only", "CONTROLLED",
    "INDEPENDENT", "SCREEN_INDEPENDENT", "SIGNALS", "SurrogateIndex", "outcome_r2", "PastChange", "Validation",
    "huber_slope", "sign_agreement", "trial_r2", "validate",
]
