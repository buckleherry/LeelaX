from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class EloResult:
    elo_diff: float
    score: float   # (W + 0.5*D)/N
    n: int
    wins: int
    draws: int
    losses: int

def elo_from_wdl(w: int, d: int, l: int, eps: float = 1e-6) -> EloResult:
    n = max(w + d + l, 1)
    score = (w + 0.5 * d) / n
    # clamp to avoid inf
    score = min(max(score, eps), 1 - eps)
    # logistic elo diff (A vs B): score = 1/(1+10^(-Δ/400))  => Δ = 400 * log10(score/(1-score))
    elo = 400.0 * math.log10(score / (1.0 - score))
    return EloResult(elo, score, n, w, d, l)

