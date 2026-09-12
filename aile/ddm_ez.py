import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class DDMResult:
    drift_rate: float        # v: evidence accumulation rate / cognitive fluency
    boundary_separation: float # a: decision caution / threshold
    non_decision_time: float   # Ter: sensory & motor execution latency (seconds)
    mean_decision_time: float  # MDT: cognitive deliberation duration
    sample_size: int
    mean_accuracy: float
    mean_rt: float
    rt_variance: float
    is_valid: bool
    status_message: str


def compute_ez_diffusion(
    accuracies: List[int],
    reaction_times: List[float],
    scaling_constant: float = 0.1,
    min_trials: int = 15,
) -> DDMResult:
    """
    Closed-form EZ-diffusion model estimator (Wagenmakers, van der Maas & Grasman, 2007).
    Derives drift rate (v), boundary separation (a), and non-decision time (Ter)
    from sample accuracy, mean RT, and RT variance without MCMC sampling.
    """
    n = len(accuracies)
    if n < min_trials or len(reaction_times) < min_trials:
        return DDMResult(
            drift_rate=0.0,
            boundary_separation=0.0,
            non_decision_time=0.0,
            mean_decision_time=0.0,
            sample_size=n,
            mean_accuracy=float(np.mean(accuracies)) if n > 0 else 0.0,
            mean_rt=float(np.mean(reaction_times)) if len(reaction_times) > 0 else 0.0,
            rt_variance=float(np.var(reaction_times, ddof=1)) if len(reaction_times) > 1 else 0.0,
            is_valid=False,
            status_message=f"Insufficient trials ({n}/{min_trials}). Cold-start fallback active.",
        )

    # Filter invalid reaction times (must be positive)
    valid_pairs = [(acc, rt) for acc, rt in zip(accuracies, reaction_times) if rt > 0.1]
    if len(valid_pairs) < min_trials:
        return DDMResult(
            drift_rate=0.0,
            boundary_separation=0.0,
            non_decision_time=0.0,
            mean_decision_time=0.0,
            sample_size=len(valid_pairs),
            mean_accuracy=0.0,
            mean_rt=0.0,
            rt_variance=0.0,
            is_valid=False,
            status_message="Insufficient valid positive reaction times.",
        )

    accs = [p[0] for p in valid_pairs]
    rts = [p[1] for p in valid_pairs]

    pc_raw = float(np.mean(accs))
    mrt = float(np.mean(rts))
    vrt = float(np.var(rts, ddof=1))

    # Guard against zero or near-zero variance
    if vrt <= 1e-6:
        vrt = 1e-4

    # Clip Pc away from 0, 0.5, and 1 to prevent degenerate logit / division by zero
    pc = max(0.02, min(0.98, pc_raw))
    if abs(pc - 0.5) < 0.01:
        pc = 0.51 if pc >= 0.5 else 0.49

    s = scaling_constant
    s2 = s * s

    # Logit transform: L = log(Pc / (1 - Pc))
    logit_pc = math.log(pc / (1.0 - pc))

    # Closed-form intermediate variable x
    inner = logit_pc * (logit_pc * (pc**2) - logit_pc * pc + pc - 0.5)
    x = inner / vrt

    # x must be positive for the fourth root
    if x <= 0:
        x = 1e-6

    # Drift rate v
    sign_val = 1.0 if pc > 0.5 else -1.0
    v = sign_val * s * (x ** 0.25)

    # Boundary separation a
    a = (s2 * logit_pc) / v if abs(v) > 1e-7 else (s2 * logit_pc) / 1e-7
    a = max(0.01, abs(a))

    # Mean decision time MDT
    y = -v * a / s2
    # Bound y to avoid overflow in exp(y)
    y_clipped = max(-50.0, min(50.0, y))
    exp_y = math.exp(y_clipped)

    if abs(v) > 1e-7:
        mdt = (a / (2.0 * v)) * (1.0 - exp_y) / (1.0 + exp_y)
    else:
        mdt = (a * a) / (2.0 * s2)

    # Non-decision time Ter = MRT - MDT
    ter = mrt - mdt

    # Physiological plausibility check: Ter should typically be >= 100ms
    if ter < 0.05:
        ter = max(0.05, mrt * 0.2)
        mdt = mrt - ter

    return DDMResult(
        drift_rate=float(v),
        boundary_separation=float(a),
        non_decision_time=float(ter),
        mean_decision_time=float(mdt),
        sample_size=len(valid_pairs),
        mean_accuracy=pc_raw,
        mean_rt=mrt,
        rt_variance=vrt,
        is_valid=True,
        status_message="EZ-Diffusion parameters successfully estimated.",
    )
