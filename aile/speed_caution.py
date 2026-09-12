from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .ddm_ez import DDMResult, compute_ez_diffusion


@dataclass
class CognitivePacingProfile:
    caution_label: str       # "impulsive", "cautious", "balanced"
    fluency_label: str       # "high_fluency", "developing", "struggling"
    mean_rt: float
    tau_speed: float         # Latent processing speed parameter tau (van der Linden model)
    z_score_rt: float        # Normalized against item intensity lambda_j
    ddm_result: Optional[DDMResult]
    recommendation: str
    active_estimator: str    # "hierarchical_rt_z_score" vs "ez_diffusion"


def analyze_pacing_and_caution(
    accuracies: List[int],
    reaction_times: List[float],
    item_intensities: Optional[List[float]] = None,
    default_intensity_sec: float = 12.0,
    population_std_rt: float = 5.0,
) -> CognitivePacingProfile:
    """
    Evaluates learner pacing, impulsivity, and cognitive fluency.
    Normalizes response time against item intensity lambda_j using hierarchical
    response-time principles (log Tij ~ N(lambda_j - tau_i, sigma^2)).
    Dispatches to closed-form EZ-diffusion model once sample size reaches >= 15 trials.
    """
    n = len(reaction_times)
    if n == 0:
        return CognitivePacingProfile(
            caution_label="unknown",
            fluency_label="unknown",
            mean_rt=0.0,
            tau_speed=0.0,
            z_score_rt=0.0,
            ddm_result=None,
            recommendation="Collect more interaction data.",
            active_estimator="none",
        )

    # Compute item-normalized log-residual response times
    intensities = item_intensities or [default_intensity_sec] * n
    if len(intensities) < n:
        intensities = intensities + [default_intensity_sec] * (n - len(intensities))

    log_residuals = []
    for rt, intensity in zip(reaction_times, intensities):
        valid_rt = max(0.5, rt)
        valid_int = max(1.0, intensity)
        # Residual: log(T_ij) - log(lambda_j)
        log_residuals.append(np.log(valid_rt) - np.log(valid_int))

    # Latent speed parameter tau: positive means faster than item expected duration
    tau_speed = float(-np.mean(log_residuals))
    mean_rt = float(np.mean(reaction_times))
    
    # Normalized z-score relative to item intensities
    mean_residual = float(np.mean(log_residuals))
    z_rt = mean_residual / 0.40  # 0.40 standard deviation on log scale is typical in psychometrics

    # Cold start check: fall back to hierarchical RT analysis if under 15 trials
    if n < 15:
        acc_rate = float(np.mean(accuracies)) if accuracies else 0.5
        if z_rt < -0.8 and acc_rate < 0.6:
            caution_label = "impulsive"
            recommendation = "Low caution detected. Fast responses relative to item complexity with low accuracy suggest guessing."
        elif z_rt > 1.0:
            caution_label = "cautious"
            recommendation = "Deliberate verification. High response time relative to item complexity indicates careful reasoning or hesitation."
        else:
            caution_label = "balanced"
            recommendation = "Balanced pacing relative to question complexity."

        fluency_label = "high_fluency" if acc_rate >= 0.8 else ("developing" if acc_rate >= 0.5 else "struggling")

        return CognitivePacingProfile(
            caution_label=caution_label,
            fluency_label=fluency_label,
            mean_rt=mean_rt,
            tau_speed=tau_speed,
            z_score_rt=float(z_rt),
            ddm_result=None,
            recommendation=recommendation,
            active_estimator="hierarchical_rt_z_score",
        )

    # Sufficient trials (>= 15): compute EZ-diffusion
    ddm = compute_ez_diffusion(accuracies, reaction_times)

    # DDM parameter interpretation (Wagenmakers et al., 2007)
    if ddm.boundary_separation < 0.10:
        caution_label = "impulsive"
        recommendation = "Low decision boundary (a) detected. Prompt student to verify intermediate steps before committing."
    elif ddm.boundary_separation > 0.18:
        caution_label = "cautious"
        recommendation = "High decision threshold (a) detected. The student accumulates extensive evidence before answering; build self-efficacy."
    else:
        caution_label = "balanced"
        recommendation = "Healthy speed-accuracy balance."

    if ddm.drift_rate > 0.22:
        fluency_label = "high_fluency"
    elif ddm.drift_rate > 0.08:
        fluency_label = "developing"
    else:
        fluency_label = "struggling"

    return CognitivePacingProfile(
        caution_label=caution_label,
        fluency_label=fluency_label,
        mean_rt=mean_rt,
        tau_speed=tau_speed,
        z_score_rt=float(z_rt),
        ddm_result=ddm,
        recommendation=recommendation,
        active_estimator="ez_diffusion",
    )
