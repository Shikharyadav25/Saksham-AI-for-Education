from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class MetacognitiveProfile:
    brier_score: float
    calibration_bias: float  # > 0: overconfident, < 0: underconfident
    bias_label: str          # "well_calibrated", "overconfident", "underconfident"
    sample_size: int
    recommendation: str


def compute_metacognitive_calibration(
    confidences: List[float],
    accuracies: List[int],
    min_samples: int = 5,
) -> MetacognitiveProfile:
    """
    Computes Brier score and directional calibration bias.
    Confidence ratings should be normalized in [0.0, 1.0].
    """
    n = len(confidences)
    if n < min_samples or len(accuracies) < min_samples:
        return MetacognitiveProfile(
            brier_score=0.25,
            calibration_bias=0.0,
            bias_label="insufficient_data",
            sample_size=n,
            recommendation="Rate your confidence on more questions to track self-calibration.",
        )

    pairs = [(float(c), int(a)) for c, a in zip(confidences, accuracies) if 0.0 <= c <= 1.0]
    if len(pairs) < min_samples:
        return MetacognitiveProfile(
            brier_score=0.25,
            calibration_bias=0.0,
            bias_label="insufficient_data",
            sample_size=len(pairs),
            recommendation="Valid confidence scores needed.",
        )

    confs = np.array([p[0] for p in pairs])
    accs = np.array([p[1] for p in pairs])

    # Brier score: mean squared error between subjective probability and binary outcome
    brier = float(np.mean((confs - accs) ** 2))

    # Directional bias: mean(confidence) - mean(accuracy)
    mean_conf = float(np.mean(confs))
    mean_acc = float(np.mean(accs))
    bias = mean_conf - mean_acc

    if bias > 0.15:
        bias_label = "overconfident"
        rec = "You tend to rate confidence higher than actual accuracy. Review edge cases and double-check your steps."
    elif bias < -0.15:
        bias_label = "underconfident"
        rec = "You perform better than you think! Trust your initial reasoning more."
    else:
        bias_label = "well_calibrated"
        rec = "Your self-assessment closely mirrors your actual knowledge."

    return MetacognitiveProfile(
        brier_score=brier,
        calibration_bias=bias,
        bias_label=bias_label,
        sample_size=len(pairs),
        recommendation=rec,
    )
