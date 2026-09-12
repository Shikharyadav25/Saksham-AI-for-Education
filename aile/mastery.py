from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .mastery_dkt import DKTMasteryPredictor
from .skill_graph import SkillGraph


@dataclass
class BetaDistribution:
    alpha: float = 2.0
    beta_param: float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta_param
        return (self.alpha * self.beta_param) / (total * total * (total + 1.0))


@dataclass
class MasteryEstimate:
    means: Dict[str, float]
    variances: Dict[str, float]
    active_estimator: str  # "bayesian_cold_start" vs "dkt_lstm"
    total_interactions: int


class MasteryEngine:
    """
    Unified mastery estimation engine.
    Applies Bayesian Beta-Binomial updating during cold start (< 5 interactions)
    and hands off to Deep Knowledge Tracing (DKT) once sufficient sequence history exists.
    """

    def __init__(
        self,
        skill_graph: SkillGraph,
        dkt_weights_path: Optional[Path] = None,
        cold_start_threshold: int = 5,
    ):
        self.skill_graph = skill_graph
        self.cold_start_threshold = cold_start_threshold
        num_skills = skill_graph.num_skills

        # DKT predictor
        self.dkt = DKTMasteryPredictor(
            num_skills=num_skills,
            weights_path=dkt_weights_path or (Path(__file__).resolve().parent.parent / "models" / "dkt.pt"),
        )

        # Per-skill Bayesian priors
        self.bayesian_beliefs: Dict[str, BetaDistribution] = {
            s: BetaDistribution() for s in skill_graph.skill_ids
        }
        self.interaction_history: List[Tuple[int, bool]] = []

    def register_interaction(self, skill_id: str, is_correct: bool) -> None:
        """Updates internal state with a new question response."""
        if skill_id in self.bayesian_beliefs:
            dist = self.bayesian_beliefs[skill_id]
            if is_correct:
                dist.alpha += 1.0
            else:
                dist.beta_param += 1.0

        if skill_id in self.skill_graph.skill_to_index:
            s_idx = self.skill_graph.skill_to_index[skill_id]
            self.interaction_history.append((s_idx, is_correct))

    def get_mastery_estimate(self) -> MasteryEstimate:
        n = len(self.interaction_history)

        # Fallback to Bayesian if below cold-start threshold or DKT not loaded
        if n < self.cold_start_threshold or not self.dkt.is_loaded:
            means = {s: self.bayesian_beliefs[s].mean for s in self.skill_graph.skill_ids}
            variances = {s: self.bayesian_beliefs[s].variance for s in self.skill_graph.skill_ids}
            return MasteryEstimate(
                means=means,
                variances=variances,
                active_estimator="bayesian_cold_start",
                total_interactions=n,
            )

        # DKT LSTM estimation
        dkt_raw = self.dkt.predict_mastery_from_history(self.interaction_history)
        means: Dict[str, float] = {}
        variances: Dict[str, float] = {}

        for sid in self.skill_graph.skill_ids:
            idx = self.skill_graph.skill_to_index[sid]
            means[sid] = dkt_raw.get(idx, 0.5)
            # Variance reflects amount of observations for this specific skill
            variances[sid] = self.bayesian_beliefs[sid].variance

        return MasteryEstimate(
            means=means,
            variances=variances,
            active_estimator="dkt_lstm",
            total_interactions=n,
        )
