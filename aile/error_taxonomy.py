from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorDiagnosis:
    error_type: str
    title: str
    description: str
    recommended_action: str


def classify_interaction_error(
    is_correct: bool,
    skill_id: str,
    prior_mastery: float,
    retrievability: float,
    response_time: float,
    expected_intensity_sec: float = 12.0,
    confidence_rating: Optional[float] = None,
    hint_requested: bool = False,
    p_fatigued: float = 0.10,
    distractor_rationale: Optional[str] = None,
    root_cause_prereq: Optional[str] = None,
) -> Optional[ErrorDiagnosis]:
    """
    Classifies errors into one of six psychometrically defensible categories:
    1. Knowledge Deficit: Concept not yet acquired; foundational gap.
    2. Retrieval Failure: Previously mastered, but memory decayed over time.
    3. Procedural Error: Concept understood, but algebraic/calculation step misapplied.
    4. Careless / Slip Error: High competence, transient error under normal deliberation.
    5. Impulsive Error: Sub-threshold decision time; answered before adequate evidence accumulation.
    6. Cognitive Overload: Task complexity exceeded capacity; elevated latency & confusion.
    """
    if is_correct:
        return None

    # Ratio of observed response time to item difficulty intensity
    rt_ratio = response_time / max(1.0, expected_intensity_sec)

    # 1. Impulsive Error: fast response + incorrect (low decision boundary)
    if rt_ratio < 0.50:
        return ErrorDiagnosis(
            error_type="impulsive_error",
            title="Impulsive Response",
            description="Submitted answer significantly faster than problem complexity warrants, prior to sufficient evidence accumulation.",
            recommended_action="Introduce cognitive friction: prompt verification of intermediate steps before confirming selection.",
        )

    # 2. Cognitive Overload: high fatigue, prolonged latency, or explicit hint dependency
    if p_fatigued > 0.65 or (rt_ratio > 1.8 and hint_requested):
        return ErrorDiagnosis(
            error_type="cognitive_overload",
            title="Cognitive Overload & Fatigue",
            description="High latency and fatigue indicate the task's interacting elements exceed current working memory capacity.",
            recommended_action="Decompose problem into smaller sub-steps or schedule a 3-minute cognitive micro-break.",
        )

    # 3. Retrieval Failure: high prior mastery, but retrievability decayed
    if prior_mastery >= 0.70 and retrievability < 0.65:
        return ErrorDiagnosis(
            error_type="retrieval_failure",
            title="Memory Retrieval Lapse",
            description=f"Concept '{skill_id}' was previously mastered, but memory stability has decayed since last practice.",
            recommended_action="Schedule low-stakes spaced retrieval practice to rebuild memory stability.",
        )

    # 4. Careless / Slip Error: high prior mastery, high retrievability, deliberate timing
    if prior_mastery >= 0.75 and retrievability >= 0.75:
        return ErrorDiagnosis(
            error_type="careless_slip",
            title="Careless Slip",
            description="High competence with healthy retrievability; error is an isolated transient slip rather than a conceptual deficiency.",
            recommended_action="Acknowledge understanding, briefly highlight the slip, and continue curriculum progression.",
        )

    # 5. Knowledge Deficit: low mastery, especially with an unmet upstream prerequisite
    if prior_mastery < 0.45 or (root_cause_prereq and root_cause_prereq != skill_id):
        gap_target = root_cause_prereq or skill_id
        return ErrorDiagnosis(
            error_type="knowledge_deficit",
            title="Foundational Knowledge Deficit",
            description=f"Evidence indicates a genuine conceptual gap in prerequisite foundation: '{gap_target}'.",
            recommended_action=f"Prerequisite Repair: temporarily route learner to reinforce '{gap_target}' before re-attempting.",
        )

    # 6. Procedural Error: default when concept is emerging (0.45 <= mastery < 0.75)
    rationale_note = f" Pitfall observed: {distractor_rationale}" if distractor_rationale else ""
    return ErrorDiagnosis(
        error_type="procedural_error",
        title="Procedural Misconception",
        description=f"Concept framework is developing, but mathematical execution rules were misapplied.{rationale_note}",
        recommended_action="Provide faded worked examples illustrating the correct procedural sequence.",
    )
