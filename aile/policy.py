from dataclasses import dataclass
from typing import Dict, List, Optional
from .skill_graph import SkillGraph


@dataclass
class ActionCandidate:
    action_type: str        # "practice", "remediate_root_cause", "advance_new_skill", "scaffold_hint", "spaced_review", "rest_break"
    target_skill: Optional[str]
    description: str
    immediate_utility: float = 0.0


def evaluate_belief_utility(
    mastery_means: Dict[str, float],
    p_fatigued: float,
    root_cause_skill: Optional[str],
    decayed_skills: Optional[List[str]] = None,
    action_type: str = "practice",
    target_skill: Optional[str] = None,
) -> float:
    """
    Evaluates pedagogical utility for a given belief state and intervention.
    Balances learning yield, cognitive load, fatigue mitigation, and prerequisite health.
    """
    decayed = decayed_skills or []
    utility = 0.0

    # Rest break evaluation
    if action_type == "rest_break":
        if p_fatigued > 0.65:
            return 8.0 * p_fatigued  # High utility when genuinely fatigued
        return -2.0  # Unnecessary downtime if engaged

    # Penalty for pushing hard tasks when fatigued
    if p_fatigued > 0.70 and action_type in ["practice", "advance_new_skill"]:
        utility -= 6.0 * p_fatigued

    if target_skill is None:
        return utility

    current_m = mastery_means.get(target_skill, 0.5)

    # Remediating root cause
    if action_type == "remediate_root_cause":
        if target_skill == root_cause_skill:
            # High reward for fixing the upstream foundation
            utility += 5.0 * (1.0 - current_m)
        else:
            utility += 1.0

    # Scaffolding when struggling
    elif action_type == "scaffold_hint":
        if current_m < 0.50:
            utility += 4.0 * (1.0 - current_m)
        else:
            utility -= 1.0  # Over-scaffolding mastered skills is unhelpful

    # Spaced review for decaying memory
    elif action_type == "spaced_review":
        if target_skill in decayed:
            utility += 4.5
        else:
            utility += 0.5

    # Advancing new skill (ZPD: Zone of Proximal Development)
    elif action_type == "advance_new_skill":
        if 0.35 <= current_m <= 0.70:
            utility += 3.5
        elif current_m > 0.85:
            utility -= 2.0  # Already mastered
        else:
            utility += 1.0

    # Standard practice
    elif action_type == "practice":
        if 0.40 <= current_m <= 0.75:
            utility += 3.0  # In optimal learning zone
        elif current_m > 0.85:
            utility -= 1.5  # Redundant practice
        else:
            utility += 1.5

    # Global curriculum mastery bonus
    avg_mastery = sum(mastery_means.values()) / max(1, len(mastery_means))
    utility += avg_mastery * 2.0

    return utility
