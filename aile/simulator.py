import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .skill_graph import SkillGraph


@dataclass
class SimulatedStudentProfile:
    student_id: str
    true_mastery: Dict[str, float]
    learning_rate: float            # 0.05 to 0.25
    base_rt: float                  # Mean RT in seconds (8.0 to 18.0)
    rt_variance: float              # 2.0 to 8.0
    caution_a: float                # DDM boundary separation (0.08 to 0.22)
    fluency_v: float                # DDM drift rate (0.05 to 0.35)
    guess_rate: float = 0.20
    slip_rate: float = 0.10
    fatigue_threshold: int = 15     # Number of interactions before fatigue kicks in
    is_fatigued: bool = False
    interaction_count: int = 0


class StudentSimulator:
    """
    Generative student simulator and environment transition model.
    Models latent skill acquisition, reaction times, guessing, slipping,
    and fatigue onset. Used for offline corpus synthesis and POMDP rollouts.
    """

    def __init__(self, skill_graph: SkillGraph):
        self.skill_graph = skill_graph

    def create_random_student(
        self,
        student_id: str,
        prerequisite_gap_skill: Optional[str] = None,
    ) -> SimulatedStudentProfile:
        """Generates a heterogeneous synthetic student."""
        skills = self.skill_graph.skill_ids
        true_mastery: Dict[str, float] = {}

        # Base mastery distribution
        base_ability = random.uniform(0.3, 0.8)
        for s in self.skill_graph.get_topological_order():
            prereqs = self.skill_graph.get_prerequisites(s)
            if not prereqs:
                true_mastery[s] = min(0.95, max(0.1, random.gauss(base_ability, 0.15)))
            else:
                # Upstream prerequisite mastery bounds downstream mastery
                parent_masteries = [true_mastery[p] for p in prereqs]
                mean_parent = float(np.mean(parent_masteries))
                true_mastery[s] = min(0.95, max(0.05, mean_parent * random.uniform(0.6, 0.95)))

        if prerequisite_gap_skill and prerequisite_gap_skill in true_mastery:
            true_mastery[prerequisite_gap_skill] = 0.20  # Injected gap

        caution = random.uniform(0.09, 0.21)
        fluency = random.uniform(0.08, 0.32)
        base_rt = random.uniform(7.0, 16.0)

        return SimulatedStudentProfile(
            student_id=student_id,
            true_mastery=true_mastery,
            learning_rate=random.uniform(0.08, 0.20),
            base_rt=base_rt,
            rt_variance=random.uniform(3.0, 7.0),
            caution_a=caution,
            fluency_v=fluency,
            guess_rate=0.20,
            slip_rate=0.10,
            fatigue_threshold=random.randint(12, 22),
        )

    def step(
        self,
        student: SimulatedStudentProfile,
        skill_id: str,
        action_type: str = "practice",
    ) -> Tuple[bool, float, bool]:
        """
        Simulates one interaction step.
        Returns:
            is_correct: bool
            reaction_time: float (seconds)
            hint_requested: bool
        """
        student.interaction_count += 1
        if student.interaction_count >= student.fatigue_threshold:
            student.is_fatigued = True

        if action_type == "rest_break":
            student.is_fatigued = False
            student.interaction_count = max(0, student.interaction_count - 10)
            return True, 5.0, False

        mastery = student.true_mastery.get(skill_id, 0.5)

        # Scaffolding increases effective mastery for this step
        effective_mastery = mastery
        if action_type == "scaffold_hint":
            effective_mastery = min(0.95, mastery + 0.35)

        # Fatigue penalty on performance
        if student.is_fatigued:
            effective_mastery = max(0.05, effective_mastery - 0.25)

        # 3-Parameter Logistic / Guess-Slip calculation
        p_correct = (
            student.guess_rate + (1.0 - student.guess_rate - student.slip_rate) * effective_mastery
        )
        is_correct = random.random() < p_correct

        # Reaction time simulation using DDM drift / non-decision time
        noise = random.gauss(0.0, student.rt_variance)
        if student.is_fatigued:
            noise += random.uniform(2.0, 6.0)

        rt = max(1.5, student.base_rt + noise)

        hint_requested = False
        if action_type == "scaffold_hint":
            hint_requested = True
        elif not is_correct and (mastery < 0.4 or student.is_fatigued):
            hint_requested = random.random() < 0.60

        # Learning transition: mastery increments
        learning_gain = student.learning_rate
        if action_type == "scaffold_hint":
            learning_gain *= 1.4
        elif action_type == "remediate_root_cause":
            learning_gain *= 1.6

        if is_correct:
            student.true_mastery[skill_id] = min(0.98, mastery + learning_gain * (1.0 - mastery))
        else:
            student.true_mastery[skill_id] = min(0.98, mastery + (learning_gain * 0.4) * (1.0 - mastery))

        return is_correct, float(rt), hint_requested
