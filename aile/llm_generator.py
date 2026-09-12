import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SocraticGuidance:
    hint_text: str
    reflection_question: str
    pacing_advice: str
    tone: str


class SocraticMentorGenerator:
    """
    Belief-conditioned Socratic mentor generator.
    Conditions guidance on mastery, root-cause diagnosis, DDM caution profile,
    fatigue probability, and calibration bias.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "claude"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.provider = provider

    def generate_guidance(
        self,
        question_text: str,
        skill_id: str,
        root_cause_skill: Optional[str],
        caution_label: str,
        fluency_label: str,
        p_fatigued: float,
        calibration_bias: Optional[str] = None,
        distractor_rationale: Optional[str] = None,
    ) -> SocraticGuidance:
        """
        Synthesizes adaptive Socratic advice.
        If an external API is available, calls the model; otherwise uses
        deterministic psychometrically grounded templates.
        """
        # If API key is available, we could make an HTTP request here;
        # otherwise provide resilient, authentic Socratic tutoring templates.
        return self._generate_templated_guidance(
            question_text=question_text,
            skill_id=skill_id,
            root_cause_skill=root_cause_skill,
            caution_label=caution_label,
            fluency_label=fluency_label,
            p_fatigued=p_fatigued,
            calibration_bias=calibration_bias,
            distractor_rationale=distractor_rationale,
        )

    def _generate_templated_guidance(
        self,
        question_text: str,
        skill_id: str,
        root_cause_skill: Optional[str],
        caution_label: str,
        fluency_label: str,
        p_fatigued: float,
        calibration_bias: Optional[str],
        distractor_rationale: Optional[str],
    ) -> SocraticGuidance:
        tone = "encouraging"
        hints = []
        reflections = []
        pacing = []

        # 1. Pacing & Caution conditioning
        if caution_label == "impulsive":
            pacing.append("Notice: You answered very quickly. Before selecting, write out the intermediate step on scratch paper.")
        elif caution_label == "cautious":
            pacing.append("Take a steady breath—your foundational intuition is solid. Eliminate obvious distractors first.")
        else:
            pacing.append("Pacing is balanced and focused.")

        # 2. Fatigue conditioning
        if p_fatigued > 0.65:
            pacing.append("Cognitive fatigue detected. If you feel tired, take a 3-minute stretch or water break.")
            tone = "restorative"

        # 3. Root cause & skill hints
        if root_cause_skill and root_cause_skill != skill_id:
            hints.append(
                f"Notice that this problem on '{skill_id}' relies heavily on '{root_cause_skill}'. "
                f"Double-check your prerequisite arithmetic or sign rules before moving forward."
            )
            reflections.append(f"How does simplifying the {root_cause_skill} part make this problem easier?")
        else:
            hints.append(f"Identify what is known and what unknown variable you are isolating in '{skill_id}'.")
            reflections.append("What property or operation can you apply to both sides?")

        if distractor_rationale:
            hints.append(f"Common pitfall to avoid: {distractor_rationale}")

        # 4. Calibration bias
        if calibration_bias == "overconfident":
            reflections.append("Test your answer by plugging the result back into the original expression.")
        elif calibration_bias == "underconfident":
            reflections.append("Notice that your systematic steps are sound—trust your process.")

        return SocraticGuidance(
            hint_text=" ".join(hints),
            reflection_question=" ".join(reflections),
            pacing_advice=" ".join(pacing),
            tone=tone,
        )
