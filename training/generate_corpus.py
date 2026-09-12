import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aile.simulator import StudentSimulator
from aile.skill_graph import SkillGraph

OUTPUT_PATH = Path(__file__).resolve().parent / "synthetic_corpus.pkl"


def generate_synthetic_corpus(
    num_students: int = 1500,
    min_interactions: int = 25,
    max_interactions: int = 40,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    """
    Synthesizes a large corpus of student learning trajectories across the prerequisite DAG.
    Saves sequences with ground truth mastery, reaction times, hints, and fatigue states.
    """
    skill_graph = SkillGraph()
    simulator = StudentSimulator(skill_graph)
    skills = skill_graph.skill_ids
    topo_order = skill_graph.get_topological_order()

    trajectories = []
    print(f"Generating synthetic interaction trajectories for {num_students} students...")

    for i in range(num_students):
        # In a third of students, inject a specific prerequisite gap
        gap_skill = None
        if i % 3 == 0:
            # Pick a foundational skill as an injected gap
            gap_skill = random.choice(topo_order[:3])

        student = simulator.create_random_student(
            student_id=f"synth_student_{i:04d}",
            prerequisite_gap_skill=gap_skill,
        )

        num_steps = random.randint(min_interactions, max_interactions)
        student_history = []

        # Curriculum progression with occasional backtracking
        current_skill_idx = random.randint(0, 2)

        for step_idx in range(num_steps):
            # Target skill selection: mostly progress along curriculum, occasionally backtrack
            if random.random() < 0.20:
                skill_id = random.choice(topo_order[: max(1, current_skill_idx + 1)])
            else:
                skill_id = topo_order[min(current_skill_idx, len(topo_order) - 1)]

            s_idx = skill_graph.skill_to_index[skill_id]
            is_correct, rt, hint = simulator.step(student, skill_id, action_type="practice")

            student_history.append({
                "step": step_idx,
                "skill_id": skill_id,
                "skill_idx": s_idx,
                "is_correct": 1 if is_correct else 0,
                "reaction_time": rt,
                "hint_requested": 1 if hint else 0,
                "is_fatigued": 1 if student.is_fatigued else 0,
                "true_mastery": dict(student.true_mastery),
                "injected_gap_skill": gap_skill,
            })

            # If student is mastering the skill, advance
            if student.true_mastery[skill_id] > 0.75 and current_skill_idx < len(topo_order) - 1:
                current_skill_idx += 1

        trajectories.append(student_history)

    corpus_data = {
        "num_students": num_students,
        "skills": skills,
        "trajectories": trajectories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(corpus_data, f)

    print(f"Corpus generated successfully with {num_students} trajectories -> {output_path}")
    return corpus_data


if __name__ == "__main__":
    generate_synthetic_corpus()
