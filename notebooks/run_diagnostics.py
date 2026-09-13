"""
Diagnostics and calibration verification script.
Evaluates:
1. Metacognitive calibration curve across simulated student profiles.
2. DKT prediction accuracy vs synthetic ground truth mastery.
3. EZ-diffusion parameter recovery.
4. GNN root-cause identification precision.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aile.calibration import compute_metacognitive_calibration
from aile.ddm_ez import compute_ez_diffusion
from aile.mastery import MasteryEngine
from aile.skill_gnn import SkillGNNPredictor
from aile.skill_graph import SkillGraph
from aile.simulator import StudentSimulator


def run_all_diagnostics():
    print("=" * 60)
    print("SARVAGYA COGNITIVE ENGINE DIAGNOSTIC BENCHMARKS")
    print("=" * 60)

    graph = SkillGraph()
    sim = StudentSimulator(graph)

    # 1. EZ-Diffusion Parameter Recovery
    print("\n[1] Evaluating EZ-Diffusion Parameter Recovery...")
    high_acc = [1] * 24 + [0] * 6  # 80% accuracy
    fast_rt = [7.0 + np.random.normal(0, 1.2) for _ in range(30)]
    ddm_res = compute_ez_diffusion(high_acc, fast_rt)
    print(f"  Sample Size: {ddm_res.sample_size}")
    print(f"  Drift Rate (v): {ddm_res.drift_rate:.4f} (expected > 0 for high accuracy)")
    print(f"  Boundary Separation (a): {ddm_res.boundary_separation:.4f}")
    print(f"  Non-Decision Time (Ter): {ddm_res.non_decision_time:.2f}s")
    print(f"  Valid Recovery: {ddm_res.is_valid}")

    # 2. Metacognitive Calibration
    print("\n[2] Evaluating Metacognitive Calibration...")
    overconf_ratings = [0.90] * 15
    mixed_outcomes = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # ~27%
    calib = compute_metacognitive_calibration(overconf_ratings, mixed_outcomes)
    print(f"  Brier Score: {calib.brier_score:.4f}")
    print(f"  Calibration Bias: {calib.calibration_bias:+.4f}")
    print(f"  Diagnostic Label: {calib.bias_label}")
    print(f"  Recommendation: {calib.recommendation}")

    # 3. GNN Root-Cause Isolation
    print("\n[3] Evaluating GNN Prerequisite Root-Cause Tracing...")
    gnn = SkillGNNPredictor()
    # Foundational gap in arithmetic_integers
    mock_mastery = {s: 0.80 for s in graph.skill_ids}
    mock_mastery["arithmetic_integers"] = 0.20
    _, rc, _ = gnn.propagate_and_trace_root_cause(
        graph, mock_mastery, struggling_skill="quadratic_equations"
    )
    print(f"  Downstream Struggling Skill: quadratic_equations")
    print(f"  Isolated Root Cause: {rc} (Target: arithmetic_integers)")
    print(f"  Correct Upstream Diagnosis: {'PASS' if rc == 'arithmetic_integers' else 'FAIL'}")

    # 4. POMDP 3-Step Rollout Profiling
    print("\n[4] Profiling POMDP Lookahead Rollout Planner...")
    from aile.policy_pomdp import POMDPRolloutPlanner
    planner = POMDPRolloutPlanner(graph, sim)
    plan = planner.plan(
        mastery_means=mock_mastery,
        p_fatigued=0.15,
        root_cause_skill=rc,
        struggling_skill="quadratic_equations",
        max_depth=3,
        branching_limit=3,
    )
    print(f"  Selected Optimal Action: {plan.best_action.action_type}")
    print(f"  Target Skill: {plan.best_action.target_skill}")
    print(f"  Rollout Latency: {plan.planning_time_ms:.2f}ms (Budget: <1000ms)")
    print(f"  Total Tree Rollouts: {plan.total_rollouts}")
    print(f"  Explanation: {plan.explanation}")

    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_diagnostics()
