import math
import pytest
import numpy as np
from aile.calibration import compute_metacognitive_calibration
from aile.ddm_ez import compute_ez_diffusion
from aile.memory import MemoryModel
from aile.policy_pomdp import POMDPRolloutPlanner
from aile.simulator import StudentSimulator
from aile.skill_gnn import SkillGNNPredictor
from aile.skill_graph import SkillGraph
from aile.speed_caution import analyze_pacing_and_caution


def test_ez_diffusion_numerical_validity():
    # Construct a sample with high accuracy and moderate RT
    accs = [1] * 16 + [0] * 4  # 80% accuracy (20 trials)
    rts = [10.0 + (i % 3) * 1.5 for i in range(20)]  # mean around 11.5s, positive variance

    res = compute_ez_diffusion(accs, rts, min_trials=15)
    assert res.is_valid is True
    assert res.drift_rate > 0.0, "High accuracy should yield positive drift rate"
    assert res.boundary_separation > 0.0, "Boundary separation must be positive"
    assert res.non_decision_time > 0.0, "Non-decision time must be positive"
    assert res.mean_decision_time < res.mean_rt, "MDT must be less than MRT"


def test_speed_caution_cold_start_handoff():
    # 10 trials should use hierarchical RT z-score estimator
    accs_short = [1] * 8 + [0] * 2
    rts_short = [5.0] * 10
    profile_short = analyze_pacing_and_caution(accs_short, rts_short)
    assert profile_short.active_estimator == "hierarchical_rt_z_score"
    assert profile_short.caution_label in ["impulsive", "cautious", "balanced"]
    assert hasattr(profile_short, "tau_speed")

    # 18 trials should dispatch to ez_diffusion
    accs_long = [1] * 14 + [0] * 4
    rts_long = [10.0 + (i % 4) for i in range(18)]
    profile_long = analyze_pacing_and_caution(accs_long, rts_long)
    assert profile_long.active_estimator == "ez_diffusion"
    assert profile_long.ddm_result is not None


def test_error_taxonomy_classification():
    from aile.error_taxonomy import classify_interaction_error

    # 1. Impulsive
    err_imp = classify_interaction_error(
        is_correct=False, skill_id="linear_equations_1var", prior_mastery=0.6,
        retrievability=0.8, response_time=2.5, expected_intensity_sec=14.0
    )
    assert err_imp.error_type == "impulsive_error"

    # 2. Retrieval failure
    err_ret = classify_interaction_error(
        is_correct=False, skill_id="factoring_polynomials", prior_mastery=0.85,
        retrievability=0.45, response_time=12.0, expected_intensity_sec=12.0
    )
    assert err_ret.error_type == "retrieval_failure"

    # 3. Careless slip
    err_slip = classify_interaction_error(
        is_correct=False, skill_id="order_of_operations", prior_mastery=0.90,
        retrievability=0.90, response_time=11.0, expected_intensity_sec=10.0
    )
    assert err_slip.error_type == "careless_slip"

    # 4. Knowledge deficit
    err_def = classify_interaction_error(
        is_correct=False, skill_id="quadratic_equations", prior_mastery=0.25,
        retrievability=0.50, response_time=13.0, expected_intensity_sec=15.0,
        root_cause_prereq="arithmetic_integers"
    )
    assert err_def.error_type == "knowledge_deficit"

    # 5. Cognitive overload
    err_over = classify_interaction_error(
        is_correct=False, skill_id="quadratic_functions_graph", prior_mastery=0.5,
        retrievability=0.7, response_time=25.0, expected_intensity_sec=12.0,
        p_fatigued=0.80, hint_requested=True
    )
    assert err_over.error_type == "cognitive_overload"


def test_skill_graph_dag_structure():
    graph = SkillGraph()
    assert graph.num_skills >= 8
    topo = graph.get_topological_order()
    assert "arithmetic_integers" in topo
    # arithmetic_integers must appear before order_of_operations
    assert topo.index("arithmetic_integers") < topo.index("order_of_operations")

    # Transitive ancestors of quadratic_equations must include arithmetic_integers
    ancestors = graph.get_all_ancestors("quadratic_equations")
    assert "arithmetic_integers" in ancestors
    assert "factoring_polynomials" in ancestors


def test_gnn_root_cause_isolation():
    graph = SkillGraph()
    gnn = SkillGNNPredictor()

    # Injected gap in foundational integer arithmetic
    mastery_means = {s: 0.85 for s in graph.skill_ids}
    mastery_means["arithmetic_integers"] = 0.25  # severe gap upstream

    # Student struggling on downstream quadratic_equations
    beliefs, root_cause, _ = gnn.propagate_and_trace_root_cause(
        graph,
        mastery_means=mastery_means,
        struggling_skill="quadratic_equations",
    )
    # The GNN/heuristic should identify arithmetic_integers as the root cause
    assert root_cause == "arithmetic_integers"


def test_memory_retrievability_decay():
    mem = MemoryModel()
    state = mem.update("linear_equations_1var", is_correct=True)
    assert state.stability > mem.default_stability
    assert state.retrievability == 1.0

    # Retrievability right after update is 1.0
    r = mem.compute_retrievability("linear_equations_1var")
    assert pytest.approx(r, 0.01) == 1.0


def test_metacognitive_calibration():
    # Overconfident: student rates 0.95 confidence but gets only 20% right
    confs = [0.95] * 10
    accs = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 20%
    profile = compute_metacognitive_calibration(confs, accs)
    assert profile.bias_label == "overconfident"
    assert profile.calibration_bias > 0.15


def test_pomdp_rollout_planner_latency_and_output():
    graph = SkillGraph()
    sim = StudentSimulator(graph)
    planner = POMDPRolloutPlanner(graph, sim)

    mastery_means = {s: 0.55 for s in graph.skill_ids}
    res = planner.plan(
        mastery_means=mastery_means,
        p_fatigued=0.20,
        root_cause_skill=None,
        struggling_skill="linear_equations_1var",
        max_depth=3,
        branching_limit=3,
    )

    assert res.best_action is not None
    assert res.depth_reached == 3
    assert len(res.rollout_tree) > 0
    # Must execute within sub-second limit (1000ms)
    assert res.planning_time_ms < 1000.0, f"Rollout took {res.planning_time_ms}ms, exceeding 1000ms budget"


def test_dkt_lstm_forward_and_weights():
    from pathlib import Path
    from aile.mastery_dkt import DKTMasteryPredictor

    weights_path = Path(__file__).resolve().parent.parent / "models" / "dkt.pt"
    predictor = DKTMasteryPredictor(num_skills=9, weights_path=weights_path)
    assert predictor.is_loaded is True

    # 6 interactions
    history = [(0, True), (0, True), (1, False), (1, True), (3, False), (0, True)]
    preds = predictor.predict_mastery_from_history(history)
    assert len(preds) == 9
    for skill_idx, p in preds.items():
        assert 0.0 <= p <= 1.0
