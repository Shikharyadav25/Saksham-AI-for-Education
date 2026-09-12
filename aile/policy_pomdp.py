import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .policy import ActionCandidate, evaluate_belief_utility
from .simulator import SimulatedStudentProfile, StudentSimulator
from .skill_graph import SkillGraph


@dataclass
class RolloutNode:
    action_type: str
    target_skill: Optional[str]
    description: str
    depth: int
    expected_score: float = 0.0
    children: List["RolloutNode"] = field(default_factory=list)


@dataclass
class POMDPPlanningResult:
    best_action: ActionCandidate
    rollout_tree: List[Dict[str, Any]]
    planning_time_ms: float
    depth_reached: int
    total_rollouts: int
    explanation: str


class POMDPRolloutPlanner:
    """
    Pedagogical policy planner using Depth-3 Monte Carlo tree lookahead rollouts.
    Simulates forward transitions using the environment simulator to optimize multi-step
    learning trajectories instead of greedy one-step decisions.
    """

    def __init__(self, skill_graph: SkillGraph, simulator: StudentSimulator):
        self.skill_graph = skill_graph
        self.simulator = simulator

    def generate_candidate_actions(
        self,
        mastery_means: Dict[str, float],
        p_fatigued: float,
        root_cause_skill: Optional[str],
        decayed_skills: Optional[List[str]] = None,
        struggling_skill: Optional[str] = None,
    ) -> List[ActionCandidate]:
        candidates: List[ActionCandidate] = []
        decayed = decayed_skills or []

        # 1. Rest break option
        if p_fatigued > 0.35:
            candidates.append(
                ActionCandidate(
                    action_type="rest_break",
                    target_skill=None,
                    description="Take a 3-minute cognitive micro-break to reset attention and reduce fatigue.",
                )
            )

        # 2. Remediate root cause if identified
        if root_cause_skill:
            candidates.append(
                ActionCandidate(
                    action_type="remediate_root_cause",
                    target_skill=root_cause_skill,
                    description=f"Remediate foundational prerequisite bottleneck: {root_cause_skill}.",
                )
            )

        # 3. Spaced review for decayed skills
        for d_skill in decayed[:2]:
            candidates.append(
                ActionCandidate(
                    action_type="spaced_review",
                    target_skill=d_skill,
                    description=f"Spaced review for retention on {d_skill}.",
                )
            )

        # 4. Scaffolding on struggling skill
        if struggling_skill:
            candidates.append(
                ActionCandidate(
                    action_type="scaffold_hint",
                    target_skill=struggling_skill,
                    description=f"Socratic scaffolding & hints for {struggling_skill}.",
                )
            )
            candidates.append(
                ActionCandidate(
                    action_type="practice",
                    target_skill=struggling_skill,
                    description=f"Direct practice on current skill {struggling_skill}.",
                )
            )

        # 5. Advancement in topological order
        topo_order = self.skill_graph.get_topological_order()
        for s in topo_order:
            m = mastery_means.get(s, 0.5)
            prereqs = self.skill_graph.get_prerequisites(s)
            prereqs_met = all(mastery_means.get(p, 0.5) >= 0.65 for p in prereqs)

            if prereqs_met and m < 0.75:
                candidates.append(
                    ActionCandidate(
                        action_type="advance_new_skill",
                        target_skill=s,
                        description=f"Advance curriculum: Introduce {s}.",
                    )
                )
                break

        # Fallback if no specific candidates
        if not candidates:
            first_skill = topo_order[0]
            candidates.append(
                ActionCandidate(
                    action_type="practice",
                    target_skill=first_skill,
                    description=f"Practice {first_skill}.",
                )
            )

        # Deduplicate candidates
        unique_candidates: List[ActionCandidate] = []
        seen = set()
        for c in candidates:
            key = (c.action_type, c.target_skill)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        return unique_candidates

    def plan(
        self,
        mastery_means: Dict[str, float],
        p_fatigued: float,
        root_cause_skill: Optional[str],
        decayed_skills: Optional[List[str]] = None,
        struggling_skill: Optional[str] = None,
        max_depth: int = 3,
        branching_limit: int = 3,
    ) -> POMDPPlanningResult:
        start_time = time.perf_counter()
        decayed = decayed_skills or []

        # Candidate level 1 actions
        candidates_l1 = self.generate_candidate_actions(
            mastery_means, p_fatigued, root_cause_skill, decayed, struggling_skill
        )

        # Score immediate utility to prune to top candidates
        for c in candidates_l1:
            c.immediate_utility = evaluate_belief_utility(
                mastery_means, p_fatigued, root_cause_skill, decayed, c.action_type, c.target_skill
            )

        candidates_l1.sort(key=lambda x: x.immediate_utility, reverse=True)
        pruned_l1 = candidates_l1[:branching_limit]

        rollout_tree_data: List[Dict[str, Any]] = []
        best_overall_score = -1e9
        best_candidate = pruned_l1[0]
        total_rollouts = 0

        # Rollout loop
        for a1 in pruned_l1:
            l1_node = {
                "action": a1.action_type,
                "target": a1.target_skill,
                "description": a1.description,
                "immediate_utility": round(a1.immediate_utility, 2),
                "depth": 1,
                "branches": [],
            }

            # Simulate transition from a1
            sim_student_1 = self._create_surrogate_student(mastery_means, p_fatigued)
            s1_skill = a1.target_skill or self.skill_graph.skill_ids[0]
            corr1, _, _ = self.simulator.step(sim_student_1, s1_skill, a1.action_type)
            sim_m1 = dict(sim_student_1.true_mastery)
            sim_fatigue1 = 0.85 if sim_student_1.is_fatigued else max(0.05, p_fatigued * 0.9)
            if a1.action_type == "rest_break":
                sim_fatigue1 = 0.10

            # Level 2 candidates
            candidates_l2 = self.generate_candidate_actions(
                sim_m1, sim_fatigue1, root_cause_skill, decayed, s1_skill
            )
            for c in candidates_l2:
                c.immediate_utility = evaluate_belief_utility(
                    sim_m1, sim_fatigue1, root_cause_skill, decayed, c.action_type, c.target_skill
                )
            candidates_l2.sort(key=lambda x: x.immediate_utility, reverse=True)
            pruned_l2 = candidates_l2[:branching_limit]

            l1_scores: List[float] = []

            for a2 in pruned_l2:
                sim_student_2 = copy.deepcopy(sim_student_1)
                s2_skill = a2.target_skill or self.skill_graph.skill_ids[0]
                self.simulator.step(sim_student_2, s2_skill, a2.action_type)
                sim_m2 = dict(sim_student_2.true_mastery)
                sim_fatigue2 = 0.85 if sim_student_2.is_fatigued else max(0.05, sim_fatigue1 * 0.9)
                if a2.action_type == "rest_break":
                    sim_fatigue2 = 0.10

                # Level 3 candidates
                candidates_l3 = self.generate_candidate_actions(
                    sim_m2, sim_fatigue2, root_cause_skill, decayed, s2_skill
                )
                for c in candidates_l3:
                    c.immediate_utility = evaluate_belief_utility(
                        sim_m2, sim_fatigue2, root_cause_skill, decayed, c.action_type, c.target_skill
                    )
                candidates_l3.sort(key=lambda x: x.immediate_utility, reverse=True)
                top_a3 = candidates_l3[0] if candidates_l3 else a2

                # Final terminal score evaluation
                terminal_utility = evaluate_belief_utility(
                    sim_m2, sim_fatigue2, root_cause_skill, decayed, top_a3.action_type, top_a3.target_skill
                )
                cumulative_path_score = a1.immediate_utility + 0.8 * a2.immediate_utility + 0.6 * terminal_utility
                l1_scores.append(cumulative_path_score)
                total_rollouts += 1

                l1_node["branches"].append({
                    "action": a2.action_type,
                    "target": a2.target_skill,
                    "terminal_action": top_a3.action_type,
                    "path_score": round(cumulative_path_score, 2),
                })

            avg_score = float(sum(l1_scores) / max(1, len(l1_scores)))
            l1_node["expected_rollout_score"] = round(avg_score, 2)
            rollout_tree_data.append(l1_node)

            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_candidate = a1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Construct explanatory rationale
        explanation = (
            f"Evaluated {len(pruned_l1)} primary branches across depth {max_depth} ({total_rollouts} simulated paths). "
            f"Selected action '{best_candidate.action_type}' targeting '{best_candidate.target_skill or 'general'}' "
            f"yielding highest 3-step discounted utility ({best_overall_score:.2f})."
        )

        return POMDPPlanningResult(
            best_action=best_candidate,
            rollout_tree=rollout_tree_data,
            planning_time_ms=elapsed_ms,
            depth_reached=max_depth,
            total_rollouts=total_rollouts,
            explanation=explanation,
        )

    def _create_surrogate_student(
        self,
        mastery_means: Dict[str, float],
        p_fatigued: float,
    ) -> SimulatedStudentProfile:
        """Creates a surrogate student reflecting the current estimated belief state."""
        return SimulatedStudentProfile(
            student_id="surrogate_rollout",
            true_mastery=dict(mastery_means),
            learning_rate=0.15,
            base_rt=11.0,
            rt_variance=3.5,
            caution_a=0.14,
            fluency_v=0.18,
            guess_rate=0.20,
            slip_rate=0.10,
            fatigue_threshold=15,
            is_fatigued=(p_fatigued > 0.60),
        )
