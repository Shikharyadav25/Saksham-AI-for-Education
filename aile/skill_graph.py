import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "questions.json"


class SkillGraph:
    """
    Represents the prerequisite curriculum Directed Acyclic Graph (DAG).
    Direct edges point from prerequisite -> dependent skill.
    """

    def __init__(self, config_path: Path = DEFAULT_QUESTIONS_PATH):
        self.graph = nx.DiGraph()
        self.skill_metadata: Dict[str, Dict[str, Any]] = {}
        self.skill_to_index: Dict[str, int] = {}
        self.index_to_skill: Dict[int, str] = {}
        self._load_from_config(config_path)

    def _load_from_config(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        skills = data.get("skills", [])
        for idx, skill_info in enumerate(skills):
            sid = skill_info["id"]
            self.skill_to_index[sid] = idx
            self.index_to_skill[idx] = sid
            self.skill_metadata[sid] = skill_info
            self.graph.add_node(
                sid,
                name=skill_info.get("name", sid),
                subject=skill_info.get("subject", "General Mathematics"),
            )

        for skill_info in skills:
            sid = skill_info["id"]
            for prereq in skill_info.get("prerequisites", []):
                if prereq in self.skill_metadata:
                    # Edge goes: prerequisite -> target skill
                    self.graph.add_edge(prereq, sid)

    @property
    def num_skills(self) -> int:
        return len(self.skill_to_index)

    @property
    def skill_ids(self) -> List[str]:
        return list(self.skill_to_index.keys())

    def get_subjects(self) -> List[str]:
        """Returns ordered list of unique major subjects present in the curriculum."""
        subjects = []
        for s in self.skill_metadata.values():
            subj = s.get("subject", "General Mathematics")
            if subj not in subjects:
                subjects.append(subj)
        return subjects

    def get_skills_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Returns metadata of skills belonging to the specified subject."""
        return [meta for meta in self.skill_metadata.values() if meta.get("subject") == subject]


    def get_prerequisites(self, skill_id: str) -> List[str]:
        """Immediate parents in the prerequisite graph."""
        if skill_id not in self.graph:
            return []
        return list(self.graph.predecessors(skill_id))

    def get_all_ancestors(self, skill_id: str) -> Set[str]:
        """All upstream transitive prerequisites."""
        if skill_id not in self.graph:
            return set()
        return nx.ancestors(self.graph, skill_id)

    def get_dependents(self, skill_id: str) -> List[str]:
        """Immediate downstream skills that require this skill."""
        if skill_id not in self.graph:
            return []
        return list(self.graph.successors(skill_id))

    def get_topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def trace_root_cause_heuristic(
        self,
        struggling_skill: str,
        mastery_estimates: Dict[str, float],
        threshold: float = 0.6,
    ) -> Optional[str]:
        """
        V1 baseline: BFS up the prerequisite tree to find the earliest ancestor
        whose mastery falls below the threshold.
        """
        ancestors = self.get_all_ancestors(struggling_skill)
        if not ancestors:
            return None

        # Sort ancestors by topological order so we check foundational skills first
        topo_order = self.get_topological_order()
        sorted_ancestors = [s for s in topo_order if s in ancestors]

        for ancestor in sorted_ancestors:
            if mastery_estimates.get(ancestor, 0.5) < threshold:
                return ancestor
        return None

    def build_edge_index_tensor(self):
        """
        Builds edge_index tensor (shape [2, num_edges]) for PyTorch / GNN message passing.
        """
        import torch

        srcs = []
        dsts = []
        for src, dst in self.graph.edges():
            srcs.append(self.skill_to_index[src])
            dsts.append(self.skill_to_index[dst])

        if not srcs:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor([srcs, dsts], dtype=torch.long)

    def build_node_features(
        self,
        mastery_means: Dict[str, float],
        mastery_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Constructs [num_skills, 2] feature matrix where each row is [mean_mastery, variance].
        """
        import torch

        vars_dict = mastery_vars or {}
        features = []
        for i in range(self.num_skills):
            sid = self.index_to_skill[i]
            mean_m = mastery_means.get(sid, 0.5)
            var_m = vars_dict.get(sid, 0.08)  # default prior variance for Beta(2,2)
            features.append([mean_m, var_m])

        return torch.tensor(features, dtype=torch.float32)
