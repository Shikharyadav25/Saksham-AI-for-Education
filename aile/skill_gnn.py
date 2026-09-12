from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .skill_graph import SkillGraph

DEFAULT_GNN_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "gnn.pt"


class GraphAttentionLayer(nn.Module):
    """
    Message passing layer with learned attention coefficients.
    Propagates uncertainty-weighted mastery signal along prerequisite edges.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [num_nodes, in_features]
        edge_index: [2, num_edges] (src, dst)
        Returns:
            h_out: [num_nodes, out_features]
            alpha: [num_edges] attention weights
        """
        num_nodes = h.size(0)
        h_proj = self.W(h)  # [num_nodes, out_features]

        if edge_index.size(1) == 0:
            return h_proj, torch.empty((0,), device=h.device)

        src, dst = edge_index[0], edge_index[1]

        # Self-loops so node retains its own belief
        self_loop_indices = torch.arange(num_nodes, device=h.device)
        src_all = torch.cat([src, self_loop_indices])
        dst_all = torch.cat([dst, self_loop_indices])

        # Attention coefficients
        score_src = self.a_src(h_proj[src_all]).squeeze(-1)
        score_dst = self.a_dst(h_proj[dst_all]).squeeze(-1)
        e = self.leaky_relu(score_src + score_dst)

        # Softmax per destination node
        # Compute exp(e) with max-subtraction for numerical stability
        exp_e = torch.exp(e - e.max())
        denom = torch.zeros(num_nodes, device=h.device)
        denom.scatter_add_(0, dst_all, exp_e)
        alpha = exp_e / (denom[dst_all] + 1e-8)

        # Message passing aggregation
        messages = h_proj[src_all] * alpha.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.out_features), device=h.device)
        out.scatter_add_(0, dst_all.unsqueeze(-1).expand(-1, self.out_features), messages)

        # Return original edge attention (excluding added self loops) for visualization
        num_orig_edges = edge_index.size(1)
        return out, alpha[:num_orig_edges]


class PrerequisiteGNN(nn.Module):
    """
    2-layer Graph Attention Network over the prerequisite skill DAG.
    Input features: [mastery_mean, mastery_variance]
    Output: adjusted mastery belief per node with residual belief modulation.
    """

    def __init__(self, in_features: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_features, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1, _ = self.gat1(x, edge_index)
        h1 = F.elu(h1)
        h2, attn_weights = self.gat2(h1, edge_index)
        
        # Residual modulation: start with observed mastery x[:, 0] and adjust via graph signal
        delta = torch.tanh(h2.squeeze(-1)) * 0.25
        beliefs = torch.clamp(x[:, 0] + delta, 0.02, 0.98)
        return beliefs, attn_weights


class SkillGNNPredictor:
    """Manages GNN inference, weights, and root-cause prerequisite tracing."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_GNN_WEIGHTS_PATH,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = PrerequisiteGNN().to(self.device)
        self.is_loaded = False
        self.weights_path = weights_path

        if weights_path.exists():
            self.load_weights(weights_path)

    def load_weights(self, path: Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
        except Exception:
            self.is_loaded = False

    def propagate_and_trace_root_cause(
        self,
        skill_graph: SkillGraph,
        mastery_means: Dict[str, float],
        mastery_vars: Optional[Dict[str, float]] = None,
        struggling_skill: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[str], Dict[Tuple[str, str], float]]:
        """
        Executes GNN message passing across the DAG.
        Returns:
            adjusted_beliefs: {skill_id: propagated_mastery}
            root_cause_skill: upstream ancestor with lowest post-propagation belief
            attention_map: {(prereq, skill): attention_weight}
        """
        x = skill_graph.build_node_features(mastery_means, mastery_vars).to(self.device)
        edge_index = skill_graph.build_edge_index_tensor().to(self.device)

        if not self.is_loaded:
            # If weights not yet trained, return raw mastery means and heuristic root cause
            rc = None
            if struggling_skill:
                rc = skill_graph.trace_root_cause_heuristic(struggling_skill, mastery_means)
            return mastery_means, rc, {}

        self.model.eval()
        with torch.no_grad():
            beliefs_tensor, attn_weights = self.model(x, edge_index)
            beliefs_arr = beliefs_tensor.cpu().numpy()

        adjusted_beliefs = {
            skill_graph.index_to_skill[i]: float(beliefs_arr[i])
            for i in range(skill_graph.num_skills)
        }

        # Build human-readable attention map
        attention_map: Dict[Tuple[str, str], float] = {}
        if edge_index.size(1) > 0 and attn_weights.numel() > 0:
            attn_arr = attn_weights.cpu().numpy()
            srcs = edge_index[0].cpu().numpy()
            dsts = edge_index[1].cpu().numpy()
            for s_idx, d_idx, w in zip(srcs, dsts, attn_arr):
                src_id = skill_graph.index_to_skill[int(s_idx)]
                dst_id = skill_graph.index_to_skill[int(d_idx)]
                attention_map[(src_id, dst_id)] = float(w)

        # Identify root cause: lowest post-propagation belief among ancestors
        root_cause_skill = None
        if struggling_skill:
            ancestors = skill_graph.get_all_ancestors(struggling_skill)
            if ancestors:
                # Rank ancestors by adjusted belief
                ranked_ancestors = sorted(ancestors, key=lambda s: adjusted_beliefs.get(s, 1.0))
                lowest_ancestor = ranked_ancestors[0]
                # Only flag as root cause if belief is lower than a reasonable adequacy threshold (e.g. 0.7)
                if adjusted_beliefs.get(lowest_ancestor, 1.0) < 0.70:
                    root_cause_skill = lowest_ancestor

        return adjusted_beliefs, root_cause_skill, attention_map
