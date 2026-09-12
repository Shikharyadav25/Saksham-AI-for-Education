import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aile.skill_gnn import PrerequisiteGNN
from aile.skill_graph import SkillGraph

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def generate_synthetic_graph_scenarios(
    skill_graph: SkillGraph,
    num_scenarios: int = 500,
):
    """
    Generates synthetic graph states with known ground-truth mastery distributions.
    Injected gaps in foundational prerequisites propagate lower beliefs to downstream skills.
    """
    scenarios = []
    topo_order = skill_graph.get_topological_order()

    for _ in range(num_scenarios):
        # Base mastery
        m_dict = {s: random.uniform(0.65, 0.95) for s in topo_order}
        v_dict = {s: random.uniform(0.02, 0.08) for s in topo_order}

        # 50% chance to inject a prerequisite bottleneck
        if random.random() < 0.70:
            bottleneck = random.choice(topo_order[:3])
            m_dict[bottleneck] = random.uniform(0.15, 0.35)
            v_dict[bottleneck] = 0.03

            # Downstream descendants inherit deficit
            descendants = skill_graph.get_dependents(bottleneck)
            for d in descendants:
                m_dict[d] = min(m_dict[d], random.uniform(0.25, 0.50))

        # Target post-propagation belief reflects ground truth
        target_beliefs = []
        for i in range(skill_graph.num_skills):
            sid = skill_graph.index_to_skill[i]
            target_beliefs.append(m_dict[sid])

        x = skill_graph.build_node_features(m_dict, v_dict)
        y = torch.tensor(target_beliefs, dtype=torch.float32)
        scenarios.append((x, y))

    return scenarios


def train_gnn_model(
    epochs: int = 30,
    lr: float = 0.01,
    save_path: Path = MODELS_DIR / "gnn.pt",
) -> None:
    skill_graph = SkillGraph()
    edge_index = skill_graph.build_edge_index_tensor()
    scenarios = generate_synthetic_graph_scenarios(skill_graph, num_scenarios=400)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PrerequisiteGNN(in_features=2, hidden_dim=16).to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training Prerequisite GNN on {len(scenarios)} graph states ({epochs} epochs on {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in scenarios:
            x_dev = x.to(device)
            y_dev = y.to(device)

            pred_beliefs, _ = model(x_dev, edge_index)
            loss = criterion(pred_beliefs, y_dev)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(scenarios)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} - GNN MSE Loss: {avg_loss:.5f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"GNN model saved successfully to {save_path}")


if __name__ == "__main__":
    train_gnn_model()
