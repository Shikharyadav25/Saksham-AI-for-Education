import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aile.mastery_dkt import DKTNet
from aile.skill_graph import SkillGraph

CORPUS_PATH = Path(__file__).resolve().parent / "synthetic_corpus.pkl"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class DKTDataset(Dataset):
    def __init__(self, trajectories, num_skills, max_seq_len=40):
        self.samples = []
        self.num_skills = num_skills
        self.input_dim = 2 * num_skills

        for traj in trajectories:
            if len(traj) < 2:
                continue
            seq_len = min(len(traj), max_seq_len)
            x_seq = torch.zeros((seq_len - 1, self.input_dim), dtype=torch.float32)
            target_skill_idx = torch.zeros(seq_len - 1, dtype=torch.long)
            target_correctness = torch.zeros(seq_len - 1, dtype=torch.float32)

            for t in range(seq_len - 1):
                # Input at t: (skill_idx, is_correct)
                curr = traj[t]
                s_curr = curr["skill_idx"]
                c_curr = curr["is_correct"]
                offset = 0 if c_curr == 1 else num_skills
                x_seq[t, offset + s_curr] = 1.0

                # Target at t+1: did student get next question right?
                nxt = traj[t + 1]
                target_skill_idx[t] = nxt["skill_idx"]
                target_correctness[t] = float(nxt["is_correct"])

            self.samples.append((x_seq, target_skill_idx, target_correctness))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_dkt_model(
    corpus_path: Path = CORPUS_PATH,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.005,
    save_path: Path = MODELS_DIR / "dkt.pt",
) -> None:
    if not corpus_path.exists():
        from training.generate_corpus import generate_synthetic_corpus
        generate_synthetic_corpus(output_path=corpus_path)

    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    trajectories = corpus["trajectories"]
    skill_graph = SkillGraph()
    num_skills = skill_graph.num_skills

    # Train / val split (80 / 20)
    split_idx = int(len(trajectories) * 0.8)
    train_data = DKTDataset(trajectories[:split_idx], num_skills)
    val_data = DKTDataset(trajectories[split_idx:], num_skills)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DKTNet(num_skills=num_skills, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Training DKT LSTM on {len(train_data)} sequences ({epochs} epochs on {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        # Iterate sequences
        for x_seq, target_skill, target_corr in train_data:
            x_batch = x_seq.unsqueeze(0).to(device)  # [1, seq_len, 2*num_skills]
            preds, _ = model(x_batch)                # [1, seq_len, num_skills]

            # Gather predictions for the specific next skill attempted
            target_s = target_skill.to(device)
            target_c = target_corr.to(device)
            seq_len = x_seq.size(0)

            step_indices = torch.arange(seq_len, device=device)
            relevant_preds = preds[0, step_indices, target_s]

            loss = criterion(relevant_preds, target_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {avg_loss:.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"DKT model saved successfully to {save_path}")


if __name__ == "__main__":
    train_dkt_model()
