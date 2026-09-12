from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "dkt.pt"


class DKTNet(nn.Module):
    """
    Deep Knowledge Tracing LSTM network (Piech et al., 2015).
    Takes sequence of (skill_id, is_correct) one-hot vectors and predicts
    the probability of answering correctly on every skill at the next step.
    """

    def __init__(self, num_skills: int, hidden_dim: int = 64):
        super().__init__()
        self.num_skills = num_skills
        self.input_dim = 2 * num_skills
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, self.num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x shape: [batch_size, seq_len, 2 * num_skills]
        Returns:
            predictions: [batch_size, seq_len, num_skills]
            hidden: updated (h_n, c_n)
        """
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        preds = self.sigmoid(logits)
        return preds, hidden

    def encode_interaction(self, skill_idx: int, is_correct: bool) -> torch.Tensor:
        """
        Encodes a single interaction into a 2 * num_skills one-hot vector.
        Index [0, num_skills - 1] for correct; [num_skills, 2*num_skills - 1] for incorrect.
        """
        vec = torch.zeros(self.input_dim, dtype=torch.float32)
        offset = 0 if is_correct else self.num_skills
        target_idx = offset + skill_idx
        if 0 <= target_idx < self.input_dim:
            vec[target_idx] = 1.0
        return vec


class DKTMasteryPredictor:
    """Wrapper managing inference, weight loading, and state tracking for DKT."""

    def __init__(
        self,
        num_skills: int,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        device: Optional[str] = None,
    ):
        self.num_skills = num_skills
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = DKTNet(num_skills=num_skills).to(self.device)
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

    def predict_mastery_from_history(
        self,
        history: List[Tuple[int, bool]],
    ) -> Dict[int, float]:
        """
        Runs the full interaction sequence through the LSTM and returns
        the latest step's predicted P(correct) for all skills.
        """
        if not history:
            return {i: 0.5 for i in range(self.num_skills)}

        self.model.eval()
        with torch.no_grad():
            seq_tensors = [self.model.encode_interaction(s_idx, corr) for s_idx, corr in history]
            # Shape: [1, seq_len, 2 * num_skills]
            x_seq = torch.stack(seq_tensors).unsqueeze(0).to(self.device)
            preds, _ = self.model(x_seq)
            latest_preds = preds[0, -1, :].cpu().numpy()

        return {i: float(latest_preds[i]) for i in range(self.num_skills)}
