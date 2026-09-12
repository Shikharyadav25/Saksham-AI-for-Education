import pickle
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aile.attention_hmm import FatigueHMM

CORPUS_PATH = Path(__file__).resolve().parent / "synthetic_corpus.pkl"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def extract_rolling_observations(
    trajectories: List[List[dict]],
    window_size: int = 5,
    pop_mean_rt: float = 12.0,
    pop_std_rt: float = 5.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts rolling window feature vectors [accuracy, rt_z_score, hint_rate]
    across synthetic student interaction trajectories.
    """
    all_observations = []
    sequence_lengths = []

    for traj in trajectories:
        if len(traj) < window_size:
            continue

        session_obs = []
        for i in range(window_size, len(traj) + 1):
            window = traj[i - window_size : i]
            acc = float(np.mean([w["is_correct"] for w in window]))
            mean_rt = float(np.mean([w["reaction_time"] for w in window]))
            rt_z = (mean_rt - pop_mean_rt) / pop_std_rt
            hints = float(np.mean([w["hint_requested"] for w in window]))

            session_obs.append([acc, rt_z, hints])

        if session_obs:
            all_observations.extend(session_obs)
            sequence_lengths.append(len(session_obs))

    return np.array(all_observations, dtype=np.float64), sequence_lengths


def train_hmm_model(
    corpus_path: Path = CORPUS_PATH,
    save_path: Path = MODELS_DIR / "hmm.pkl",
) -> None:
    if not corpus_path.exists():
        from training.generate_corpus import generate_synthetic_corpus
        generate_synthetic_corpus(output_path=corpus_path)

    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    trajectories = corpus["trajectories"]
    print(f"Extracting rolling observation windows from {len(trajectories)} trajectories...")

    obs, lengths = extract_rolling_observations(trajectories)
    print(f"Total observation points: {len(obs)} across {len(lengths)} sessions.")

    hmm = FatigueHMM(model_path=save_path)
    print("Fitting 2-state Gaussian HMM on multivariate observations...")
    hmm.fit(obs, lengths)
    hmm.save(save_path)

    print(f"HMM model fitted and saved to {save_path}")
    print(f"Fitted means:\n{hmm.model.means_}")
    print(f"Fatigued state index identified: {hmm.fatigued_state_idx}")


if __name__ == "__main__":
    train_hmm_model()
