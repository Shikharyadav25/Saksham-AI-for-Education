import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from hmmlearn.hmm import GaussianHMM

DEFAULT_HMM_PATH = Path(__file__).resolve().parent.parent / "models" / "hmm.pkl"


class FatigueHMM:
    """
    2-state Gaussian Hidden Markov Model for tracking cognitive engagement and fatigue.
    Latent states:
        State 0: Engaged (higher accuracy, normal/efficient RT, low hint usage)
        State 1: Fatigued (degraded accuracy, elevated/erratic RT, increased hint requests)
    Observations: [accuracy, rt_z_score, hint_rate]
    """

    def __init__(self, model_path: Path = DEFAULT_HMM_PATH):
        self.model_path = model_path
        self.model: Optional[GaussianHMM] = None
        self.fatigued_state_idx: int = 1
        self.is_loaded: bool = False

        if model_path.exists():
            self.load(model_path)
        else:
            self._init_default_priors()

    def _init_default_priors(self) -> None:
        """Initializes a reasonable prior HMM before empirical training."""
        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        # Prior probabilities: student starts engaged
        model.startprob_ = np.array([0.90, 0.10])
        # Transition matrix: fatigue is sticky once entered
        model.transmat_ = np.array([
            [0.92, 0.08],
            [0.10, 0.90],
        ])
        # Means: [accuracy, rt_z_score, hint_rate]
        # State 0 (engaged): acc ~ 0.75, rt_z ~ 0.0, hint ~ 0.10
        # State 1 (fatigued): acc ~ 0.40, rt_z ~ 1.5, hint ~ 0.45
        model.means_ = np.array([
            [0.75, 0.0, 0.10],
            [0.40, 1.5, 0.45],
        ])
        model.covars_ = np.array([
            [0.04, 0.5, 0.03],
            [0.06, 0.8, 0.05],
        ])
        self.model = model
        self.fatigued_state_idx = 1
        self.is_loaded = True

    def fit(self, sequences: List[np.ndarray], lengths: List[int]) -> None:
        """
        Trains HMM on synthetic student trajectory observations.
        sequences: concatenated 2D array of shape [total_observations, 3]
        lengths: list of sequence lengths per simulated session
        """
        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=150,
            random_state=42,
        )
        model.fit(sequences, lengths)

        # Ensure state 1 represents the fatigued state (lower accuracy mean)
        acc_means = model.means_[:, 0]
        if acc_means[0] < acc_means[1]:
            # State 0 has lower accuracy -> State 0 is fatigued
            self.fatigued_state_idx = 0
        else:
            self.fatigued_state_idx = 1

        self.model = model
        self.is_loaded = True

    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "fatigued_state_idx": self.fatigued_state_idx,
            }, f)

    def load(self, path: Optional[Path] = None) -> None:
        load_path = path or self.model_path
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.fatigued_state_idx = data["fatigued_state_idx"]
                self.is_loaded = True
        except Exception:
            self._init_default_priors()

    def predict_fatigue_probability(self, observations: np.ndarray) -> float:
        """
        Computes the posterior probability P(fatigued) given recent observation sequence.
        observations: shape [window_size, 3]
        """
        if self.model is None or observations.shape[0] == 0:
            return 0.10

        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        try:
            # Predict state posteriors
            posteriors = self.model.predict_proba(obs)
            latest_posterior = posteriors[-1]
            p_fatigued = float(latest_posterior[self.fatigued_state_idx])
            return max(0.01, min(0.99, p_fatigued))
        except Exception:
            # Fallback heuristic if covariance matrix becomes ill-conditioned
            mean_acc = float(np.mean(obs[:, 0]))
            mean_rt_z = float(np.mean(obs[:, 1]))
            if mean_acc < 0.45 and mean_rt_z > 1.0:
                return 0.85
            return 0.15
