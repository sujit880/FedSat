import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:
    KMeans = None
    GaussianMixture = None


class Strategy(Enum):
    GENERALIZE = 0
    PERSONALIZE = 1
    HYBRID = 2


@dataclass
class ClientFeatures:
    delta_loss: float
    grad_norm: float
    cos_sim: float
    entropy: float
    participation: int
    local_val_acc: float
    global_val_acc: float
    cluster_id: int = -1

    def to_vector(self, include_cluster: bool = True) -> np.ndarray:
        vec = np.array(
            [
                float(self.delta_loss),
                float(self.grad_norm),
                float(self.cos_sim),
                float(self.entropy),
                float(self.participation),
                float(self.local_val_acc),
                float(self.global_val_acc),
            ],
            dtype=np.float32,
        )
        if include_cluster:
            vec = np.concatenate([vec, np.array([float(self.cluster_id)], dtype=np.float32)])
        return vec

    @staticmethod
    def clip_bounds(vec: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        return np.clip(vec, bounds[0], bounds[1])


class StrategyControllerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyController:
    """Lightweight controller that learns to map client statistics to strategies.

    The controller is trained using pseudo-labels produced by the counterfactual oracle.
    It only uses aggregated client statistics (no raw data) and supports DP noise.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = device
        self.model = StrategyControllerMLP(feature_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.memory_x: List[torch.Tensor] = []
        self.memory_y: List[torch.Tensor] = []
        self.criterion = nn.CrossEntropyLoss()
        self.trained_steps: int = 0
        self.ready: bool = False

    def add_sample(self, features: ClientFeatures, label: Strategy, noise_std: float = 0.0) -> None:
        vec = features.to_vector(include_cluster=True)
        if noise_std > 0:
            vec = vec + np.random.normal(scale=noise_std, size=vec.shape).astype(np.float32)
        x = torch.from_numpy(vec).float()
        y = torch.tensor(label.value, dtype=torch.long)
        self.memory_x.append(x)
        self.memory_y.append(y)

    def maybe_train(self, epochs: int = 5, batch_size: int = 32, min_samples: int = 16) -> None:
        if len(self.memory_x) < min_samples:
            return
        dataset = list(zip(self.memory_x, self.memory_y))
        random.shuffle(dataset)
        for _ in range(epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                bx = torch.stack([b[0] for b in batch]).to(self.device)
                by = torch.stack([b[1] for b in batch]).to(self.device)
                logits = self.model(bx)
                loss = self.criterion(logits, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.trained_steps += epochs
        self.ready = True

    @torch.no_grad()
    def predict(self, features: ClientFeatures) -> Strategy:
        vec = torch.from_numpy(features.to_vector(include_cluster=True)).float().to(self.device)
        logits = self.model(vec.unsqueeze(0))
        pred = int(logits.argmax(dim=1).item())
        return Strategy(pred)

    def cluster(self, feats: List[ClientFeatures], k: int = 3, use_gmm: bool = False) -> List[int]:
        if len(feats) == 0:
            return []
        X = np.stack([f.to_vector(include_cluster=False) for f in feats], axis=0)
        if use_gmm and GaussianMixture is not None:
            model = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            labels = model.fit_predict(X)
        elif KMeans is not None:
            model = KMeans(n_clusters=k, n_init="auto") if hasattr(KMeans, "n_init") else KMeans(n_clusters=k)
            labels = model.fit_predict(X)
        else:
            # Fallback: random partitioning (still deterministic given seed if set elsewhere)
            labels = np.mod(np.arange(len(feats)), k)
        return labels.tolist()


class DRLController:
    """Actor-critic bandit controller trained from real reward signals.

    Instead of distilling soft targets, this controller samples an action
    (strategy) from its policy, receives a scalar reward, and updates with a
    policy-gradient + value (baseline) loss. Rewards can come from the Oracle's
    counterfactual scores or from live rollouts; the code is agnostic.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gamma: float = 0.9,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = device
        # Shared torso + separate policy/value heads
        self.torso = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        self.policy_head = nn.Linear(hidden_dim, len(Strategy)).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.torso.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        # memory holds (state, action, reward, next_state, done)
        self.memory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.trained_steps: int = 0
        self.ready: bool = False

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def add_transition(
        self,
        state: ClientFeatures,
        action: Strategy,
        reward: float,
        next_state: Optional[ClientFeatures] = None,
        noise_std: float = 0.0,
    ) -> None:
        """Store a transition (s, a, r, s').

        If next_state is missing, we treat it as terminal for this transition.
        """
        s_vec = state.to_vector(include_cluster=True)
        if noise_std > 0:
            s_vec = s_vec + np.random.normal(scale=noise_std, size=s_vec.shape).astype(np.float32)
        x = torch.from_numpy(s_vec).float()
        if next_state is not None:
            ns_vec = next_state.to_vector(include_cluster=True)
            if noise_std > 0:
                ns_vec = ns_vec + np.random.normal(scale=noise_std, size=ns_vec.shape).astype(np.float32)
            nx = torch.from_numpy(ns_vec).float()
            done = torch.tensor(0.0, dtype=torch.float)
        else:
            nx = torch.zeros_like(x)
            done = torch.tensor(1.0, dtype=torch.float)

        a = torch.tensor(action.value, dtype=torch.long)
        r = torch.tensor(float(reward), dtype=torch.float)
        self.memory.append((x, a, r, nx, done))

    def maybe_train(self, epochs: int = 5, batch_size: int = 32, min_samples: int = 16) -> None:
        if len(self.memory) < min_samples:
            return
        dataset = list(self.memory)
        random.shuffle(dataset)
        for _ in range(epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                bx = torch.stack([b[0] for b in batch]).to(self.device)
                ba = torch.stack([b[1] for b in batch]).to(self.device)
                br = torch.stack([b[2] for b in batch]).to(self.device)
                bnx = torch.stack([b[3] for b in batch]).to(self.device)
                bdone = torch.stack([b[4] for b in batch]).to(self.device)

                logits, values = self._forward(bx)
                with torch.no_grad():
                    _, next_values = self._forward(bnx)
                    targets = br + self.gamma * (1.0 - bdone) * next_values

                log_probs = torch.log_softmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                chosen_logp = log_probs.gather(1, ba.unsqueeze(1)).squeeze(1)
                advantage = targets - values
                policy_loss = -(chosen_logp * advantage.detach()).mean()
                value_loss = torch.mean((values - targets.detach()) ** 2)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.trained_steps += epochs
        self.ready = True

    @torch.no_grad()
    def predict(self, features: ClientFeatures) -> Strategy:
        vec = torch.from_numpy(features.to_vector(include_cluster=True)).float().to(self.device)
        logits, _ = self._forward(vec.unsqueeze(0))
        pred = int(logits.argmax(dim=1).item())
        return Strategy(pred)

    def cluster(self, feats: List[ClientFeatures], k: int = 3, use_gmm: bool = False) -> List[int]:
        """Mirror the StrategyController clustering to keep pipeline compatibility."""
        if len(feats) == 0:
            return []
        X = np.stack([f.to_vector(include_cluster=False) for f in feats], axis=0)
        if use_gmm and GaussianMixture is not None:
            model = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            labels = model.fit_predict(X)
        elif KMeans is not None:
            model = KMeans(n_clusters=k, n_init="auto") if hasattr(KMeans, "n_init") else KMeans(n_clusters=k)
            labels = model.fit_predict(X)
        else:
            labels = np.mod(np.arange(len(feats)), k)
        return labels.tolist()


class OracleLabeler:
    """Counterfactual oracle to assign pseudo-labels without ground-truth supervision."""

    def __init__(
        self,
        simulate_rounds: int = 1,
        score_weights: Tuple[float, float, float] = (0.6, 0.4, 0.1),
        device: str | torch.device = "cpu",
    ) -> None:
        self.sim_rounds = simulate_rounds
        self.w_global, self.w_local, self.w_var = score_weights
        self.device = device

    def _score(self, global_improve: float, local_improve: float, variance: float) -> float:
        return self.w_global * global_improve + self.w_local * local_improve - self.w_var * variance

    def label_clients(
        self,
        clients,
        global_state: Dict,
        prev_global_state: Optional[Dict],
        server_test_fn,
        aggregate_fn,
        prepare_model_fn=None,
        alpha: float = 0.5,
        clients_per_round: Optional[int] = None,
        return_scores: bool = False,
    ) -> Dict[int, Strategy] | Tuple[Dict[int, Strategy], Dict[int, Dict[Strategy, float]]]:
        pseudo_labels: Dict[int, Strategy] = {}
        strategy_scores: Dict[int, Dict[Strategy, float]] = {}
        baseline_global_acc = server_test_fn(global_state)
        for client in clients:
            try:
                best_score = -math.inf
                best_strategy = Strategy.GENERALIZE
                per_strategy_scores: Dict[Strategy, float] = {}
                for strategy in Strategy:
                    if prepare_model_fn:
                        start_state = prepare_model_fn(client, strategy, global_state)
                    else:
                        start_state = global_state

                    sim_state, local_before, local_after = client.simulate_strategy(
                        start_state=start_state,
                        rounds=self.sim_rounds,
                    )
                    # aggregate a hypothetical global model using this client's simulated update
                    contrib = 0.1
                    if clients_per_round is not None and clients_per_round > 0:
                        contrib = 1.0 / float(clients_per_round)
                    contrib = max(1e-3, min(0.5, contrib))  # keep within a sane range
                    agg_state = aggregate_fn(
                        [(contrib, sim_state), (1.0 - contrib, global_state)],
                        global_state,
                    )
                    sim_global_acc = server_test_fn(agg_state)
                    score = self._score(
                        global_improve=float(sim_global_acc - baseline_global_acc),
                        local_improve=float(local_after - local_before),
                        variance=0.0,
                    )
                    per_strategy_scores[strategy] = score
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
                pseudo_labels[client.id] = best_strategy
                strategy_scores[client.id] = per_strategy_scores
            except Exception:
                pseudo_labels[client.id] = Strategy.HYBRID
        if return_scores:
            return pseudo_labels, strategy_scores
        return pseudo_labels
