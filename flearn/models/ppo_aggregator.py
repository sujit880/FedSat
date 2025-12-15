"""
PPO-based Federated Aggregator with Contextual Features

This module implements a PPO (Proximal Policy Optimization) agent for federated learning
aggregation that:
1. Uses client struggling scores, accuracies, and sample sizes as state features
2. Learns to adjust weights around a baseline (mean or FedAvg)
3. Uses advantage over baseline for reward shaping
4. Regularizes with KL penalty to prevent extreme deviations
5. Includes entropy bonus for exploration

Key advantages over DDPG/SAC for FL:
- Better suited for noisy, long-horizon environments
- Natural baseline comparison via advantage estimation
- Easier to constrain and regularize
- More stable training with PPO clipping
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict, deque
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


CLIENT_FEATURE_DIM_BASIC = 4
HISTORY_FEATURE_DIM_BASIC = 1
BASE_GLOBAL_FEATURE_DIM_BASIC = 5

CLIENT_FEATURE_DIM_ENHANCED = 8
HISTORY_FEATURE_DIM_ENHANCED = 4
BASE_GLOBAL_FEATURE_DIM_ENHANCED = 11


CONTEXT_VARIANTS = {
    "basic": {
        "client_dim": CLIENT_FEATURE_DIM_BASIC,
        "history_dim": HISTORY_FEATURE_DIM_BASIC,
        "global_base_dim": BASE_GLOBAL_FEATURE_DIM_BASIC,
    },
    "enhanced": {
        "client_dim": CLIENT_FEATURE_DIM_ENHANCED,
        "history_dim": HISTORY_FEATURE_DIM_ENHANCED,
        "global_base_dim": BASE_GLOBAL_FEATURE_DIM_ENHANCED,
    },
}


def get_prl_state_specs(
    num_clients_per_round: int,
    history_size: int,
    context_variant: str = "basic",
) -> Dict[str, int]:
    """Return state dimension specs for the PPO aggregator."""
    variant = (context_variant or "basic").lower()
    if variant not in CONTEXT_VARIANTS:
        raise ValueError(f"Unknown context_variant '{context_variant}'.")
    spec = CONTEXT_VARIANTS[variant]
    global_dim = spec["global_base_dim"] + spec["history_dim"] * history_size
    state_dim = num_clients_per_round * spec["client_dim"] + global_dim
    return {
        "variant": variant,
        "client_feature_dim": spec["client_dim"],
        "history_feature_dim": spec["history_dim"],
        "global_feature_dim": global_dim,
        "state_dim": state_dim,
    }


def _empty_eval_stats() -> Dict[str, float]:
    return {
        "recall_mean": 0.0,
        "recall_std": 0.0,
        "recall_min": 0.0,
        "recall_max": 0.0,
    }


def _default_history_entry() -> Tuple[float, float, float, float]:
    return (0.0, 0.5, 0.5, 0.0)


class RunningMeanStd:
    """Online mean/std tracker for reward normalization."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def _std(self) -> float:
        if self.count <= 1.0:
            return 1.0
        return math.sqrt(max(self.var / (self.count - 1.0), 1e-6))

    def normalize(self, value: float) -> float:
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        self.var += delta * (value - self.mean)
        std = self._std()
        return (value - self.mean) / std


# ============================== Environment ==============================

class AggregationEnvPPO:
    """
    Environment for federated aggregation with PPO.
    
    Key differences from SAC version:
    - State includes explicit client features (struggle, accuracy, samples)
    - Reward is computed as advantage over baseline (mean aggregation)
    - Tracks baseline performance for comparison
    """
    
    def __init__(
        self,
        eval_fn,
        build_class_prototypes,
        num_classes,
        num_clients_per_round,
        device,
        baseline_type: str = "mean",  # "mean" or "fedavg"
        reward_case: str = "acc_align",
        use_advantage: bool = True,
        kl_weight: float = 0.01,
        verbose: bool = False,
    delta_scale: float = 0.5,
    adaptive_delta: bool = True,
    delta_scale_min: float = 0.1,
    delta_scale_max: float = 0.9,
    delta_scale_gain: float = 0.05,
    delta_scale_margin: float = 0.002,
    history_size: int = 5,
        context_variant: str = "basic",
        enable_safety: bool = True,
        safety_tolerance: float = 0.005,
        safety_patience: int = 2,
        safety_cooldown_rounds: int = 1,
        safety_penalty: float = 0.05,
    ):
        self.device = device
        self.num_clients_per_round = num_clients_per_round
        self.num_classes = num_classes
        self.evaluate_fn = eval_fn
        self.build_class_prototypes = build_class_prototypes
        self.baseline_type = baseline_type
        self.reward_case = reward_case
        self.use_advantage = use_advantage
        self.kl_weight = kl_weight
        self.verbose = verbose
        self.delta_scale = delta_scale
        self.adaptive_delta = bool(adaptive_delta)
        self.history_size = max(1, history_size)
        self.global_history = deque(maxlen=self.history_size)
        self.reward_normalizer = RunningMeanStd()
        self.prev_client_accuracies: Dict[int, float] = {}
        self.context_variant = (context_variant or "basic").lower()
        if self.context_variant not in CONTEXT_VARIANTS:
            raise ValueError(f"Unknown context_variant '{context_variant}'.")
        self.variant_spec = CONTEXT_VARIANTS[self.context_variant]
        ds_min = max(1e-3, float(delta_scale_min))
        ds_max = max(ds_min, float(delta_scale_max))
        self.delta_scale_min = ds_min if ds_min <= ds_max else ds_max
        self.delta_scale_max = ds_max if ds_max >= ds_min else ds_min
        self.delta_scale_gain = max(1e-4, float(delta_scale_gain))
        self.delta_scale_margin = max(0.0, float(delta_scale_margin))
        init_delta = float(delta_scale)
        if not math.isfinite(init_delta):
            init_delta = 0.5 * (self.delta_scale_min + self.delta_scale_max)
        self.delta_scale_init = float(
            max(self.delta_scale_min, min(self.delta_scale_max, init_delta))
        )
        if not self.adaptive_delta:
            self.delta_scale_min = self.delta_scale_max = self.delta_scale_init
        self.current_delta_scale = self.delta_scale_init
        self.enable_safety = bool(enable_safety)
        self.safety_tolerance = max(0.0, float(safety_tolerance))
        self.safety_patience = max(1, int(safety_patience))
        self.safety_cooldown_rounds = max(0, int(safety_cooldown_rounds))
        self.safety_penalty = max(0.0, float(safety_penalty))
        
        # Tracking
        self.highest_accuracy = -100.0
        self.global_accuracy_ema = None  # Exponential moving average
        self.ema_alpha = 0.1
        self.last_alignment = 0.0
        self.last_eval_stats = _empty_eval_stats()
        
        self.reset_internal()
        
    def reset_internal(self):
        """Reset episode-specific variables"""
        self.prev_client_accuracies = getattr(self, "client_accuracies", {}).copy()
        self.previous_global_accuracy = getattr(self, "global_accuracy", None)
        self.global_parameters = None
        self.global_accuracy = None
        self.baseline_accuracy = None
        self.current_global_parameters = None
        self.clients_parameters_dict = OrderedDict()
        self.client_features = None  # [K, F] tensor of per-client features
        self.client_num_samples = {}
        self.client_struggle_scores = {}
        self.client_accuracies = {}
        self.weights_vector = None
        self.current_accuracy = None
        self.is_done = False
        self.best_accuracy = None
        self.best_params = None
        self.best_action = None
        self.client_prototypes = None
        self.global_proto = None
        self.last_fairness = 0.0
        self.last_alignment = 0.0
        self.last_eval_stats = _empty_eval_stats()
        self.baseline_params = None
        self.baseline_weights_tensor = None
        self.baseline_client_ids: List = []
        self.baseline_alignment = 0.0
        self._reset_safety_state()

    def _reset_safety_state(self):
        self.safety_fail_streak = 0
        self.safety_cooldown_left = 0
        self.current_delta_scale = self.delta_scale_init

    @torch.no_grad()
    def aggregate(self, client_solutions: List[tuple]):
        """Weighted aggregation of client models"""
        total_weight = 0.0
        model_state_dict: OrderedDict = client_solutions[0][1]
        base = [torch.zeros_like(param) for param in model_state_dict.values()]
        
        for weight, client_state_dict in client_solutions:
            total_weight += weight
            for i, param in enumerate(client_state_dict.values()):
                base_dtype = base[i].dtype
                base[i] += (weight * param).to(base_dtype)
        
        averaged_params = [param / (total_weight + 1e-12) for param in base]
        return OrderedDict(zip(model_state_dict.keys(), averaged_params))

    def _build_baseline_weights(self, client_ids: List):
        """
        Build baseline weights for comparison.
        
        Returns:
            torch.Tensor: baseline weights [K]
        """
        K = len(client_ids)
        
        if self.baseline_type == "mean":
            # Equal weights
            return torch.ones(K, device=self.device) / K
        
        elif self.baseline_type == "fedavg":
            # Sample-size proportional
            weights = torch.tensor(
                [self.client_num_samples[cid] for cid in client_ids],
                dtype=torch.float32,
                device=self.device
            )
            return weights / weights.sum()
        
        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Take a step with RL-chosen weights.
        
        Args:
            action: probability distribution over clients [K]
        
        Returns:
            next_state: [1, num_features]
            reward: scalar (advantage over baseline if use_advantage=True)
            done: bool
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)
        
        # Normalize to simplex (safety)
        action = action.clamp(min=1e-6)
        action = action / action.sum()

        if self.enable_safety and self.safety_cooldown_left > 0:
            self.safety_cooldown_left -= 1
            return self._baseline_transition(reward_override=0.0)
        
        client_ids = list(self.weights_vector.keys())
        baseline_weights = self._build_baseline_weights(client_ids)
        adjusted_action = self._blend_with_baseline(action, baseline_weights)
        
        # 1. Aggregate with RL weights
        for i, client in enumerate(client_ids):
            self.weights_vector[client] = adjusted_action[i].item()
        
        client_solutions = [
            (self.weights_vector[client], self.clients_parameters_dict[client])
            for client in client_ids
        ]
        self.current_global_parameters = self.aggregate(client_solutions)
        
        # 2. Evaluate RL aggregation
        new_accuracy, new_eval_mx = self.evaluate_fn(
            model_params=self.current_global_parameters,
            render=False
        )

        if self.enable_safety and self._maybe_trigger_guard(new_accuracy):
            return self._baseline_transition(reward_override=-self.safety_penalty)
        
        # 3. Build prototypes for RL model
        global_proto, _ = self.build_class_prototypes(
            model_params=self.current_global_parameters
        )
        alignment_score = _proto_alignment_score(global_proto, self.client_prototypes)
        self.last_alignment = alignment_score
        
        # 4. Compute RL reward
        reward_rl = compute_reward(
            new_accuracy=new_accuracy,
            new_eval_mx=new_eval_mx,
            current_accuracy=self.current_accuracy,
            global_proto=global_proto,
            client_proto_avg=self.client_prototypes,
            reward_case=self.reward_case,
        )
        
        # 5. Compute baseline reward (if using advantage)
        if self.use_advantage and self.baseline_accuracy is not None:
            reward_baseline = compute_reward(
                new_accuracy=self.baseline_accuracy,
                new_eval_mx=self.baseline_eval_mx,
                current_accuracy=self.current_accuracy,
                global_proto=self.baseline_proto,
                client_proto_avg=self.client_prototypes,
                reward_case=self.reward_case,
            )
            # Advantage-style reward
            reward = reward_rl - reward_baseline
        else:
            reward = reward_rl
        
        # 6. Add KL penalty to discourage extreme deviations from baseline
        if self.kl_weight > 0:
            kl_div = F.kl_div(
                torch.log(adjusted_action + 1e-8),
                baseline_weights,
                reduction='sum'
            )
            reward = reward - self.kl_weight * float(kl_div.item())
        
        # 7. Update tracking
        self.current_accuracy = new_accuracy
        fairness_score = self._compute_fairness_score(new_eval_mx)
        self.last_fairness = fairness_score
        self.last_eval_stats = self._extract_eval_stats(new_eval_mx)
        baseline_gap = new_accuracy - (self.baseline_accuracy if self.baseline_accuracy else 0.0)
        self._update_history(new_accuracy, fairness_score, alignment_score, baseline_gap)
        self._update_delta_scale(new_accuracy)
        
        # Update EMA
        if self.global_accuracy_ema is None:
            self.global_accuracy_ema = new_accuracy
        else:
            self.global_accuracy_ema = (
                self.ema_alpha * new_accuracy +
                (1 - self.ema_alpha) * self.global_accuracy_ema
            )
        
        # Track best
        if (self.best_accuracy is None) or (self.best_accuracy < new_accuracy):
            if self.verbose:
                print(f"New best accuracy: {new_accuracy:.4f} (reward: {reward:.4f})")
            self.best_accuracy = new_accuracy
            self.best_action = action.clone()
            self.best_params = OrderedDict({
                k: v.clone() for k, v in self.current_global_parameters.items()
            })
        
        if self.highest_accuracy < new_accuracy:
            self.highest_accuracy = new_accuracy
        
        # Done condition
        base_ref = self.global_accuracy if self.global_accuracy else new_accuracy
        self.is_done = new_accuracy > min(0.9999, base_ref * 1.5)

        # Normalize reward for stable PPO updates
        normalized_reward = self.reward_normalizer.normalize(float(reward))
        reward = float(torch.tanh(torch.tensor(normalized_reward)).item())
        
        # Build next state (global features)
        next_state = self._build_global_state(new_accuracy, new_eval_mx)
        
        return next_state, reward, new_accuracy, self.is_done

    def _build_client_features(
        self,
        client_ids: List,
        client_accuracies: List[float],
        client_struggle_scores: Dict[int, float],
        client_num_samples: Dict[int, int],
        baseline_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-client feature matrix for the configured context variant."""
        if self.context_variant == "enhanced":
            return self._build_client_features_enhanced(
                client_ids,
                client_accuracies,
                client_struggle_scores,
                client_num_samples,
                baseline_weights,
            )
        return self._build_client_features_basic(
            client_ids,
            client_accuracies,
            client_struggle_scores,
            client_num_samples,
        )

    def _build_client_features_basic(
        self,
        client_ids: List,
        client_accuracies: List[float],
        client_struggle_scores: Dict[int, float],
        client_num_samples: Dict[int, int],
    ) -> torch.Tensor:
        features = []
        total_samples = max(1, sum(client_num_samples.values()))
        struggles = np.array([
            client_struggle_scores.get(cid, 0.0) for cid in client_ids
        ], dtype=np.float32)
        struggle_mean = float(np.mean(struggles)) if len(struggles) else 0.0
        struggle_std = float(np.std(struggles)) + 1e-6
        prev_accs = getattr(self, 'prev_client_accuracies', {})
        
        for i, cid in enumerate(client_ids):
            acc = client_accuracies[i]
            prev_acc = prev_accs.get(cid, acc)
            acc_delta = acc - prev_acc
            struggle = client_struggle_scores.get(cid, 0.0)
            struggle_norm = (struggle - struggle_mean) / struggle_std
            num_samples = client_num_samples.get(cid, 1)
            log_sample_ratio = np.log((num_samples / total_samples) + 1e-8)
            features.append([acc, acc_delta, struggle_norm, log_sample_ratio])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _build_client_features_enhanced(
        self,
        client_ids: List,
        client_accuracies: List[float],
        client_struggle_scores: Dict[int, float],
        client_num_samples: Dict[int, int],
        baseline_weights: torch.Tensor,
    ) -> torch.Tensor:
        features = []
        total_samples = max(1, sum(client_num_samples.values()))
        struggles = np.array([
            client_struggle_scores.get(cid, 0.0) for cid in client_ids
        ], dtype=np.float32)
        struggle_mean = float(np.mean(struggles)) if len(struggles) else 0.0
        struggle_std = float(np.std(struggles)) + 1e-6
        struggle_ranks = np.argsort(np.argsort(struggles)) if len(struggles) else np.array([])
        rank_den = max(len(client_ids) - 1, 1)
        prev_accs = getattr(self, 'prev_client_accuracies', {})
        baseline_np = (
            baseline_weights.detach().cpu().numpy()
            if baseline_weights.numel()
            else np.zeros(len(client_ids), dtype=np.float32)
        )
        
        for i, cid in enumerate(client_ids):
            acc = client_accuracies[i]
            prev_acc = prev_accs.get(cid, acc)
            acc_delta = acc - prev_acc
            struggle = client_struggle_scores.get(cid, 0.0)
            struggle_norm = (struggle - struggle_mean) / struggle_std
            rank_norm = float(struggle_ranks[i] / rank_den) if len(struggles) else 0.0
            num_samples = client_num_samples.get(cid, 1)
            sample_ratio = float(num_samples / total_samples)
            log_sample_ratio = np.log(sample_ratio + 1e-8)
            baseline_weight = float(baseline_np[i]) if len(baseline_np) > i else 0.0
            features.append([
                acc,
                prev_acc,
                acc_delta,
                struggle_norm,
                rank_norm,
                log_sample_ratio,
                sample_ratio,
                baseline_weight,
            ])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _build_global_state(
        self,
        global_acc: float,
        eval_mx: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_variant == "enhanced":
            return self._build_global_state_enhanced(global_acc, eval_mx)
        return self._build_global_state_basic(global_acc)

    def _build_global_state_basic(self, global_acc: float) -> torch.Tensor:
        client_feats_flat = self.client_features.flatten()
        baseline_acc = self.baseline_accuracy if self.baseline_accuracy else 0.0
        ema_acc = self.global_accuracy_ema if self.global_accuracy_ema else 0.0
        prev_acc = self.previous_global_accuracy if self.previous_global_accuracy else 0.0
        global_feats = torch.tensor([
            global_acc,
            baseline_acc,
            ema_acc,
            global_acc - prev_acc,
            self.last_fairness,
        ], dtype=torch.float32, device=self.device)
        history_tensor = self._build_history_tensor_basic()
        return torch.cat([client_feats_flat, global_feats, history_tensor]).unsqueeze(0)

    def _build_global_state_enhanced(
        self,
        global_acc: float,
        eval_mx: torch.Tensor,
    ) -> torch.Tensor:
        client_feats_flat = self.client_features.flatten()
        baseline_acc = self.baseline_accuracy if self.baseline_accuracy else 0.0
        ema_acc = self.global_accuracy_ema if self.global_accuracy_ema else 0.0
        prev_acc = self.previous_global_accuracy if self.previous_global_accuracy else 0.0
        baseline_gap = global_acc - baseline_acc
        stats = self.last_eval_stats or _empty_eval_stats()
        global_feats = torch.tensor([
            global_acc,
            baseline_acc,
            ema_acc,
            global_acc - prev_acc,
            self.last_fairness,
            self.last_alignment,
            baseline_gap,
            stats.get('recall_mean', 0.0),
            stats.get('recall_std', 0.0),
            stats.get('recall_min', 0.0),
            stats.get('recall_max', 0.0),
        ], dtype=torch.float32, device=self.device)
        history_tensor = self._build_history_tensor_enhanced()
        return torch.cat([client_feats_flat, global_feats, history_tensor]).unsqueeze(0)

    def _build_history_tensor_basic(self) -> torch.Tensor:
        history = list(self.global_history)
        values = []
        for entry in history:
            if isinstance(entry, (list, tuple)):
                values.append(float(entry[0]))
            else:
                values.append(float(entry))
        if len(values) < self.history_size:
            padding = [0.0] * (self.history_size - len(values))
            values = padding + values
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _build_history_tensor_enhanced(self) -> torch.Tensor:
        history = list(self.global_history)
        normalized_history = []
        for entry in history:
            if isinstance(entry, (list, tuple)) and len(entry) == HISTORY_FEATURE_DIM_ENHANCED:
                normalized_history.append(tuple(float(v) for v in entry))
            else:
                normalized_history.append(_default_history_entry())
        if len(normalized_history) < self.history_size:
            padding = [
                _default_history_entry() for _ in range(self.history_size - len(normalized_history))
            ]
            normalized_history = padding + normalized_history
        return torch.tensor(
            normalized_history,
            dtype=torch.float32,
            device=self.device,
        ).flatten()

    def _blend_with_baseline(
        self,
        action: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Blend policy action with baseline weights using delta scaling."""
        delta = action - baseline
        scale = torch.tensor(self.current_delta_scale, device=action.device, dtype=action.dtype)
        adjusted = baseline + scale * delta
        adjusted = adjusted.clamp(min=1e-6)
        return adjusted / adjusted.sum()

    def _update_history(
        self,
        new_accuracy: float,
        fairness: float,
        alignment: float,
        baseline_gap: float,
    ):
        if not math.isfinite(new_accuracy):
            return
        history_entry = (
            float(new_accuracy),
            float(fairness),
            float(alignment),
            float(baseline_gap),
        )
        self.global_history.append(history_entry)

    def _baseline_transition(self, reward_override: float = 0.0):
        if self.baseline_params is not None:
            self.current_global_parameters = OrderedDict({
                k: v.clone() for k, v in self.baseline_params.items()
            })
        self.current_accuracy = self.baseline_accuracy
        fairness = self._compute_fairness_score(self.baseline_eval_mx)
        self.last_fairness = fairness
        self.last_alignment = getattr(self, "baseline_alignment", self.last_alignment)
        self.last_eval_stats = self._extract_eval_stats(self.baseline_eval_mx)
        self._update_history(
            self.baseline_accuracy,
            fairness,
            self.last_alignment,
            0.0,
        )
        if self.baseline_weights_tensor is not None:
            for i, cid in enumerate(self.baseline_client_ids):
                self.weights_vector[cid] = float(self.baseline_weights_tensor[i].item())
        if self.adaptive_delta:
            self.current_delta_scale = max(
                self.delta_scale_min,
                self.current_delta_scale - self.delta_scale_gain,
            )
        next_state = self._build_global_state(
            self.baseline_accuracy,
            self.baseline_eval_mx,
        )
        return next_state, reward_override, self.is_done

    def _maybe_trigger_guard(self, new_accuracy: float) -> bool:
        if not self.enable_safety or self.baseline_accuracy is None:
            return False
        baseline = self.baseline_accuracy
        drop = baseline - new_accuracy
        if drop <= self.safety_tolerance:
            self.safety_fail_streak = 0
            return False
        self.safety_fail_streak += 1
        if self.safety_fail_streak < self.safety_patience:
            return False
        self.safety_fail_streak = 0
        if self.safety_cooldown_rounds > 0:
            self.safety_cooldown_left = self.safety_cooldown_rounds
        return True

    def _update_delta_scale(self, new_accuracy: float):
        if not self.adaptive_delta or self.baseline_accuracy is None:
            return
        gap = new_accuracy - self.baseline_accuracy
        if gap > self.delta_scale_margin:
            self.current_delta_scale = min(
                self.delta_scale_max,
                self.current_delta_scale + self.delta_scale_gain,
            )
        elif gap < -self.delta_scale_margin:
            self.current_delta_scale = max(
                self.delta_scale_min,
                self.current_delta_scale - self.delta_scale_gain,
            )

    def _compute_fairness_score(self, eval_mx) -> float:
        try:
            recalls = _per_class_recall_from_confmat(eval_mx)
            if recalls.numel() > 1:
                return float(1.0 - torch.std(recalls).clamp_max(1.0).item())
        except Exception:
            pass
        return 0.5

    def _extract_eval_stats(self, eval_mx) -> Dict[str, float]:
        stats = _empty_eval_stats()
        try:
            recalls = _per_class_recall_from_confmat(eval_mx)
            if recalls.numel() > 0:
                stats['recall_mean'] = float(recalls.mean().item())
                stats['recall_std'] = float(recalls.std(unbiased=False).item())
                stats['recall_min'] = float(recalls.min().item())
                stats['recall_max'] = float(recalls.max().item())
        except Exception:
            pass
        return stats

    def reset(
        self,
        parameters_dict: Dict,
        client_metadata: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Reset environment with new client models and metadata.
        
        Args:
            parameters_dict: {client_id: model_state_dict}
            client_metadata: {
                client_id: {
                    'num_samples': int,
                    'struggle_score': float,
                }
            }
        
        Returns:
            initial_state: [1, D]
            done: bool
        """
        prev_client_acc = getattr(self, "client_accuracies", {}).copy()
        prev_global_acc = getattr(self, "global_accuracy", None)
        history_snapshot = deque(self.global_history, maxlen=self.history_size)
        self.reset_internal()
        self.prev_client_accuracies = prev_client_acc
        self.previous_global_accuracy = prev_global_acc
        self.global_history = history_snapshot
        
        client_ids = list(parameters_dict.keys())
        if all(isinstance(cid, (int, np.integer)) for cid in client_ids):
            client_ids = sorted(client_ids)
        else:
            client_ids = sorted(client_ids, key=lambda cid: str(cid))
        client_accuracies = []
        
        # Extract metadata
        if client_metadata is None:
            client_metadata = {}
            raise ValueError("Client metadata must be provided for PPO aggregation.")
        
        for cid in client_ids:
            meta = client_metadata.get(cid, {})
            self.client_num_samples[cid] = meta.get('num_samples', 1)
            self.client_struggle_scores[cid] = meta.get('struggle_score', 0.0)
        
        # Build client prototypes (average across clients)
        global_proto_sums = {}
        global_counts = {}
        
        for cid in client_ids:
            param = parameters_dict[cid]
            # Evaluate client
            acc, eval_matrix = self.evaluate_fn(
                model_params=param,
                num_batches=65,
                name=f'client({cid}): ',
                render=False
            )
            client_accuracies.append(acc)
            self.client_accuracies[cid] = acc
            
            # Client prototypes
            client_proto, proto_count = self.build_class_prototypes(model_params=param)
            for class_id, proto in client_proto.items():
                if class_id not in global_proto_sums:
                    global_proto_sums[class_id] = proto * proto_count[class_id]
                    global_counts[class_id] = proto_count[class_id]
                else:
                    global_proto_sums[class_id] += proto * proto_count[class_id]
                    global_counts[class_id] += proto_count[class_id]
        
        self.client_prototypes = {
            cid: (global_proto_sums[cid] / global_counts[cid])
            for cid in global_proto_sums.keys()
            if global_counts[cid] > 0
        }
        
        baseline_weights_tensor = self._build_baseline_weights(client_ids)
        
        # Build client feature matrix
        self.client_features = self._build_client_features(
            client_ids,
            client_accuracies,
            self.client_struggle_scores,
            self.client_num_samples,
            baseline_weights_tensor,
        )
        
        if self.verbose:
            print('Client Accuracies:', ', '.join(f'{acc:.3f}' for acc in client_accuracies))
            print('Client Struggle Scores:', ', '.join(
                f'{self.client_struggle_scores.get(cid, 0.0):.3f}' for cid in client_ids
            ))
        
        self.clients_parameters_dict = OrderedDict((cid, parameters_dict[cid]) for cid in client_ids)
        
        # 1. Evaluate baseline aggregation (mean or FedAvg)
        self.weights_vector = {
            cid: float(baseline_weights_tensor[i].item())
            for i, cid in enumerate(client_ids)
        }
        
        baseline_solutions = [
            (self.weights_vector[cid], parameters_dict[cid])
            for cid in client_ids
        ]
        baseline_params = self.aggregate(baseline_solutions)
        self.baseline_accuracy, self.baseline_eval_mx = self.evaluate_fn(
            model_params=baseline_params,
            render=False
        )
        self.baseline_proto, _ = self.build_class_prototypes(model_params=baseline_params)
        baseline_alignment = _proto_alignment_score(self.baseline_proto, self.client_prototypes)
        self.last_alignment = baseline_alignment
        self.baseline_alignment = baseline_alignment
        self.last_eval_stats = self._extract_eval_stats(self.baseline_eval_mx)
        self.baseline_params = OrderedDict({
            k: v.clone() for k, v in baseline_params.items()
        })
        self.baseline_weights_tensor = baseline_weights_tensor.clone()
        self.baseline_client_ids = list(client_ids)
        
        if self.verbose:
            print(f"Baseline ({self.baseline_type}) accuracy: {self.baseline_accuracy:.4f}")
        
        # 2. Initialize current/global with baseline
        self.current_global_parameters = OrderedDict({
            k: v.clone() for k, v in baseline_params.items()
        })
        self.global_parameters = OrderedDict({
            k: v.clone() for k, v in baseline_params.items()
        })
        self.current_accuracy = self.baseline_accuracy
        self.global_accuracy = self.baseline_accuracy
        self.best_accuracy = self.baseline_accuracy
        self.best_params = OrderedDict({
            k: v.clone() for k, v in baseline_params.items()
        })
        self.best_action = baseline_weights_tensor.clone()
        self.last_fairness = self._compute_fairness_score(self.baseline_eval_mx)
        self._update_history(
            self.baseline_accuracy,
            self.last_fairness,
            baseline_alignment,
            0.0,
        )
        
        if self.highest_accuracy < self.baseline_accuracy:
            self.highest_accuracy = self.baseline_accuracy
        
        # Done condition
        self.is_done = self.global_accuracy > 0.9999
        
        # Build initial state
        initial_state = self._build_global_state(
            self.baseline_accuracy,
            self.baseline_eval_mx
        )
        
        return initial_state, self.is_done


# ============================== PPO Networks ==============================

class PPOPolicy(nn.Module):
    """
    PPO policy network that outputs logits for a categorical distribution over client weights.
    
    Architecture: MLP on flattened state → logits [K]
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_dim)
        
        # Initialize with small weights to start near uniform
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.constant_(self.logits.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, D]
        Returns:
            logits: [B, K]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits
    
    def get_action_and_logprob(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            state: [B, D]
            action: [B, K] optional; if provided, compute log prob for this action
        
        Returns:
            action: [B, K] sampled weights (softmax)
            log_prob: [B] log probability of action
            entropy: [B] entropy of distribution
        """
        logits = self.forward(state)  # [B, K]
        probs = F.softmax(logits, dim=-1)  # [B, K]
        
        # Sample or use provided action
        if action is None:
            # Sample from categorical, then convert to one-hot-like weights via softmax
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()  # [B]
            # For continuous weights, we use Gumbel-Softmax or just the probs directly
            # Here we'll use the softmax probabilities as continuous weights
            action = probs
            log_prob = torch.log(probs + 1e-8).sum(dim=-1)  # Not exactly right for continuous
            
            # Better: use explicit entropy and log_prob from categorical over a relaxed continuous action
            # For simplicity, treat action as the softmax probabilities
            # Log prob = sum of log(probs) weighted by action (which equals probs here)
            log_prob = (action * torch.log(probs + 1e-8)).sum(dim=-1)  # [B]
        else:
            # Compute log prob of given action (weights)
            # Treat as if action is a sample from the softmax
            log_prob = (action * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B]
        
        return action, log_prob, entropy


class PPOValue(nn.Module):
    """
    Value network for PPO (critic).
    
    Estimates V(s): expected return from state s.
    """
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, D]
        Returns:
            value: [B, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


# ============================== Rollout Buffer ==============================

class RolloutBuffer:
    """
    Buffer for storing trajectories for PPO update.
    """
    
    def __init__(self, device):
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        """Get all data as tensors"""
        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        log_probs = torch.cat(self.log_probs).to(self.device)
        values = torch.cat(self.values).to(self.device).squeeze(-1)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, log_probs, values, dones
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.rewards)


# ============================== PPO Agent ==============================

class PPOAgentR:
    """
    PPO agent for federated aggregation with:
    - Advantage over baseline (mean/FedAvg)
    - KL penalty for regularization
    - Entropy bonus for exploration
    - Generalized Advantage Estimation (GAE)
    """
    
    def __init__(
        self,
        eval_fn,
        build_class_prototypes,
        num_classes: int,
        device,
        num_clients_per_round: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        baseline_type: str = "mean",
        reward_case: str = "acc_align",
        use_advantage: bool = True,
        kl_weight: float = 0.01,
        delta_scale: float = 0.5,
    adaptive_delta: bool = True,
    delta_scale_min: float = 0.1,
    delta_scale_max: float = 0.9,
    delta_scale_gain: float = 0.05,
    delta_scale_margin: float = 0.002,
        history_size: int = 5,
        context_variant: str = "basic",
        enable_safety: bool = True,
        safety_tolerance: float = 0.005,
        safety_patience: int = 2,
        safety_cooldown_rounds: int = 1,
        safety_penalty: float = 0.05,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Environment
        self.context_variant = (context_variant or "basic").lower()
        self.env = AggregationEnvPPO(
            eval_fn,
            build_class_prototypes,
            num_classes,
            num_clients_per_round,
            device,
            baseline_type=baseline_type,
            reward_case=reward_case,
            use_advantage=use_advantage,
            kl_weight=kl_weight,
            delta_scale=delta_scale,
            adaptive_delta=adaptive_delta,
            delta_scale_min=delta_scale_min,
            delta_scale_max=delta_scale_max,
            delta_scale_gain=delta_scale_gain,
            delta_scale_margin=delta_scale_margin,
            history_size=history_size,
            context_variant=self.context_variant,
            enable_safety=enable_safety,
            safety_tolerance=safety_tolerance,
            safety_patience=safety_patience,
            safety_cooldown_rounds=safety_cooldown_rounds,
            safety_penalty=safety_penalty,
        )
        
        # Networks
        self.policy = PPOPolicy(state_dim, action_dim, hidden_size).to(device)
        self.value = PPOValue(state_dim, hidden_size).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Buffer
        self.buffer = RolloutBuffer(device)
        
        self.action_dim = action_dim
    
    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for data collection.
        
        Returns:
            action: [K] weights
            log_prob: [1]
            value: [1]
        """
        state = state.to(self.device)
        action, log_prob, _ = self.policy.get_action_and_logprob(state)
        value = self.value(state)
        
        return action[0], log_prob, value
    
    @torch.no_grad()
    def eval_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation (use mode/mean of policy).
        
        Returns:
            action: [K]
        """
        state = state.to(self.device)
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        return probs[0]
    
    def step(self, action: torch.Tensor):
        """Take environment step"""
        return self.env.step(action)
    
    def reset(self, parameters_dict: Dict, client_metadata: Optional[Dict] = None):
        """Reset environment"""
        return self.env.reset(parameters_dict, client_metadata)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: [T]
            values: [T]
            dones: [T]
            next_value: [1]
        
        Returns:
            advantages: [T]
            returns: [T]
        """
        rewards = rewards.view(-1)
        values = values.view(-1)
        dones = dones.view(-1)
        next_value = next_value.view(-1)
        if next_value.numel() == 0:
            next_value = torch.zeros(1, device=self.device, dtype=values.dtype)
        elif next_value.numel() > 1:
            next_value = next_value[:1]
        else:
            next_value = next_value.to(device=self.device, dtype=values.dtype)

        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Append next_value for bootstrapping
        values_ext = torch.cat([values, next_value])
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values_ext[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, n_epochs: int = 10, batch_size: int = 64):
        """
        PPO update using collected rollouts.
        
        Args:
            n_epochs: number of optimization epochs
            batch_size: mini-batch size
        """
        if len(self.buffer) == 0:
            return {}
        
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()
        
        # Compute GAE
        with torch.no_grad():
            # Bootstrap with last state value if not done
            if dones[-1] == 0:
                next_value = self.value(states[-1].unsqueeze(0))
            else:
                next_value = torch.zeros(1, 1, device=self.device)
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            # Generate random indices
            indices = torch.randperm(len(states), device=self.device)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                _, new_log_probs, entropy = self.policy.get_action_and_logprob(
                    batch_states, batch_actions
                )
                new_values = self.value(batch_states).squeeze(-1)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }


# ============================== Reward Function ==============================

def _per_class_recall_from_confmat(confmat: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class recall from confusion matrix.
    """
    if isinstance(confmat, (list, tuple)):
        confmat = torch.tensor(confmat, dtype=torch.float32)
    elif not torch.is_tensor(confmat):
        confmat = torch.as_tensor(confmat, dtype=torch.float32)
    
    tp = torch.diag(confmat)
    per_class_total = confmat.sum(dim=1).clamp_min(1e-6)
    recall = (tp / per_class_total).to(torch.float32)
    return recall


def _proto_alignment_score(global_proto_dict, client_proto_dict):
    """
    Mean cosine similarity across classes.
    Returns scalar in [0, 1].
    """
    common = sorted(set(global_proto_dict.keys()) & set(client_proto_dict.keys()))
    if len(common) == 0:
        return 0.5
    
    sims = []
    for c in common:
        if c not in global_proto_dict or c not in client_proto_dict:
            continue
        g = global_proto_dict[c]
        p = client_proto_dict[c]
        
        if g.dim() > 1:
            g = g.view(-1)
        if p.dim() > 1:
            p = p.view(-1)
        
        dev = g.device if g.is_cuda else p.device
        g = g.to(dev).float()
        p = p.to(dev).float()
        
        cos = F.cosine_similarity(g.unsqueeze(0), p.unsqueeze(0), dim=1).item()
        sims.append(cos)
    
    mean_cos = float(sum(sims) / len(sims))
    return 0.5 * (mean_cos + 1.0)


def compute_reward(
    new_accuracy: float,
    new_eval_mx,
    current_accuracy: float | None,
    global_proto: dict,
    client_proto_avg: dict,
    w_acc: float = 1.0,
    w_align: float = 0.5,
    w_fair: float = 0.5,
    reward_case: str = "acc_align",
):
    """
    Compute reward with smoothed accuracy improvement, alignment, and fairness.
    
    Much smoother than original version for stable PPO training.
    """
    # Accuracy improvement (clipped and scaled)
    if current_accuracy is None:
        current_accuracy = 0.0
    
    acc_improvement = new_accuracy - current_accuracy
    # Smoother scaling: clip to reasonable range
    acc_score = float(torch.clamp(torch.tensor(acc_improvement), -0.1, 0.1).item())
    acc_score = acc_score * 10.0  # Scale to roughly [-1, 1]
    
    # Alignment
    align = _proto_alignment_score(global_proto, client_proto_avg)
    
    # Fairness
    try:
        recalls = _per_class_recall_from_confmat(new_eval_mx)
        if recalls.numel() > 1:
            fairness = float(1.0 - torch.std(recalls).clamp_max(1.0).item())
        else:
            fairness = 0.5
    except Exception:
        fairness = 0.5
    
    # Combine
    case_key = (reward_case or "acc_align").lower()
    
    if case_key in {"acc_align_fair", "all", "full"}:
        reward = w_acc * acc_score + w_align * align + w_fair * fairness
    elif case_key in {"acc_align", "acc_align_only"}:
        reward = w_acc * acc_score + w_align * align
    elif case_key in {"acc_fair", "acc_fairness"}:
        reward = w_acc * acc_score + w_fair * fairness
    elif case_key in {"align_fair", "align_fairness"}:
        reward = w_align * align + w_fair * fairness
    elif case_key in {"align", "alignment"}:
        reward = align
    elif case_key in {"fair", "fairness"}:
        reward = fairness
    elif case_key in {"acc", "accuracy", "acc_score"}:
        reward = acc_score
    elif case_key in {"align_plus_fair", "alignfair", "align_plus_fairness"}:
        reward = align + fairness
    else:
        raise ValueError(f"Unknown reward_case '{reward_case}'.")
    
    return float(reward)
