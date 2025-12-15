from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import trange

from flearn.trainers.fedavg import FedAvgServer
from flearn.strategy.hybrid_controller import (
    ClientFeatures,
    OracleLabeler,
    Strategy,
    StrategyController,
    DRLController,
)


class HybridStrategyServer(FedAvgServer):
    """Hybrid federated learning server with dynamic per-client strategy selection.

    This trainer keeps the FedAvg skeleton but augments it with:
      - privacy-preserving client statistics
      - unsupervised client clustering
      - self-supervised counterfactual oracle for pseudo-labeling
      - lightweight MLP controller to decide GENERALIZE/PERSONALIZE/HYBRID
    """

    def __init__(self, params):
        params = params.copy()
        params.setdefault("agg", "fedavg")
        params.setdefault("trainer", "hybrid")
        if "controller_type" not in params or params.get("controller_type") is None:
            try:
                user_choice = input("Enter controller type (mlp/drl) [mlp]: ").strip().lower()
                params["controller_type"] = user_choice if user_choice else "mlp"
            except Exception:
                params["controller_type"] = "mlp"
        self.hybrid_alpha = params.get("hybrid_alpha", 0.5)
        self.dp_noise_std = params.get("dp_noise_std", 0.0)
        self.basis_rank = int(params.get("basis_rank", 5))
        self.basis_memory = int(params.get("basis_memory", 30))
        self.warmup_rounds = int(params.get("warmup_rounds", 3))
        self.cluster_k = params.get("cluster_k", 3)
        self.use_gmm = bool(params.get("use_gmm", False))
        self.controller_train_interval = int(params.get("controller_train_interval", 5))
        self.controller_min_samples = int(params.get("controller_min_samples", 32))
        self.controller_batch_size = int(params.get("controller_batch_size", 32))
        self.controller_epochs = int(params.get("controller_epochs", 8))
        self.controller_type = params.get("controller_type", "mlp").lower()
        super().__init__(params)
        # caches for server-side personalization
        self.client_flat_params = {}
        self.client_prev_params = {}
        self.client_last_state = {}
        self.U = None
        self.global_flat = None
        self.strategy_fractions = []
        self.feature_cache = []
        self.client_feature_dict = {}
        self.client_personal_state = {}
        self.prev_global_model = deepcopy(self.latest_model)
        # RL bookkeeping for DRLController transitions
        self.rl_last_feature: Dict[int, ClientFeatures] = {}
        self.rl_last_action: Dict[int, Strategy] = {}

        feature_dim = 8  # 7 stats (incl. accuracies) + cluster id
        controller_kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=params.get("controller_hidden", 64),
            lr=params.get("controller_lr", 1e-3),
            weight_decay=params.get("controller_weight_decay", 1e-4),
            device=self.device,
        )
        if self.controller_type == "drl":
            controller_kwargs["temperature"] = params.get("controller_temperature", 1.0)
            self.controller = DRLController(**controller_kwargs)
        else:
            self.controller = StrategyController(**controller_kwargs)
        self.oracle = OracleLabeler(
            simulate_rounds=params.get("oracle_sim_rounds", 1),
            score_weights=tuple(params.get("oracle_score_weights", (0.6, 0.4, 0.1))),
            device=self.device,
        )

    # -------------------------------------------------------------
    def _aggregate(self, wsolns: List[tuple[int, OrderedDict]], base_state: OrderedDict) -> OrderedDict:
        """Type-safe aggregation that skips non-floating buffers (e.g., num_batches_tracked).

        Only floating tensors are averaged; integer/bool buffers are copied from the first client/state.
        """
        if len(wsolns) == 0:
            return base_state

        keys = list(base_state.keys())
        # Initialize accumulators
        accum: OrderedDict[str, torch.Tensor] = OrderedDict()
        is_float: Dict[str, bool] = {}
        for k in keys:
            v0 = base_state[k]
            is_float[k] = v0.is_floating_point()
            accum[k] = torch.zeros_like(v0) if is_float[k] else v0.clone()

        total_weight = 0.0
        for w, state in wsolns:
            total_weight += w
            for k in keys:
                v = state[k]
                if is_float[k]:
                    accum[k] = accum[k] + v * float(w)
                else:
                    # For non-float buffers, just keep the latest copy (no averaging needed)
                    accum[k] = v.clone()

        averaged = OrderedDict()
        denom = total_weight + 1e-12
        for k in keys:
            if is_float[k]:
                averaged[k] = accum[k] / denom
            else:
                averaged[k] = accum[k]
        return averaged

    def _head_keys(self, state: OrderedDict) -> List[str]:
        return [k for k in state.keys() if k.startswith("fc.") or k.startswith("resnet.fc.") or k.startswith("linear")]

    def _decide_strategy(self, client_id: int) -> Strategy:
        # Learned policy if controller ready
        feat = self.client_feature_dict.get(client_id, None)
        if feat is not None and self.controller.ready:
            return self.controller.predict(feat)
        # Fallback: basis availability heuristic
        if self.U is None or client_id not in self.client_flat_params:
            return Strategy.GENERALIZE
        if client_id in self.client_prev_params:
            return Strategy.HYBRID
        return Strategy.PERSONALIZE

    def _update_clusters(self):
        feats = [f for _, f in self.feature_cache]
        if len(feats) == 0:
            return
        # For DRL controller, skip clustering; just pass through features with cluster_id=-1
        if getattr(self, "controller_type", "mlp") == "drl":
            for cid, feat in self.feature_cache:
                feat.cluster_id = -1
                self.client_feature_dict[cid] = feat
            self.feature_cache.clear()
            return
        labels = self.controller.cluster(feats, k=self.cluster_k, use_gmm=self.use_gmm)
        for (cid, feat), lid in zip(self.feature_cache, labels):
            feat.cluster_id = int(lid)
            self.client_feature_dict[cid] = feat
        self.feature_cache.clear()

    def _train_controller_if_ready(self):
        self.controller.maybe_train(
            epochs=self.controller_epochs,
            batch_size=self.controller_batch_size,
            min_samples=self.controller_min_samples,
        )

    def _compute_drl_reward(self, prev_feat: ClientFeatures, next_feat: ClientFeatures) -> float:
        """Compute a per-client reward from consecutive feature states.

        The reward encourages lower loss, higher alignment, higher accuracy, and
        lower uncertainty between successive client updates.
        """

        # Improvement in local loss (smaller is better)
        loss_gain = float(prev_feat.delta_loss - next_feat.delta_loss)
        # Alignment with global model
        cos_gain = float(next_feat.cos_sim - prev_feat.cos_sim)
        # Accuracy improvements (local post-train and global pre-train surrogate)
        local_acc_gain = float(next_feat.local_val_acc - prev_feat.local_val_acc)
        global_acc_gain = float(next_feat.global_val_acc - prev_feat.global_val_acc)
        # Preference for lower uncertainty/entropy
        entropy_gain = float(prev_feat.entropy - next_feat.entropy)
        # Penalize exploding gradients
        grad_stability = -float(abs(next_feat.grad_norm - prev_feat.grad_norm))

        reward = (
            0.25 * loss_gain
            + 0.25 * cos_gain
            + 0.2 * local_acc_gain
            + 0.15 * global_acc_gain
            + 0.1 * entropy_gain
            + 0.05 * grad_stability
        )
        return float(reward)

    def _server_test_accuracy(self, state: OrderedDict) -> float:
        model = deepcopy(self.client_model).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return float(correct) / max(1, total)

    def train(self):
        print("Training with hybrid strategy controller ---")
        for rnd in trange(self.num_rounds, desc=self.desc):
            self.current_round = rnd
            # store previous global for cosine stats
            self.prev_global_model = deepcopy(self.latest_model)
            self.eval(rnd, self.set_client_model_test)
            if self.loss_converged:
                break

            selected_clients = list(
                self.select_clients(rnd, num_clients=min(self.clients_per_round, len(self.clients)))
            )
            csolns: List[tuple[int, OrderedDict]] = []
            strategy_counter: Dict[str, int] = defaultdict(int)
            self.feature_cache = []
            for client in selected_clients:
                soln, feats = client.train_and_report(
                    global_state=self.latest_model,
                    prev_global_state=self.prev_global_model,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                )
                csolns.append(soln)
                # cache client trained state for personalization
                self.client_last_state[client.id] = client.model.state_dict()
                self.feature_cache.append((client.id, feats))

            self.latest_model = self._aggregate(csolns, self.latest_model)
            self.client_model.load_state_dict(self.latest_model, strict=False)

            # Build personalization subspace on server (heads only)
            diffs = []
            client_flat_params = {}
            head_keys = self._head_keys(self.latest_model)
            self.global_flat = torch.cat([self.latest_model[k].flatten() for k in head_keys]).cpu().numpy()
            for cid, state in self.client_last_state.items():
                client_flat = torch.cat([state[k].flatten() for k in head_keys]).cpu().numpy()
                client_flat_params[cid] = client_flat
                diffs.append(client_flat - self.global_flat)
            self.client_flat_params = client_flat_params
            if len(diffs) >= self.basis_rank:
                try:
                    X = np.stack(diffs, axis=1)
                    U, _, _ = np.linalg.svd(X, full_matrices=False)
                    self.U = U[:, : self.basis_rank]
                except Exception:
                    self.U = None
            else:
                self.U = None

            # update clustering and controller training
            self._update_clusters()
            # For DRL, build per-client transitions using last stored (s_t, a_t) and new s_{t+1}
            if self.controller_type == "drl":
                for client in selected_clients:
                    cid = client.id
                    if cid in self.rl_last_feature and cid in self.rl_last_action and cid in self.client_feature_dict:
                        reward = self._compute_drl_reward(self.rl_last_feature[cid], self.client_feature_dict[cid])
                        self.controller.add_transition(
                            state=self.rl_last_feature[cid],
                            action=self.rl_last_action[cid],
                            reward=reward,
                            next_state=self.client_feature_dict[cid],
                            noise_std=self.dp_noise_std,
                        )
                # Update stored last features to the freshly observed states for the next round
                for client in selected_clients:
                    cid = client.id
                    if cid in self.client_feature_dict:
                        self.rl_last_feature[cid] = self.client_feature_dict[cid]
            if (rnd + 1) % self.controller_train_interval == 0:
                # Use all selected clients for richer supervision to avoid degenerate labels
                sampled = selected_clients
                if self.controller_type != "drl":
                    pseudo = self.oracle.label_clients(
                        sampled,
                        global_state=self.latest_model,
                        prev_global_state=self.prev_global_model,
                        server_test_fn=self._server_test_accuracy,
                        aggregate_fn=self._aggregate,
                        prepare_model_fn=self._prepare_strategy_model,
                        alpha=self.hybrid_alpha,
                        clients_per_round=self.clients_per_round,
                    )
                # Log pseudo-label distribution for debugging (MLP path only)
                if self.controller_type != "drl":
                    label_hist = defaultdict(int)
                    for cid, label in pseudo.items():
                        label_hist[label.name] += 1
                    if len(label_hist) > 0:
                        print(f"[Oracle] round {rnd} labels: {dict(label_hist)}")

                if self.controller_type != "drl":
                    for cid, label in pseudo.items():
                        if cid in self.client_feature_dict:
                            self.controller.add_sample(self.client_feature_dict[cid], label, noise_std=self.dp_noise_std)
                self._train_controller_if_ready()

            # After controller training (or fallback heuristic), assign per-client personalized state
            for client in selected_clients:
                strat = self._decide_strategy(client.id)
                state_for_client = self._prepare_strategy_model(client, strat, self.latest_model)
                self.client_personal_state[client.id] = state_for_client
                if self.controller_type == "drl" and client.id in self.client_feature_dict:
                    # Remember action taken for next round's transition
                    self.rl_last_action[client.id] = strat

            # record strategy fractions (what would be applied at eval time)
            strat_counts = defaultdict(int)
            for cid in self.client_feature_dict.keys():
                s = self._decide_strategy(cid)
                strat_counts[s.name] += 1
            total_clients = sum(strat_counts.values())
            if total_clients > 0:
                frac = {k: v / total_clients for k, v in strat_counts.items()}
                self.strategy_fractions.append(frac)
                if (rnd + 1) % self.controller_train_interval == 0:
                    print(f"[Controller] round {rnd} strategy fractions (would apply if ready): {frac}")

        self.eval_end()

    def set_client_model_test(self, client):
        # Prefer the cached personalized/generalized state for this client; fallback to latest global
        cached = self.client_personal_state.get(client.id, None)
        if cached is not None:
            client.set_model_params(cached)
            return

        # Fallback: use latest global (legacy path)
        client.set_model_params(self.latest_model)

    def _prepare_strategy_model(self, client, strategy: Strategy, global_state: OrderedDict = None) -> OrderedDict:
        if global_state is None:
            global_state = self.latest_model

        if strategy == Strategy.GENERALIZE:
            return global_state

        if self.U is None or client.id not in self.client_flat_params:
            return global_state

        head_keys = self._head_keys(global_state)
        client_flat = self.client_flat_params[client.id]
        personalized_grass_flat = self.personalize_via_grassmann(
            client_flat, self.global_flat, self.U
        )

        pointer = 0
        personalized_params = OrderedDict()
        for name, tensor in global_state.items():
            if name in head_keys:
                numel = tensor.numel()
                new_vals = personalized_grass_flat[pointer : pointer + numel].reshape(tensor.shape)
                personalized_params[name] = torch.tensor(new_vals, dtype=tensor.dtype)
                pointer += numel
            else:
                personalized_params[name] = tensor.clone()

        if strategy == Strategy.PERSONALIZE:
            return personalized_params

        if strategy == Strategy.HYBRID:
            prev_params = self.client_prev_params.get(client.id, personalized_params)
            mixed = OrderedDict()
            for k in personalized_params.keys():
                mixed[k] = (1 - self.hybrid_alpha) * personalized_params[k] + self.hybrid_alpha * prev_params.get(k, personalized_params[k])
            return mixed

        return global_state

    # ------------ personalization helpers (server-side) -----------------
    def learn_subspace_basis(self, diffs: List[np.ndarray], r: int):
        X = np.stack(diffs, axis=1)  # d × N
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        return U[:, :r]  # d × r

    def personalize_via_grassmann(self, client_flat: np.ndarray, global_flat: np.ndarray, U: np.ndarray):
        diff = client_flat - global_flat
        alpha = U.T @ diff  # coefficients
        personalized_flat = global_flat + U @ alpha
        return personalized_flat

    def personalize_optimal_transport(self, client_params: OrderedDict, prev_params: OrderedDict, tau: float = 0.5):
        personalized = OrderedDict()
        for k in client_params.keys():
            personalized[k] = (1 - tau) * client_params[k] + tau * prev_params[k]
        return personalized
