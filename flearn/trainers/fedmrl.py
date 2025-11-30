import json
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedmrl import FedMRLClient
from flearn.utils.aggregator import Aggregator
from flearn.utils.constants import CLASSES
from flearn.config.trainer_params import FEDMRL_ARGS
from copy import deepcopy

class FedMRLServer(BaseServer):
    def __init__(self, params):
        print("Using Federated MapRL to Train")
        adv_gain = float(params.get("adv_gain", FEDMRL_ARGS["adv_gain"]))
        FEDMRL_ARGS["adv_gain"] = 0
        FEDMRL_ARGS["loss"] = "CALC"
        FEDMRL_ARGS["agg"] = "mean"
        params["server_learning_rate"] = 1.0
        FEDMRL_ARGS["version"] = "contrastive"
        params['max_rl_steps'] = 15
        params.update({"use_prev_global_model":  False})
        params.update(FEDMRL_ARGS)
        super().__init__(params)
        self.adv_gain = adv_gain
        self._set_client_advantage_gain()
        self._log_adv_gain_meta()
        self.num_classes = CLASSES[self.dataset]
        for client in self.clients:
            client.num_classes = self.num_classes
            client.init_client_specific_params(
                tau=float(getattr(self, "tau", 0.5)),
                mu=0.01,
                adv_gain=self.adv_gain,
                version=str(getattr(self, "version", "mse")),
            )

        # choose aggregation method
        if self.agg is not None and not self.agg=="drl":
            self.aggregator = Aggregator(method=self.agg, model=self.client_model, lr=self.server_learning_rate)

        # cache RL weights from previous aggregation (defaults to uniform)
        self.last_client_weights: dict[int, float] = {}
        self.client_weight_stats: dict[int, dict[str, float]] = {}

        # Check and print overall dataset stats (train + val) across all clients
        if hasattr(self, "clients") and self.clients:
            def subset_labels(subset) -> torch.Tensor:
                # subset is a torch.utils.data.Subset wrapping our pickled Dataset
                if hasattr(subset, "dataset") and hasattr(subset, "indices"):
                    base_ds = subset.dataset
                    idx = subset.indices
                    targets = getattr(base_ds, "targets", None)
                    if isinstance(targets, torch.Tensor):
                        return targets[idx].clone().detach().cpu().long()
                    elif isinstance(targets, (list, tuple, np.ndarray)):
                        return torch.as_tensor([targets[i] for i in idx], dtype=torch.long)
                # Fallback: iterate to collect labels (may miss last batch if drop_last=True)
                return torch.cat([y.detach().cpu().long() for _, y in DataLoader(subset, batch_size=256)], dim=0)

            total_train_samples = 0
            total_val_samples = 0
            train_class_counts = np.zeros(self.num_classes, dtype=np.int64)
            val_class_counts = np.zeros(self.num_classes, dtype=np.int64)

            for client in self.clients:
                # Accurate counts from underlying Subset lengths (not affected by drop_last)
                train_subset = client.trainloader.dataset
                val_subset = client.valloader.dataset
                total_train_samples += len(train_subset)
                total_val_samples += len(val_subset)

                # Per-class histogram for train
                tl = subset_labels(train_subset)
                binc = torch.bincount(tl, minlength=self.num_classes).numpy()
                train_class_counts += binc

                # Per-class histogram for val
                vl = subset_labels(val_subset)
                vbinc = torch.bincount(vl, minlength=self.num_classes).numpy()
                val_class_counts += vbinc

            print(f"Total training samples across all clients: {int(total_train_samples)}")
            print(f"Total validation samples across all clients: {int(total_val_samples)}")
            print(f"Train class distribution (global): {train_class_counts.tolist()}")
            print(f"Val class distribution (global):   {val_class_counts.tolist()}")

    def _record_client_weight(self, cid: int, weight: float, accumulate: bool = True) -> None:
        """Track running mean for each client's RL weight."""
        if not accumulate or cid not in self.client_weight_stats:
            self.client_weight_stats[cid] = {"sum": weight, "count": 1.0}
        else:
            stats = self.client_weight_stats[cid]
            stats["sum"] += weight
            stats["count"] += 1.0
        stats = self.client_weight_stats[cid]
        self.last_client_weights[cid] = float(stats["sum"] / max(stats["count"], 1.0))

    def train(self):
        """Train using Federated Averaging"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)           
            if self.loss_converged: break

            selected_clients: list[FedMRLClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            # === provide RL feedback to clients (weight + global mean) ===
            weight_fallback = 1.0
            if not self.last_client_weights:
                for c in self.clients:
                    self._record_client_weight(c.id, weight_fallback, accumulate=False)

            if len(selected_clients) > 0:
                sel_weights = [float(self.last_client_weights.get(c.id, weight_fallback)) for c in selected_clients]
                mean_val = np.mean(sel_weights)
                avg_weight = float(mean_val) if np.isfinite(mean_val) else weight_fallback
            else:
                avg_weight = weight_fallback

            for c in selected_clients:
                w = float(self.last_client_weights.get(c.id, weight_fallback))
                adv = w - avg_weight
                c.set_policy_feedback({"w": w, "adv": adv, "avg_w": avg_weight})

            csolns = []  # buffer for receiving client solutions
            client_solutions_dict = {}

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_model_params(self.latest_model)
                # solve minimization locally with FedMRL routine
                stats, soln = c.solve_inner_fedmap(
                    self.latest_model,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                )
                # gather solutions from client
                if self.agg == "drl": 
                    client_solutions_dict[c.id] = soln[1]
                else:
                    csolns.append(soln)

            # update models
            if self.agg is not None and not self.agg=="drl":
                self.latest_model = self.aggregator.aggregate(csolns, self.latest_model)
            elif self.agg=="drl":
                self.round = i
                if self.use_prev_global_model:
                    client_solutions_dict[len(self.clients)+1] = deepcopy(self.latest_model)
                self.latest_model = self.drl_aggregate(client_solutions_dict)
                # If RL aggregator exposes per-client weights, store them for next round
                if hasattr(self, 'rl_client_weights') and isinstance(self.rl_client_weights, dict):
                    for k, v in self.rl_client_weights.items():
                        cid = int(k)
                        self._record_client_weight(cid, float(v), accumulate=True)
            else:   self.latest_model = self.aggregate(csolns)
            if not (self.agg == "drl" and hasattr(self, 'rl_client_weights')):
                # fall back to uniform weights for next round (only for selected clients)
                for c in selected_clients:
                    self._record_client_weight(c.id, weight_fallback, accumulate=False)
            self.client_model.load_state_dict(self.latest_model, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: FedMRLClient):
        client.set_model_params(self.latest_model)

    def _set_client_advantage_gain(self):
        for client in self.clients:
            if hasattr(client, "adv_gain"):
                client.adv_gain = self.adv_gain

    def _log_adv_gain_meta(self):
        if not hasattr(self, "filewriter") or not hasattr(self.filewriter, "path"):
            return
        try:
            self.filewriter.metadata.setdefault("trainer_params", {})
            self.filewriter.metadata["trainer_params"]["adv_gain"] = self.adv_gain
            metadata_path = os.path.join(self.filewriter.path, "metadata.json")
            with open(metadata_path, "w") as meta_file:
                json.dump(self.filewriter.metadata, meta_file, indent=2)
        except Exception as exc:
            print(f"[FedMRL] Warning: unable to record adv_gain metadata ({exc})")

    def aggregate(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors
        # Initialize base with zeros tensors with the same size as the first solution's parameters'
        model_state_dict: OrderedDict = wsolns[0][1]
        base = [torch.zeros_like(soln) for soln in model_state_dict.values()]

        for w, client_state_dict in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v

        # Divide each aggregated tensor by the total weight to compute the average
        averaged_soln = [v / total_weight for v in base]
        averaged_state_dict = OrderedDict(zip(model_state_dict.keys(), averaged_soln))

        return averaged_state_dict
