"""
FedAdam server-side aggregator — corrected to follow the FedOpt paper (FedAdam).
Key changes vs original snippet submitted:
- Treat negative average model difference as pseudo-gradient g_t = -Delta_t (important).
- Use bias correction for m and v.
- Ensure device/dtype consistency and numerical stability.
Reference: "Adaptive Federated Optimization" (Reddi et al., 2020/ICLR). :contentReference[oaicite:1]{index=1}
"""

import torch
from collections import OrderedDict
from flearn.trainers.server import BaseServer
from flearn.clients.fedavg import FedAvgClient
from flearn.utils.constants import CLASSES
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader


class FedAdamServer(BaseServer):
    def __init__(self, params):
        print("Using FedAdam (Adaptive Federated Optimization with Adam) to Train")

        # FedAdam hyperparameters        
        params["server_learning_rate"] = 1e-2  # ensure it's in params
        self.server_lr = params.get('server_learning_rate', 0.01)  # η in the paper
        self.beta1 = params.get('beta1', 0.9)  # First moment decay
        self.beta2 = params.get('beta2', 0.99)  # Second moment decay
        # 'tau' used in some FedOpt descriptions as epsilon-like term in denom
        self.tau = params.get('tau', 1e-3)
        # also allow tiny eps (kept for numerical safety; you can leave tau as the paper's)
        self.eps = params.get('eps', 1e-12)

        super().__init__(params)

        self.num_classes = CLASSES[self.dataset]
        self.clients: list[FedAvgClient] = self.clients

        # Initialize server-side optimizer state
        self.m = None  # OrderedDict of first moments
        self.v = None  # OrderedDict of second moments
        self.t = 0     # time step / round index (starts from 0, we increment before bias-corr)

        # Set num_classes for all clients
        for client in self.clients:
            client.num_classes = self.num_classes

        # (optional) print dataset statistics (kept your helper)
        if hasattr(self, "clients") and self.clients:
            def subset_labels(subset) -> torch.Tensor:
                if hasattr(subset, "dataset") and hasattr(subset, "indices"):
                    base_ds = subset.dataset
                    idx = subset.indices
                    targets = getattr(base_ds, "targets", None)
                    if isinstance(targets, torch.Tensor):
                        return targets[idx].clone().detach().cpu().long()
                    elif isinstance(targets, (list, tuple, np.ndarray)):
                        return torch.as_tensor([targets[i] for i in idx], dtype=torch.long)
                return torch.cat([y.detach().cpu().long() for _, y in DataLoader(subset, batch_size=256)], dim=0)

            total_train_samples = 0
            total_val_samples = 0
            train_class_counts = np.zeros(self.num_classes, dtype=np.int64)
            val_class_counts = np.zeros(self.num_classes, dtype=np.int64)

            for client in self.clients:
                train_subset = client.trainloader.dataset
                val_subset = client.valloader.dataset
                total_train_samples += len(train_subset)
                total_val_samples += len(val_subset)

                tl = subset_labels(train_subset)
                binc = torch.bincount(tl, minlength=self.num_classes).numpy()
                train_class_counts += binc

                vl = subset_labels(val_subset)
                vbinc = torch.bincount(vl, minlength=self.num_classes).numpy()
                val_class_counts += vbinc

            print(f"Total training samples: {int(total_train_samples)}")
            print(f"Total validation samples: {int(total_val_samples)}")
            print(f"Train class distribution: {train_class_counts.tolist()}")
            print(f"Val class distribution: {val_class_counts.tolist()}")

    def train(self):
        """Train using FedAdam with server-side Adam optimization"""
        print(f"Training with {self.clients_per_round} workers using FedAdam")
        print(f"Server LR: {self.server_lr}, Beta1: {self.beta1}, Beta2: {self.beta2}, Tau: {self.tau}")

        for i in trange(self.num_rounds, desc=self.desc):
            # Evaluate model
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break

            # Select clients for this round
            selected_clients: list[FedAvgClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )

            # Local training with standard SGD
            csolns = []
            for c in tqdm(selected_clients, desc="Training Clients", leave=False):
                c.round = i
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )
                # soln should be a tuple (weight, state_dict) as you used earlier
                csolns.append(soln)

            # Aggregate using Adam-style adaptive optimization
            self.latest_model = self.aggregate_adam(csolns)
            self.client_model.load_state_dict(self.latest_model)

        # Final evaluation
        self.eval_end()

    def set_client_model_test(self, client: FedAvgClient):
        """Set client model for testing"""
        client.model.load_state_dict(self.client_model.state_dict())

    @torch.no_grad()
    def aggregate_adam(self, wsolns):
        """
        Aggregate client updates using Adam optimizer (FedAdam).
        Assumes wsolns is an iterable of (weight, client_state_dict).
        Returns: new global state_dict (OrderedDict) — same keys as client_model.state_dict().
        """

        # increment time step
        self.t += 1

        device = next(self.client_model.parameters()).device
        dtype = next(self.client_model.parameters()).dtype

        # current server model state
        current_state = self.client_model.state_dict()

        # build zero-filled OrderedDict for deltas / grads
        Delta = OrderedDict()
        for k in current_state.keys():
            Delta[k] = torch.zeros_like(current_state[k], device=device, dtype=dtype)

        # compute weighted average of client deltas: Δ = sum_k p_k * (client_k - server)
        total_weight = 0.0
        for w, client_state in wsolns:
            if w is None:
                w = 1.0
            total_weight += float(w)
            # ensure tensors moved to server device before arithmetic
            for k in current_state.keys():
                # client_state may be on CPU, move to device and match dtype
                cparam = client_state[k].to(device=device, dtype=dtype)
                Delta[k] += float(w) * (cparam - current_state[k])

        if total_weight == 0.0:
            # nothing to do; return current state
            return OrderedDict({k: v.clone() for k, v in current_state.items()})

        # normalize to get average model change
        for k in Delta.keys():
            Delta[k] = Delta[k] / total_weight

        # Pseudo-gradient: g_t = - Delta_t  (important — negative of model change)
        pseudo_grads = OrderedDict({k: -Delta[k].clone() for k in Delta.keys()})

        # Initialize m and v if first time (put them on same device/dtype)
        if self.m is None or self.v is None:
            self.m = OrderedDict()
            self.v = OrderedDict()
            for k, g in pseudo_grads.items():
                self.m[k] = torch.zeros_like(g, device=device, dtype=dtype)
                self.v[k] = torch.zeros_like(g, device=device, dtype=dtype)

        # Update biased first and second moments
        for k in pseudo_grads.keys():
            g = pseudo_grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)

        # Bias correction
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        # Compute parameter update and apply (gradient descent step)
        new_state = OrderedDict()
        for k in current_state.keys():
            # m_hat and v_hat
            m_hat = self.m[k] / bias_correction1
            v_hat = self.v[k] / bias_correction2
            # Adam-style update direction: m_hat / (sqrt(v_hat) + tau)
            denom = torch.sqrt(v_hat) + self.tau + self.eps
            step = self.server_lr * (m_hat / denom)
            # Remember: pseudo_grads = -Delta, and Adam update is gradient descent,
            # so we subtract the step: x_{t+1} = x_t - step
            # (equivalently, since pseudo_grads includes the negative, subtracting is correct)
            new_state[k] = (current_state[k] - step).clone()

        return new_state
