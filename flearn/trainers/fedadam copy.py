"""
FedAdam: Adaptive Federated Optimization with Adam
Paper: "Adaptive Federated Optimization" (Reddi et al., MLSys 2021)
https://arxiv.org/abs/2003.00295

Key Idea:
- Clients perform standard SGD training
- Server uses Adam optimizer for aggregation
- Combines momentum (first moment) and adaptive learning rates (second moment)
- Standard Adam algorithm applied to federated pseudo-gradients

Algorithm:
Server maintains:
- m_t: first moment (momentum)
- v_t: second moment (adaptive learning rate)
- Server learning rate η
- β1, β2: moment decay rates
- τ: adaptive learning rate parameter

Update rule:
Δ_t = Σ_i (p_i * Δx_i)  # weighted average of client updates
m_t = β1 * m_{t-1} + (1 - β1) * Δ_t
v_t = β2 * v_{t-1} + (1 - β2) * Δ_t^2  # Adam update (multiplicative)
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
x_t+1 = x_t + η * m_hat / (sqrt(v_hat) + τ)
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedavg import FedAvgClient
from flearn.utils.constants import CLASSES


class FedAdamServer(BaseServer):
    def __init__(self, params):
        print("Using FedAdam (Adaptive Federated Optimization with Adam) to Train")
        
        # FedAdam hyperparameters
        self.server_lr = params.get('server_learning_rate', 0.01)  # η in the paper
        self.beta1 = params.get('beta1', 0.9)  # First moment decay
        self.beta2 = params.get('beta2', 0.99)  # Second moment decay
        self.tau = params.get('tau', 1e-3)  # Adaptive learning rate parameter
        
        super().__init__(params)
        
        self.num_classes = CLASSES[self.dataset]
        self.clients: list[FedAvgClient] = self.clients
        
        # Initialize server-side optimizer state
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
        # Set num_classes for all clients
        for client in self.clients:
            client.num_classes = self.num_classes

        # Print dataset statistics
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
        Aggregate client updates using Adam optimizer
        
        Standard Adam algorithm:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        """
        self.t += 1
        total_weight = 0.0
        current_state = self.client_model.state_dict()
        
        # Compute pseudo-gradients (difference from current model)
        pseudo_grads = OrderedDict()
        for key in current_state.keys():
            pseudo_grads[key] = torch.zeros_like(current_state[key])
        
        # Weighted average of pseudo-gradients
        for w, client_state_dict in wsolns:
            total_weight += w
            for key in current_state.keys():
                delta = client_state_dict[key].to(self.device) - current_state[key]
                pseudo_grads[key] += w * delta
        
        # Normalize by total weight
        for key in pseudo_grads.keys():
            pseudo_grads[key] /= total_weight
        
        # Initialize m and v if first round
        if self.m is None:
            self.m = OrderedDict()
            self.v = OrderedDict()
            for key, grad in pseudo_grads.items():
                self.m[key] = torch.zeros_like(grad)
                self.v[key] = torch.zeros_like(grad)
        
        # Adam update
        new_state = OrderedDict()
        for key in current_state.keys():
            grad = pseudo_grads[key]
            
            # First moment (momentum)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Second moment (Adam-style multiplicative update)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update: x_t+1 = x_t + η * m_hat / (sqrt(v_hat) + τ)
            new_state[key] = current_state[key] + self.server_lr * m_hat / (torch.sqrt(v_hat) + self.tau)
        
        return new_state
