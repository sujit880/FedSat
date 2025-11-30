"""
FedSAM Server: Sharpness-Aware Minimization for Federated Learning
Paper: "Improving Generalization in Federated Learning by Seeking Flat Minima"
ECCV 2022
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedsam import FedSAMClient
from flearn.utils.constants import CLASSES


class FedSAMServer(BaseServer):
    def __init__(self, params):
        print("Using FedSAM (Sharpness-Aware Minimization) to Train")
        
        # FedSAM hyperparameters
        self.rho = params.get('rho', 0.05)  # Perturbation radius
        
        super().__init__(params)
        
        self.num_classes = CLASSES[self.dataset]
        self.clients: list[FedSAMClient] = self.clients
        
        # Set SAM hyperparameters for all clients
        for client in self.clients:
            client.num_classes = self.num_classes
            client.rho = self.rho

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
        """Train using FedSAM with Sharpness-Aware Minimization"""
        print(f"Training with {self.clients_per_round} workers using FedSAM")
        print(f"SAM Parameter: rho={self.rho}")

        for i in trange(self.num_rounds, desc=self.desc):
            # Evaluate model
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break

            # Select clients for this round
            selected_clients: list[FedSAMClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )

            # Local training with SAM
            csolns = []
            for c in tqdm(selected_clients, desc="Training Clients", leave=False):
                c.round = i
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )
                csolns.append(soln)

            # Aggregate updates using FedAvg
            self.latest_model = self.aggregate(csolns)
            self.client_model.load_state_dict(self.latest_model)

        # Final evaluation
        self.eval_end()

    def set_client_model_test(self, client: FedSAMClient):
        """Set client model for testing"""
        client.model.load_state_dict(self.client_model.state_dict())

    def aggregate(self, wsolns):
        """Weighted average aggregation (FedAvg style)"""
        total_weight = 0.0
        base = [torch.zeros_like(val) for val in self.client_model.state_dict().values()]

        for w, client_state_dict in wsolns:
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v.to(self.device)

        averaged_soln = [v / total_weight for v in base]
        averaged_state_dict = OrderedDict(
            zip(self.client_model.state_dict().keys(), averaged_soln)
        )
        return averaged_state_dict
