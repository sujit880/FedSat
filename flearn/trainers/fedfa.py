"""
FedFA: Federated Learning with Feature Augmentation (Server)
Server-side implementation for FedFA

The server performs standard FedAvg-style aggregation while clients apply
frequency-based feature augmentation during local training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedfa import FedFAClient
from flearn.utils.constants import CLASSES


class FedFAServer(BaseServer):
    def __init__(self, params):
        print("Using FedFA (Federated Feature Augmentation) to Train")
        
        # FedFA hyperparameters
        self.freq_mix_prob = params.get('freq_mix_prob', 0.5)  # Probability of applying mixing
        self.freq_mix_alpha = params.get('freq_mix_alpha', 0.3)  # Mixing ratio
        
        # Pass augmentation parameters to client initialization
        params['freq_mix_prob'] = self.freq_mix_prob
        params['freq_mix_alpha'] = self.freq_mix_alpha
        
        super().__init__(params)
        
        self.num_classes = CLASSES[self.dataset]
        self.clients: list[FedFAClient] = self.clients
        
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
            print(f"Frequency mix probability: {self.freq_mix_prob}")
            print(f"Frequency mix alpha: {self.freq_mix_alpha}")
    
    def train(self):
        """Train using FedFA with frequency-based augmentation"""
        print(f"Training with {self.clients_per_round} workers using FedFA")
        
        for i in trange(self.num_rounds, desc=self.desc):
            # Evaluate model
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break
            
            # Select clients for this round
            selected_clients: list[FedFAClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )
            
            # Local training with frequency-based augmentation
            csolns = []
            for c in tqdm(selected_clients, desc="Training Clients", leave=False):
                c.round = i
                # Enable training mode for augmentation
                c.model.train()
                c.training = True
                
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )
                csolns.append(soln)
                
                c.training = False
            
            # Aggregate using FedAvg (standard weighted averaging)
            self.latest_model = self.aggregate(csolns)
            self.client_model.load_state_dict(self.latest_model)
        
        # Final evaluation
        self.eval_end()
    
    def set_client_model_test(self, client: FedFAClient):
        """Set client model for testing (no augmentation during testing)"""
        client.model.load_state_dict(self.client_model.state_dict())
        client.model.eval()
        client.training = False
    
    @torch.no_grad()
    def aggregate(self, wsolns):
        """
        Standard FedAvg aggregation: weighted average of client models
        
        Args:
            wsolns: List of (weight, state_dict) tuples from clients
        
        Returns:
            Aggregated model state dict
        """
        total_weight = 0.0
        base_state = self.client_model.state_dict()
        
        # Initialize aggregated state
        aggregated_state = OrderedDict()
        for key in base_state.keys():
            aggregated_state[key] = torch.zeros_like(base_state[key])
        
        # Weighted sum of client models
        for w, client_state_dict in wsolns:
            total_weight += w
            for key in aggregated_state.keys():
                aggregated_state[key] += w * client_state_dict[key].to(self.device)
        
        # Normalize by total weight
        if total_weight > 0:
            for key in aggregated_state.keys():
                aggregated_state[key] = aggregated_state[key] / total_weight
        
        return aggregated_state
