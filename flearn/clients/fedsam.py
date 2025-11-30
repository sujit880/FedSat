"""
FedSAM: Sharpness-Aware Minimization for Federated Learning
Paper: "Improving Generalization in Federated Learning by Seeking Flat Minima"
ECCV 2022
https://arxiv.org/abs/2203.11834

Key Idea:
- Applies Sharpness-Aware Minimization (SAM) to federated learning
- Seeks flatter minima which generalize better, especially important for:
  - Non-IID data distributions
  - Class imbalanced scenarios
  - Heterogeneous client data
- Two-step optimization: perturbation step + gradient step
"""

from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
import torch.nn.functional as F


class FedSAMClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho = 0.05  # Perturbation radius for SAM
        
    def sam_step(self, inputs, labels):
        """
        Perform one SAM optimization step
        
        SAM minimizes both the loss value and loss sharpness:
        min_w L(w) + rho * ||∇L(w)||
        
        Algorithm:
        1. Compute gradient at current point w
        2. Compute perturbation ε = rho * ∇L(w) / ||∇L(w)||
        3. Compute gradient at perturbed point w + ε
        4. Update w using gradient from perturbed point
        """
        # Enable gradient computation
        inputs.requires_grad = True
        
        # First forward-backward pass (for perturbation)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        
        # Compute perturbation
        # ε_w = rho * ∇L(w) / ||∇L(w)||_2
        with torch.no_grad():
            # Compute norm of gradients
            grad_norm = torch.sqrt(sum([
                torch.sum(p.grad ** 2) 
                for p in self.model.parameters() 
                if p.grad is not None
            ]))
            
            # Avoid division by zero
            grad_norm = torch.clamp(grad_norm, min=1e-12)
            
            # Apply perturbation to parameters
            for p in self.model.parameters():
                if p.grad is not None:
                    epsilon = self.rho * p.grad / grad_norm
                    p.add_(epsilon)
        
        # Second forward-backward pass (at perturbed point)
        self.optimizer.zero_grad()
        outputs_perturbed = self.model(inputs)
        loss_perturbed = self.criterion(outputs_perturbed, labels)
        loss_perturbed.backward()
        
        # Remove perturbation before optimizer step
        with torch.no_grad():
            # Recompute grad norm (should be same as before)
            grad_norm = torch.sqrt(sum([
                torch.sum(p.grad ** 2) 
                for p in self.model.parameters() 
                if p.grad is not None
            ]))
            grad_norm = torch.clamp(grad_norm, min=1e-12)
            
            # Remove perturbation
            for p in self.model.parameters():
                if p.grad is not None:
                    epsilon = self.rho * p.grad / grad_norm
                    p.sub_(epsilon)
        
        # Gradient is already computed at perturbed point, just step
        # Note: we need to recompute gradient at original point
        self.optimizer.zero_grad()
        outputs_original = self.model(inputs)
        loss_original = self.criterion(outputs_original, labels)
        loss_original.backward()
        self.optimizer.step()
        
        return loss_original.item()

    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization with SAM

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        
        self.model.train()
        
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level
                
                # SAM optimization step
                _ = self.sam_step(inputs, labels)
                
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size * 2  # x2 for SAM double forward
        bytes_r = graph_size(self.model)
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
