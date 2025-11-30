"""
FedNTD: Not-True Distillation for Federated Learning
Paper: "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning"
NeurIPS 2022
https://proceedings.neurips.cc/paper_files/paper/2022/file/fadec8f2e65f181d777507d1df69b92f-Paper-Conference.pdf

Key Idea:
- Preserves global knowledge using "not-true" distillation
- For each sample, distills from logits of classes that are NOT the true label
- Helps prevent catastrophic forgetting of global knowledge during local training
- Particularly effective for non-IID and class-imbalanced scenarios
"""

from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
import torch.nn.functional as F
import copy


class FedNTDClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model = None
        self.beta_ntd = 1.0  # Weight for not-true distillation loss
        self.tau_ntd = 1.0   # Temperature for distillation
        
    def set_global_model(self, global_model_params):
        """Store global model for distillation"""
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.model)
        self.global_model.load_state_dict(global_model_params)
        self.global_model.eval()
        
    def not_true_distillation_loss(self, student_logits, teacher_logits, labels, tau=1.0):
        """
        Compute not-true distillation loss
        
        Args:
            student_logits: Logits from local model [B, K]
            teacher_logits: Logits from global model [B, K]
            labels: True labels [B]
            tau: Temperature for softmax
            
        Returns:
            KL divergence on not-true classes
        """
        batch_size, num_classes = student_logits.shape
        
        # Create mask for not-true classes (all except true label)
        mask = torch.ones_like(student_logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        
        # Apply temperature and compute softmax on not-true classes only
        student_not_true = student_logits.masked_fill(~mask, float('-inf'))
        teacher_not_true = teacher_logits.masked_fill(~mask, float('-inf'))
        
        # Compute soft targets (with temperature)
        student_soft = F.log_softmax(student_not_true / tau, dim=1)
        teacher_soft = F.softmax(teacher_not_true / tau, dim=1)
        
        # KL divergence on not-true classes
        # Handle -inf values from masking
        valid_mask = ~torch.isinf(student_soft)
        kl_loss = F.kl_div(
            student_soft.masked_fill(~valid_mask, 0),
            teacher_soft.masked_fill(~valid_mask, 0),
            reduction='batchmean'
        ) * (tau ** 2)
        
        return kl_loss

    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization with not-true distillation

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
        if self.global_model is not None:
            self.global_model.eval()
        
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level
                
                self.optimizer.zero_grad()
                
                # Local model forward pass
                student_logits = self.model(inputs)
                
                # Standard cross-entropy loss
                ce_loss = self.criterion(student_logits, labels)
                
                # Not-true distillation loss (if global model is available)
                ntd_loss = 0.0
                if self.global_model is not None:
                    with torch.no_grad():
                        teacher_logits = self.global_model(inputs)
                    
                    ntd_loss = self.not_true_distillation_loss(
                        student_logits, 
                        teacher_logits, 
                        labels, 
                        tau=self.tau_ntd
                    )
                
                # Combined loss
                if self.global_model is not None:
                    total_loss = ce_loss + self.beta_ntd * ntd_loss
                else:
                    total_loss = ce_loss
                
                total_loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
