"""
FedRS Client Implementation
Paper: "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data"
Authors: Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng
Venue: KDD 2021

Key Idea: Use restricted softmax that only considers classes present in local client data
"""

from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
import torch.nn.functional as F
from collections import OrderedDict


class FedRSClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_classes = None  # Will be set during initialization
        
    def init_client_specific_params(
        self,
        local_classes: list,
        **kwargs,
    ) -> None:
        """
        Initialize client-specific parameters for FedRS
        
        Args:
            local_classes: List of class labels present in this client's local dataset
        """
        self.local_classes = torch.tensor(local_classes, dtype=torch.long)
        print(f"Client {self.id} initialized with local classes: {local_classes}")
    
    def get_local_classes_from_data(self):
        """
        Extract unique class labels from client's training data
        Returns list of unique classes
        """
        all_labels = []
        for _, labels in self.trainloader:
            all_labels.extend(labels.tolist())
        unique_classes = sorted(list(set(all_labels)))
        return unique_classes
    
    def solve_inner(self, num_epochs=1, batch_size=10):
        """
        Standard training without restricted softmax (for baseline comparison)
        
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
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)
        
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def solve_inner_fedrs(self, num_epochs=1, batch_size=10):
        """
        FedRS training with restricted softmax
        Only computes softmax over classes present in client's local data
        
        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """
        if self.local_classes is None:
            # If not set, extract from data
            self.local_classes = torch.tensor(
                self.get_local_classes_from_data(), 
                dtype=torch.long,
                device=self.device
            )
        
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level
                
                self.optimizer.zero_grad()
                
                # Get full logits
                logits = self.model(inputs)
                
                # Apply restricted softmax loss
                loss = self.restricted_softmax_loss(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)
        
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def restricted_softmax_loss(self, logits, targets):
        """
        Compute cross-entropy loss with restricted softmax
        Only normalizes over classes present in local data
        
        Args:
            logits: (B, K) logits from classifier
            targets: (B,) ground truth labels
            
        Returns:
            loss: scalar tensor
        """
        batch_size, num_classes = logits.size()
        
        # Create mask for local classes
        mask = torch.zeros(num_classes, device=logits.device, dtype=torch.bool)
        mask[self.local_classes] = True
        
        # Mask out non-local classes by setting their logits to -inf
        # This ensures they don't contribute to softmax normalization
        masked_logits = logits.clone()
        masked_logits[:, ~mask] = -1e9  # Large negative value (effectively -inf)
        
        # Compute cross-entropy with masked logits
        # PyTorch's cross_entropy applies softmax internally
        loss = F.cross_entropy(masked_logits, targets, reduction='mean')
        
        return loss
    
    def test_model_(self, model: torch.nn.Module = None, modelInCPU: bool = False) -> tuple[int, int]:
        """
        Tests the model on local validation data
        Uses full softmax for testing (not restricted)
        
        Args:
            model: Model to test (default: self.model)
            modelInCPU: Whether model is on CPU
            
        Returns:
            tot_correct: total number of correct predictions
            test_samples: total number of test samples
        """
        if model is None:
            model = self.model
        
        model.eval()
        tot_correct = 0
        test_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.valloader:
                if not modelInCPU:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                tot_correct += (predicted == labels).sum().item()
                test_samples += labels.size(0)
        
        return tot_correct, test_samples
