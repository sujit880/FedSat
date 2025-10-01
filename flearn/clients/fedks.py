from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
from flearn.utils.losses import contrastive_separation_loss
from flearn.models.generative import ICVAE, CVAE, CLS
from flearn.utils.trainer_utils import get_optimizer_by_name
from flearn.models.model import TorchResNet
from typing import Tuple, Union, List, Optional
import torch.nn.functional as F
import torch
import random
from collections import OrderedDict
import copy
import numpy as np

class FedKSeedClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_pool_size = 20  # Number of candidate seeds to maintain
        self.seed_pool = None  # Will be initialized by server
        self.scalar_gradients = {}  # To store gradients for each seed
        
    def init_client_specific_params(
        self,     
        classifier: CLS,
        num_classes:int,
        lamda_falg:float,
        seed_pool=None,
        **kwargs,
    ) -> None:
        self.num_classes = num_classes
        self.classifier: CLS = classifier
        self.lambda_falg = lamda_falg 
        self.optimizer = get_optimizer_by_name(
                            optm=self.optm, 
                            parameters=list(self.model.parameters())+ list(self.classifier.parameters()), 
                            lr=self.lr, 
                            weight_decay=self.weight_decay, 
                            momentum = self.momentum, # For SGD only
                        )
        self.seed_pool = seed_pool if seed_pool is not None else list(range(self.seed_pool_size))
        
    def solve_inner_kseed(self, global_classifier, num_epochs=1, batch_size=10):
        """Train with ZOO using random seeds for parameter generation"""
        self.model.train()
        self.classifier.train()
        global_classifier.eval()
        
        # Store original model parameters
        original_params = OrderedDict({k: v.clone() for k, v in self.model.named_parameters()})
        
        # Reset scalar gradients for this round
        self.scalar_gradients = {seed: 0.0 for seed in self.seed_pool}
        
        # Training loop
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Sample a random seed from the pool
                seed = random.choice(self.seed_pool)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Generate perturbation using the seed
                perturbation = OrderedDict()
                for k, param in self.model.named_parameters():
                    if param.requires_grad:
                        # Generate random perturbation scaled by parameter magnitude
                        perturbation[k] = torch.randn_like(param) * 1e-3 * torch.norm(param)
                
                # Apply positive perturbation and compute loss
                with torch.no_grad():
                    for k, param in self.model.named_parameters():
                        if k in perturbation:
                            param.add_(perturbation[k])
                
                # Forward pass with positive perturbation
                resnet_features = self.model.get_representation_features(inputs)
                l_head = self.classifier.head(resnet_features)
                logits_pos = self.classifier.cls(l_head)
                loss_pos = self.criterion(logits_pos, labels)
                
                # Apply negative perturbation and compute loss
                with torch.no_grad():
                    for k, param in self.model.named_parameters():
                        if k in perturbation:
                            # Remove positive and add negative perturbation (-2× because we're adding the opposite)
                            param.add_(-2 * perturbation[k])
                
                # Forward pass with negative perturbation
                resnet_features = self.model.get_representation_features(inputs)
                l_head = self.classifier.head(resnet_features)
                logits_neg = self.classifier.cls(l_head)
                loss_neg = self.criterion(logits_neg, labels)
                
                # Compute scalar gradient estimate
                scalar_grad = (loss_pos - loss_neg) / 2
                self.scalar_gradients[seed] += scalar_grad.item()
                
                # Restore original parameters
                with torch.no_grad():
                    for k, param in self.model.named_parameters():
                        if k in perturbation:
                            param.copy_(original_params[k])
                
                # Update classifier normally with gradients
                self.optimizer.zero_grad()
                resnet_features = self.model.get_representation_features(inputs)
                l_head = self.classifier.head(resnet_features)
                logits = self.classifier.cls(l_head)
                
                # Get prototype from global classifier
                
                # Compute losses
                classifier_loss = self.criterion(logits, labels)
                
                # Only update classifier parameters
                total_loss = classifier_loss 
                total_loss.backward()
                self.optimizer.step()
        
        # Return classifier parameters and scalar gradients
        cls_params = OrderedDict({k: v.clone().detach() for k, v in self.classifier.named_parameters()})
        return self.scalar_gradients, (self.num_samples, cls_params)
