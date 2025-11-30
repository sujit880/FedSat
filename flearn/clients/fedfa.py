"""
FedFA: Federated Learning with Feature Augmentation
A frequency-aware federated learning approach that augments features using
frequency domain transformations to improve robustness to data heterogeneity.

Key Idea:
- Apply frequency-based data augmentation during local training
- Mix frequency components from different samples to create diverse training data
- Helps address class imbalance and non-IID data distribution

Note: This is an implementation based on the general concept of frequency-aware
federated learning for handling heterogeneous data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size


class FedFAClient(BaseClient):
    def __init__(self, freq_mix_prob=0.5, freq_mix_alpha=0.3, **kwargs):
        """
        Args:
            freq_mix_prob: Probability of applying frequency mixing augmentation
            freq_mix_alpha: Mixing ratio for frequency components (0-1)
            **kwargs: Other client parameters
        """
        super().__init__(**kwargs)
        self.freq_mix_prob = freq_mix_prob
        self.freq_mix_alpha = freq_mix_alpha
    
    def frequency_mix(self, x1, x2, alpha=0.3):
        """
        Mix frequency components of two images
        
        Args:
            x1: First image tensor [C, H, W]
            x2: Second image tensor [C, H, W]
            alpha: Mixing ratio (0-1), controls how much of x2 to mix into x1
        
        Returns:
            Mixed image in spatial domain
        """
        # Convert to frequency domain using FFT
        freq1 = torch.fft.fftn(x1, dim=(-2, -1))
        freq2 = torch.fft.fftn(x2, dim=(-2, -1))
        
        # Get magnitude and phase
        amp1, phase1 = torch.abs(freq1), torch.angle(freq1)
        amp2, phase2 = torch.abs(freq2), torch.angle(freq2)
        
        # Mix amplitudes (low frequencies from one, high from another)
        # Create a frequency mask
        _, h, w = x1.shape
        center_h, center_w = h // 2, w // 2
        
        # Use alpha to control the mixing
        # Mix amplitudes: take alpha from x2 and (1-alpha) from x1
        mixed_amp = (1 - alpha) * amp1 + alpha * amp2
        
        # Keep phase from x1 for stability
        mixed_phase = phase1
        
        # Reconstruct complex number
        mixed_freq = mixed_amp * torch.exp(1j * mixed_phase)
        
        # Convert back to spatial domain
        mixed_img = torch.fft.ifftn(mixed_freq, dim=(-2, -1)).real
        
        return mixed_img
    
    def apply_augmentation(self, inputs, labels):
        """
        Apply frequency-based augmentation with some probability
        
        Args:
            inputs: Batch of images [B, C, H, W]
            labels: Corresponding labels [B]
        
        Returns:
            Augmented inputs and labels
        """
        batch_size = inputs.size(0)
        
        # Randomly decide which samples to augment
        aug_mask = torch.rand(batch_size) < self.freq_mix_prob
        
        if not aug_mask.any():
            return inputs, labels
        
        # For samples to augment, find random pairs
        aug_indices = torch.where(aug_mask)[0]
        
        augmented_inputs = inputs.clone()
        
        for idx in aug_indices:
            # Pick a random sample to mix with
            mix_idx = torch.randint(0, batch_size, (1,)).item()
            
            # Apply frequency mixing
            augmented_inputs[idx] = self.frequency_mix(
                inputs[idx], 
                inputs[mix_idx], 
                alpha=self.freq_mix_alpha
            )
        
        return augmented_inputs, labels

    def solve_inner(self, num_epochs=1, batch_size=10):
        """
        Local training with frequency-based augmentation
        
        Returns:
            Solution and statistics
        """
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        
        for epoch in range(num_epochs):
            # Initialize epoch-specific counters if needed
            if self.loss == "PSL":
                K = self.num_classes
                conf = torch.zeros(K, K, dtype=torch.long, device=self.device)
            
            if self.loss == "DBCC":
                self.criterion.set_epoch(epoch=epoch)
            
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Apply frequency-based augmentation
                if self.training:
                    inputs, labels = self.apply_augmentation(inputs, labels)
                
                # Add noise if needed for DP
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level
                
                self.optimizer.zero_grad()
                
                # Forward pass based on loss type
                if self.loss == "DBCC":
                    feats = self.model.get_representation_features(inputs)
                    logits = self.model.classifier(feats)
                    if hasattr(self.model, "resnet"):
                        if hasattr(self.model.resnet, "fc"):
                            class_w = self.model.resnet.fc.weight
                    elif hasattr(self.model, "fc"):
                        class_w = self.model.fc.weight
                    elif hasattr(self.model, "linear"):
                        class_w = self.model.linear.weight
                    loss = self.criterion(logits, labels, feats, class_w)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Update confusion matrix for PSL
                if self.loss == "PSL":
                    with torch.no_grad():
                        self.confusion_bincount(conf, outputs, labels)
                
                # Update confusion counters for CAPA
                if self.loss == "CAPA":
                    K = self.num_classes
                    with torch.no_grad():
                        probs = torch.softmax(outputs, dim=1)
                        onehot_y = F.one_hot(labels, num_classes=K).float()
                        self.conf_N += onehot_y.T @ probs
                        self.pred_q += probs.sum(dim=0)
                        self.label_y += onehot_y.sum(dim=0)
                
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)
            
            # Update loss weights for PSL
            if self.loss == "PSL":
                self.criterion.ema_update_from_confusion(
                    conf, alpha=0.1,
                    w_min=1.0, w_max=2.0,
                    tau=10.0, gamma=1.0
                )
            
            # Update weights for CAPA
            if self.loss == "CAPA":
                with torch.no_grad():
                    W, U = self.build_W_U(
                        self.conf_N, self.pred_q, self.label_y,
                        beta=1.0, gamma=2.0, eps=1e-6
                    )
                    self.criterion.update_weights(W, U)
                # Decay counters
                decay = 0.9
                self.conf_N *= decay
                self.pred_q *= decay
                self.label_y *= decay
        
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    @torch.no_grad()
    def confusion_bincount(self, conf: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
        """Vectorized in-place accumulation of confusion counts"""
        K = logits.size(1)
        preds = logits.argmax(dim=1)
        idx = targets * K + preds
        counts = torch.bincount(idx, minlength=K*K).reshape(K, K)
        conf += counts
    
    @torch.no_grad()
    def build_W_U(self, conf_counts, pred_counts, label_counts, beta=1.0, gamma=2.0, eps=1e-6):
        """Build weight matrices for CAPA loss"""
        K = conf_counts.size(0)
        C = conf_counts.clone().float()
        C.fill_diagonal_(0.0)
        C = C / (C.sum(dim=1, keepdim=True) + eps)
        
        W = (C + eps).pow(beta)
        W.fill_diagonal_(0.0)
        W = W / (W.sum(dim=1, keepdim=True) + eps)
        
        pred = pred_counts.float()
        pred = pred / (pred.sum() + eps)
        lab = label_counts.float()
        lab = lab / (label_counts.sum() + eps)
        over = torch.clamp(pred - lab, min=0.0).pow(gamma)
        U = over / (over.sum() + eps)
        
        return W, U
