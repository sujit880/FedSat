from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch

from flearn.clients.fedavg import FedAvgClient
from flearn.strategy.hybrid_controller import ClientFeatures


class HybridClient(FedAvgClient):
    """FedAvg client that returns privacy-preserving stats for server-side strategy selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.participation_count: int = 0
        self.cached_label_entropy: Optional[float] = None

    @staticmethod
    def _flatten_state_dict(state: OrderedDict) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in state.values()])

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            total += torch.norm(p.grad.detach()) ** 2
        return float(torch.sqrt(total + 1e-12).item())

    def _eval_loss(self, loader: torch.utils.data.DataLoader, max_batches: int = 2) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                if idx >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                losses.append(loss.item())
        self.model.train()
        if len(losses) == 0:
            return 0.0
        return float(np.mean(losses))

    def _eval_acc(self, loader: torch.utils.data.DataLoader, max_batches: int = 2) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                if idx >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        self.model.train()
        if total == 0:
            return 0.0
        return float(correct) / total

    def _label_entropy(self) -> float:
        if self.cached_label_entropy is not None:
            return self.cached_label_entropy
        counts = torch.zeros(self.num_classes, device=self.device)
        with torch.no_grad():
            for _, y in self.trainloader:
                y = y.to(self.device)
                binc = torch.bincount(y, minlength=self.num_classes).float()
                counts += binc
        probs = counts / counts.sum().clamp(min=1.0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        self.cached_label_entropy = float(entropy)
        return self.cached_label_entropy

    def train_and_report(
        self,
        global_state: OrderedDict,
        prev_global_state: Optional[OrderedDict],
        num_epochs: int,
        batch_size: int,
    ) -> Tuple[Tuple[int, OrderedDict], ClientFeatures]:
        """Run local training and return model plus privacy-preserving statistics."""
        self.participation_count += 1
        # sync to current global
        self.set_model_params(global_state)

        pre_loss = self._eval_loss(self.valloader)
        pre_acc = self._eval_acc(self.valloader)
        # local train
        train_sample_size = 0
        grad_norm_accum = 0.0
        batch_count = 0
        for _ in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels) <= 1:
                    continue
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                grad_norm_accum += self._grad_norm()
                batch_count += 1
                self.optimizer.step()
                train_sample_size += len(labels)
        grad_norm_avg = grad_norm_accum / max(1, batch_count)

        post_loss = self._eval_loss(self.valloader)
        post_acc = self._eval_acc(self.valloader)

        local_state = self.get_model_params()
        delta_loss = pre_loss - post_loss

        # cosine similarity with global update if previous global exists
        if prev_global_state is not None:
            local_delta = self._flatten_state_dict(local_state) - self._flatten_state_dict(global_state)
            global_delta = self._flatten_state_dict(global_state) - self._flatten_state_dict(prev_global_state)
            denom = torch.norm(local_delta) * torch.norm(global_delta) + 1e-12
            cos_sim = float(torch.dot(local_delta, global_delta) / denom)
        else:
            cos_sim = 0.0

        entropy = self._label_entropy()
        feats = ClientFeatures(
            delta_loss=delta_loss,
            grad_norm=grad_norm_avg,
            cos_sim=cos_sim,
            entropy=entropy,
            participation=self.participation_count,
            local_val_acc=post_acc,
            global_val_acc=pre_acc,
        )

        return (train_sample_size, local_state), feats

    def simulate_strategy(
        self,
        start_state: OrderedDict,
        rounds: int = 1,
    ) -> Tuple[OrderedDict, float, float]:
        """Simulate training with a specific strategy (model state) and return outcome."""
        backup_state = self.get_model_params()
        optim_state = deepcopy(self.optimizer.state_dict())

        self.set_model_params(start_state)
        local_before = self._eval_acc(self.valloader)

        # Train for 'rounds' epochs
        for _ in range(rounds):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        local_after = self._eval_acc(self.valloader)
        sim_state = self.get_model_params()

        # Restore state
        self.set_model_params(backup_state)
        self.optimizer.load_state_dict(optim_state)

        return sim_state, local_before, local_after
