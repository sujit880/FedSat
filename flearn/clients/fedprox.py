from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict

FedProxMU = {
    "mnist":0.01, 
    "cifar":0.01, 
    "cifar100":0.001, 
    "tinyimagenet":0.001, 
    "emnist": 0.01, 
    "fashionmnist":0.01,
    "fmnist":0.01,
}

class FedAvgClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = FedProxMU[self.dataset]

    def solve_inner_fedprox(self, global_model: torch.nn.Module, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                fed_prox_reg = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    param_diff = w.data - w_t.data
                    fed_prox_reg += (self.mu / 2) * torch.norm(param_diff ** 2)
                loss += fed_prox_reg
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
