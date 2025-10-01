from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch

class FedAvgClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sensitivity = torch.zeros(
                len(list(self.model.parameters())), device=self.device
            )

    def solve_inner(self, num_epochs=1, batch_size=10):
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
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def solve_inner_elastic(self,num_epochs=1, batch_size=10):
        self.model.eval()
        for x, y in self.valloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, self.model.parameters()
                )
            ]
            for i in range(len(grads_norm)):
                self.sensitivity[i] = (
                    self.mu * self.sensitivity[i]
                    + (1 - self.mu) * grads_norm[i].abs()
                )
        self.model.train()
        soln, stats = self.solve_inner(num_epochs=num_epochs, batch_size=batch_size)
        return soln, stats, self.sensitivity 
    
