from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch


class LocalClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: torch.nn.Module = self.model

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

    def train_error_and_loss_model(self, model:torch.nn.Module, modelInCPU: bool = False):
        tot_correct, loss, train_sample = 0, 0.0, 0

        if modelInCPU:
            model = model.to(self.device)

        for inputs, labels in self.trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            loss += self.criterion(outputs, labels).item()
            train_sample += len(labels)

        if modelInCPU:
            model = model.cpu()

        return tot_correct, loss, train_sample
    
    def test_model_l(self, model:torch.nn.Module, modelInCPU: bool = False) -> tuple[int, int]:
        """tests the current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        """
        # print(f'Calling Local test')
        if modelInCPU:
            model = model.to(self.device)

        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            loss += self.criterion(outputs, labels).item()
            test_sample += len(labels)

        if modelInCPU:
            model = model.cpu()

        return tot_correct, test_sample