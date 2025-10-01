from typing import Any
from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size

class FlocoClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_client_specific_params(self, endpoints, tau, rho, **kwargs):
        self.endpoints = endpoints
        self.tau = tau
        self.rho = rho
    
    def set_parameters(self, subregion_params: (tuple[Any, Any] | None)) -> None:
        self.model.subregion_parameters = subregion_params # type: ignore
        
    def solve_inner(self, num_epochs: int = 1, batch_size: int = 10) -> tuple[Any, Any]:
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
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step() # type: ignore
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)