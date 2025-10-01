from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict
from copy import deepcopy

class SCAFFOLDClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_client_specific_params(
        self,
        c_local: OrderedDict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        self.c_local: OrderedDict[str, torch.Tensor] = c_local
        for val in self.c_local.values():
            val.requires_grad = False


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
    
    def solve_inner_scaffold(self,c_global: OrderedDict[str, torch.Tensor], 
                            global_parameters: OrderedDict[str, torch.Tensor],
                            num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        self.model.load_state_dict(global_parameters, strict=False)
        self.model.train()
        for epoch in range(num_epochs): 
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                for param, G, c_i in zip(
                    self.model.parameters(), c_global.values(), self.c_local.values()
                ):
                    param.grad.data += (G - c_i).to(self.device) if G.device !=self.device else (G - c_i)
                self.optimizer.step()
                train_sample_size += len(labels)

        self.model.eval()               
        with torch.no_grad():
            y_delta = OrderedDict()
            c_delta = OrderedDict()   
            c_plus = OrderedDict()
            # compute y_delta (difference of model before and after training)
            for k,v in self.model.named_parameters():
                y_i = v.data
                x_i = global_parameters[k].data
                y_delta[k] = y_i - x_i
            
            # compute c_plus
            # coef = 1/num_epochs * self.lr
            coef = 1/num_epochs * self.lr
            for k, G, c_l, y_del in zip(
                c_global.keys(), c_global.values(), self.c_local.values(), y_delta.values()
            ):
                c_plus[k]= c_l - G - coef * (y_del if y_del.device == G.device else y_del.to(G.device))

            # compute c_delta
            for k, c_p, c_l in zip(c_plus.keys(), c_plus.values(), self.c_local.values()):
                c_delta[k] = c_p.data - c_l.data
            for k in self.c_local.keys():
                self.c_local[k] = c_plus[k]
        self.model.train()
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), (deepcopy(y_delta), deepcopy(c_delta))
    