import torch
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.elastic import FedAvgClient
from typing import List
from copy import deepcopy


class FedAvgServer(BaseServer):
    def __init__(self, params):
        print("Using Federated avg to Train")

        params["tau"] = 0.5
        params["mu"]= 0.95
        params['endpoints'] = 20
        super().__init__(params)

        # Initialize global model parameters
        self.global_params_dict = OrderedDict(self.client_model.named_parameters())
        for param in self.global_params_dict.values():
            param.requires_grad = False

        # Set attributes for all clients
        for client in self.clients:
            client.mu = self.mu
            client.tau = self.tau
            client.client_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    client.optimizer, self.num_rounds
                )

    def train(self):
        """Train using Federated Averaging"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)           
            if self.loss_converged: break

            selected_clients: list[FedAvgClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            delta_cache = []
            weight_cache = []
            sensitivity_cache = []

            with torch.no_grad():
                old_params_dic = deepcopy(OrderedDict(self.global_params_dict))
            for _, c in enumerate(selected_clients):  
                c.model.load_state_dict(old_params_dic, strict=False) 
                soln, stats, sensitivity = c.solve_inner_elastic(num_epochs=self.num_epochs, batch_size=self.batch_size)
                delta = OrderedDict()
                for (name, p0), p1 in zip( old_params_dic.items(), c.model.parameters()):
                    delta[name] = p0 - p1
                delta_cache.append(delta)
                weight_cache.append(soln[0])
                sensitivity_cache.append(sensitivity)

                c.client_lr_scheduler.step()

            # update models
            self.aggregate(delta_cache, weight_cache, sensitivity_cache)

        self.eval_end()

    def set_client_model_test(self, client: FedAvgClient):
        client.set_model_params(self.global_params_dict)

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, torch.Tensor]],
        weight_cache: List[int],
        sensitivity_cache: List[torch.Tensor],
    ):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        stacked_sensitivity = torch.stack(sensitivity_cache, dim=-1)
        aggregated_sensitivity = torch.sum(stacked_sensitivity * weights, dim=-1)

        max_sensitivity = stacked_sensitivity.max(dim=-1)[0]
        max_sensitivity = torch.where(max_sensitivity == 0, torch.tensor(1e-7, device=self.device), max_sensitivity)
        zeta = 1 + self.tau - aggregated_sensitivity / max_sensitivity
        delta_list = [list(delta.values()) for delta in delta_cache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]

        for param, coef, diff in zip(
            self.global_params_dict.values(), zeta, aggregated_delta
        ):
            param.data -= coef * diff
        self.client_model.load_state_dict(self.global_params_dict, strict=False)
        
