import torch
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from typing import List
from copy import deepcopy
from collections import OrderedDict
from flearn.clients.scaffold import SCAFFOLDClient
from flearn.config.trainer_params import SCAFFOLD_ARGS
from flearn.utils.constants import CLASSES

from flearn.trainers.server import BaseServer
class SCAFFOLDServer(BaseServer):
    def __init__(self, params):
        print('Using MOON to Train')

        params.update(SCAFFOLD_ARGS)
        super().__init__(params)
        self.clients: list[SCAFFOLDClient] = self.clients
        self.global_lr = 1.0
        self.num_classes = CLASSES[self.dataset]                 
        self.global_params_dict = OrderedDict({k: v.clone().detach() for (k, v) in self.client_model.named_parameters()})
        self.c_global = OrderedDict((key, torch.zeros_like(value, requires_grad=False, device="cpu")) for (key, value) in self.client_model.named_parameters())
        for params in self.global_params_dict.values():
            params.requires_grad = False

        # self.global_test = True
        # Set attributes for all clients
        for client in self.clients:
            SCAFFOLD_ARGS["c_local"] = deepcopy(self.c_global)
            client.init_client_specific_params(**SCAFFOLD_ARGS)

    def train(self):
        """Train using Federated MOON"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)            
            if self.loss_converged: break

            selected_clients: list[SCAFFOLDClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            # buffer for receiving client solutions
            y_delta_cache = []
            c_delta_cache = []
            client_solutions_dict = {}

            for _, c in enumerate(selected_clients):  # simply drop the slow devices

                # solve minimization locally
                soln, stats, (y_delta, c_delta) = c.solve_inner_scaffold(
                    c_global=self.c_global, global_parameters=self.global_params_dict, num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                # gather solutions from client
                y_delta_cache.append(y_delta.values())
                c_delta_cache.append(c_delta.values())
                client_solutions_dict[c] = soln[1]

            # update models
            self.aggregate(y_delta_cache, c_delta_cache)            
            if self.ddpg_aggregation:
                self.round = i
                client_solutions_dict[len(self.clients)] = OrderedDict({k: v.clone().detach() for (k, v) in self.client_model.state_dict().items()})
                self.aggregated_params_dict = self.ddpg_aggregate(client_solutions_dict)
                print(f"Last accuracy: {self.last_acc}")
                self.client_model.load_state_dict(self.aggregated_params_dict)
                self.global_params_dict = OrderedDict({k: v.clone().detach() for (k, v) in self.client_model.named_parameters()})
            self.client_model.load_state_dict(self.global_params_dict, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: SCAFFOLDClient):
        client.set_model_params(self.global_params_dict)

    @torch.no_grad()
    def aggregate(
        self,
        y_delta_cache: List[List[torch.Tensor]],
        c_delta_cache: List[List[torch.Tensor]],
    ):
        for param, y_delta in zip(
            self.global_params_dict.values(), zip(*y_delta_cache)
        ):
            if y_delta[0].numel()==1:
                mean_value = int(sum(y_delta)/len(y_delta))
                mean_value = torch.tensor(mean_value)
            else:
                mean_value = torch.sum(1/len(y_delta) * torch.stack(y_delta, dim=-1), dim=-1)
            param.data += (self.global_lr * mean_value.data).to(param.data.dtype)
        # update global control
        for k, c_delta in zip(self.c_global.keys(), zip(*c_delta_cache)):
            self.c_global[k] = self.c_global[k] + (torch.stack(c_delta, dim=-1).sum(dim=-1) / len(self.clients)).to(self.c_global[k].data.dtype)
        self.client_model.load_state_dict(self.global_params_dict, strict=False)