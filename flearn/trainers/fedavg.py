import torch
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedavg import FedAvgClient
from flearn.utils.constants import CLASSES


class FedAvgServer(BaseServer):
    def __init__(self, params):
        print("Using Federated avg to Train")
        params['max_rl_steps'] = 100
        super().__init__(params)
        self.num_classes = CLASSES[self.dataset]
        for client in self.clients:
            client.num_classes = self.num_classes

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

            csolns = []  # buffer for receiving client solutions

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_model_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                # gather solutions from client
                csolns.append(soln)

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.load_state_dict(self.latest_model, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: FedAvgClient):
        client.set_model_params(self.latest_model)

    def aggregate(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors
        # Initialize base with zeros tensors with the same size as the first solution's parameters'
        model_state_dict: OrderedDict = wsolns[0][1]
        base = [torch.zeros_like(soln) for soln in model_state_dict.values()]

        for w, client_state_dict in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v

        # Divide each aggregated tensor by the total weight to compute the average
        averaged_soln = [v / total_weight for v in base]
        averaged_state_dict = OrderedDict(zip(model_state_dict.keys(), averaged_soln))

        return averaged_state_dict
