import torch
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from typing import List
from copy import deepcopy
from collections import OrderedDict
from flearn.clients.moon import MOONClient
from flearn.config.trainer_params import MOON_ARGS
from flearn.config.config_paths import RECONST_OUTPUT_DIR
from flearn.utils.constants import CLASSES, MOON_MU

from flearn.trainers.server import BaseServer
class FedMOONServer(BaseServer):
    def __init__(self, params):
        print('Using MOON to Train')
        
        MOON_ARGS["mu"] = 5.0 #MOON_MU[self.dataset]
        MOON_ARGS["tau"] = 0.5
        MOON_ARGS["prev_model_version"] = input("Enter previous model version (v1 or v2): ")
        if MOON_ARGS["prev_model_version"] == "v1": 
            print("Upgrading from v1: Previous model is the client's last model")
        elif MOON_ARGS["prev_model_version"] == "v2": 
            print("Upgrading from v2: Previous model is the global model")
        else: 
            print("Invalid input. Defaulting to v1.")
        params.update(MOON_ARGS)
        super().__init__(params)
        self.clients: list[MOONClient] = self.clients
        self.num_classes = CLASSES[self.dataset]
        

        # Set attributes for all clients
        for client in self.clients:
            MOON_ARGS["prev_model"] = deepcopy(self.client_model)
            client.init_client_specific_params(**MOON_ARGS)
            client.prev_model_version = self.prev_model_version
                             
        # self.robust_test = True
        # self.global_test = True
        
    def train(self):
        """Train using Federated MOON"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)           
            if self.loss_converged: break            

            selected_clients: list[MOONClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            csolns = []  # buffer for receiving client solutions
            client_solutions_dict = {}

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_model_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner_moon_t(
                    global_model=self.client_model, num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                # gather solutions from client
                csolns.append(soln)
                client_solutions_dict[c] = soln[1]

            # update models
            self.latest_model = self.aggregate(csolns)            
            if self.ddpg_aggregation:
                self.round = i
                client_solutions_dict[len(self.clients)] = self.latest_model
                self.latest_model = self.ddpg_aggregate(client_solutions_dict)
                print(f"Last accuracy: {self.last_acc}")
            self.client_model.load_state_dict(self.latest_model, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: MOONClient):
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