"""
FedRS Trainer Implementation
Paper: "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data"
Authors: Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng
Venue: KDD 2021
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedrs import FedRSClient


class FedRSServer(BaseServer):
    def __init__(self, params):
        print("Using FedRS (Restricted Softmax) to Train")
        super().__init__(params)
        
        # Initialize client-specific local classes
        self._initialize_client_local_classes()
    
    def _initialize_client_local_classes(self):
        """
        Extract and set local classes for each client
        This is done once at initialization
        """
        print("Initializing local classes for each client...")
        for client in self.clients:
            if isinstance(client, FedRSClient):
                # Extract unique classes from client's training data
                local_classes = client.get_local_classes_from_data()
                client.init_client_specific_params(local_classes=local_classes)
        print("Local classes initialized for all clients.")
    
    def train(self):
        """Train using FedRS with Restricted Softmax"""
        print("Training with FedRS - {} workers per round ---".format(self.clients_per_round))
        
        for i in trange(self.num_rounds, desc=self.desc):
            # Test model
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break
            
            # Select clients
            selected_clients: list[FedRSClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )
            
            csolns = []  # buffer for receiving client solutions
            
            for _, c in enumerate(selected_clients):
                # Communicate the latest model
                c.set_model_params(self.latest_model)
                
                # Solve minimization locally with restricted softmax
                soln, stats = c.solve_inner_fedrs(
                    num_epochs=self.num_epochs, 
                    batch_size=self.batch_size
                )
                
                # Gather solutions from client
                csolns.append(soln)
            
            # Update models using standard FedAvg aggregation
            self.latest_model = self.aggregate(csolns)
            self.client_model.load_state_dict(self.latest_model, strict=False)
        
        self.eval_end()
    
    def set_client_model_test(self, client: FedRSClient):
        """Set model parameters for testing"""
        client.set_model_params(self.latest_model)
    
    def aggregate(self, wsolns):
        """
        Standard weighted average aggregation (FedAvg)
        FedRS uses restricted softmax on clients but standard aggregation on server
        
        Args:
            wsolns: List of tuples (num_samples, state_dict) from clients
            
        Returns:
            averaged_state_dict: Aggregated model parameters
        """
        total_weight = 0.0
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
