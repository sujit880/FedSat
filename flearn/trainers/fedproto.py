import torch
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from typing import List
from copy import deepcopy
from collections import OrderedDict
from flearn.clients.fedproto import ProtoClient
from flearn.config.trainer_params import PROTO_ARGS
from flearn.utils.constants import CLASSES, LAMDA
from flearn.trainers.server import BaseServer
from flearn.config.config_paths import (
    DUMP_JSON_RESULT_PATH,
    TEST_STATS_FILE_NAME,
    TEST_ALL_STATS_FILE_NAME,
    FIGURES_PATH,
)

class ProtoServer(BaseServer):
    def __init__(self, params):
        print('Using FedProto to Train')

        params.update(PROTO_ARGS)
        super().__init__(params)
        self.experiment_name = self.experiment_name
        self.global_protos = []
        self.clients: list[ProtoClient] = self.clients
        self.global_lr = 1.0
        self.num_classes = CLASSES[self.dataset]                 
        self.global_params_dict = OrderedDict({k: v.clone().detach() for (k, v) in self.client_model.named_parameters()})
        self.c_global = OrderedDict((key, torch.zeros_like(value, requires_grad=False, device="cpu")) for (key, value) in self.client_model.named_parameters())
        for params in self.global_params_dict.values():
            params.requires_grad = False

        PROTO_ARGS["lamda"] = 0.5 #LAMDA[self.dataset]
        PROTO_ARGS["num_classes"] = self.num_classes        

        # Set attributes for all clients
        for client in self.clients:
            client.init_client_specific_params(**PROTO_ARGS)

    def train(self):
        """Train using Federated FedProto"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)
            # if self.loss_converged: break

            selected_clients: list[ProtoClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            # buffer for receiving client solutions
            cprotos = {}
            csolns = []

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                soln, stats, protos = c.solve_inner_fedproto_t(global_protos= self.global_protos, 
                                                    num_epochs=self.num_epochs, 
                                                    batch_size=self.batch_size
                            )
                agg_protos = self.agg_func_protos_t(protos = protos)
                cprotos[c]  = agg_protos
                csolns.append(soln)

            # update models
            # self.latest_model = self.aggregate(csolns)
            self.global_protos = self.proto_aggregation(cprotos)

        self.eval_end()

    def set_client_model_test(self, client: ProtoClient):
        # client.set_model_params(self.latest_model)
        pass # Parameters not updated in this approach

    @torch.no_grad()
    def agg_func_protos_t(self, protos: dict):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos
    
    @torch.no_grad()
    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for key in local_protos_list:
            local_protos = local_protos_list[key]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label
    
    @torch.no_grad()
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

    
    def test_model_(self, selected_clients: list[ProtoClient], modelInCPU: bool = False):
        """Tests custom"""
        print(f'Custom proto test.')
        num_samples = []
        tot_correct = []
        for c1 in self.clients:
            for c in selected_clients:
                ct, ns = c.test_proto(model=c1.model, prototypes=self.global_protos, modelInCPU=modelInCPU)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct