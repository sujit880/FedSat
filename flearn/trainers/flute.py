import torch
import numpy as np
import math
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from flearn.models.model import CLASSES
from flearn.config.trainer_params import FLUTE_ARGS
from flearn.clients.flute import FLUTEClient
from copy import deepcopy
from collections import OrderedDict
from flearn.models.generative import LCLS


class FLUTEServer(BaseServer):
    def __init__(self, params):
        print("Using FLUTE")

        params.update(FLUTE_ARGS)
        super().__init__(params)

        self.clients: list[FLUTEClient] = self.clients

        self.num_classes = CLASSES[self.dataset]
        self.feature_dim = self.client_model.get_feature_dim()
        self.client_num = len(self.clients)

        self.global_classifier = LCLS(input_dim=self.feature_dim, num_classes=self.num_classes).to(self.device)
        self.global_cls_dict =OrderedDict({k: v.clone().detach() for k, v in self.global_classifier.named_parameters()})
        for (k, v) in self.client_model.state_dict().items():
            if k.startswith('fc.') or k.startswith('resnet.fc.') or k.startswith('linear.') or k.startswith('resnet.linear.'):
                v.requires_grad = False
        
        self.init_client_specific_params()

    def init_client_specific_params(self):
        FLUTE_ARGS["num_classes"] = self.num_classes
        FLUTE_ARGS["finetune_epochs"] = 1 # int(self.num_epochs/2)
        FLUTE_ARGS["rep_round"] = int(self.num_epochs/2)
        for client in self.clients:
            FLUTE_ARGS.update({
                "classifier": deepcopy(self.global_classifier)
            })
            for (k, v) in client.model.state_dict().items():
                if k.startswith('fc.') or k.startswith('resnet.fc.') or k.startswith('linear.') or k.startswith('resnet.linear.'):
                    v.requires_grad = False
            client.init_client_specific_params(**FLUTE_ARGS)
            client.use_custom_classifier = True # This needs to be set true for this algorithm

    def train(self):
        """Train using Federated Proximal"""
        print("Training with {} workers ---".format(self.clients_per_round))
        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.round = i
            self.eval(i, self.set_client_model_test)       
            if self.loss_converged: break

            selected_clients: list[FLUTEClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            csolns = []  # buffer for receiving client solutions

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                # solve minimization locally
                soln, stats = c.flute_localUpdate(
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                )

                # gather solutions from client
                csolns.append(soln)

            # update models
            self.aggregate(csolns)
            for client in self.clients:
                client.model.load_state_dict(self.client_model.state_dict())
            self.update_neural_collapse(selected_clients=self.clients)

        # final test model
        self.eval_end()

    def set_client_model_test(self, client: FLUTEClient):
        # client.model.load_state_dict(self.client_model.state_dict())
        # if self.round%10== 0:
        #     client.finetune_flute(
        #         num_epochs=self.finetune_epochs, batch_size=self.batch_size
        #     )
        pass

    def aggregate(self, client_solns):
        latest_model = self.aggregate_params(client_solns)

        with torch.no_grad():
            self.client_model.load_state_dict(latest_model, strict=False)
            # for param, value in zip(self.client_model.state_dict(), latest_model.values()):
                # param.copy_(value)

    def update_neural_collapse(self, selected_clients: list[FLUTEClient]):
        classifier_weights = (
            torch.stack(
                [
                    client.classifier.cls.weight.clone().detach()
                    for client in selected_clients
                ],
                dim=0,
            )
            .requires_grad_(True)
            .to(self.device)
        )

        optimizer = torch.optim.SGD([classifier_weights], lr=self.nc2_lr)
        loss = 0

        for i, c in enumerate(selected_clients):
            labels = c.data_labels.to(self.device)

            weight_i = classifier_weights[i].to(self.device)
            weight_i_matmul = weight_i @ weight_i.t()
            norm = torch.norm(weight_i_matmul)
            loss += torch.norm(
                weight_i_matmul / norm
                - torch.mul(
                    (
                        torch.eye(self.num_classes).to(self.device)
                        - torch.ones((self.num_classes, self.num_classes)).to(
                            self.device
                        )
                        / self.num_classes
                    ),
                    labels @ labels.t(),
                )
                / math.sqrt(self.num_classes - 1)
            ) / len(selected_clients)

        loss.backward()
        optimizer.step()

        for i, c in enumerate(selected_clients):
            c.classifier.cls.weight.data.copy_(classifier_weights[i])

    @torch.no_grad()
    def aggregate_params(self, wsolns):  # Weighted average using PyTorch
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
