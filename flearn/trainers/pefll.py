import torch
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from flearn.models.model import (
    PeFLL_EmbedNetwork,
    PeFLL_HyperNetwork,
    CHANNELS,
    CLASSES,
)
from flearn.config.trainer_params import PeFLL_ARGS
from flearn.clients.pefll import PeFLLClient


class PeFLLServer(BaseServer):
    def __init__(self, params: dict):
        print("Using PeFLL to Train")

        params.update(PeFLL_ARGS)
        super().__init__(params)

        self.client_num = len(self.clients)
        self.embed_dim = int(1 + self.client_num / 4)
        PeFLL_ARGS["embed_dim"] = self.embed_dim

        self.embed_net = PeFLL_EmbedNetwork(
            input_channels=CHANNELS[self.dataset],
            num_classes=CLASSES[self.dataset],
            embed_dim=self.embed_dim,
            embed_y=self.embed_y,
            embed_num_kernels=self.embed_num_kernels,
        )

        self.hyper_net = PeFLL_HyperNetwork(
            backbone=self.client_model,
            embed_dim=self.embed_dim,
            hyper_hidden_dim=self.hyper_hidden_dim,
            hyper_num_hidden_layers=self.hyper_num_hidden_layers,
        )

        self.embed_hyper_optimizer = torch.optim.Adam(
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            lr=self.hyper_embed_lr,
        )

        self.init_client_specific_params()

        self.robust_test = True

    def init_client_specific_params(self):

        PeFLL_ARGS["hyper_net"] = self.hyper_net
        PeFLL_ARGS["embed_net"] = self.embed_net

        for client in self.clients:
            client.init_client_specific_params(**PeFLL_ARGS)

    def train(self):
        """Train using Federated Proximal"""
        print("Training with {} workers ---".format(self.clients_per_round))
        desc = f"Algo: {self.trainer}, Round: "
        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)

            selected_clients: list[PeFLLClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            # buffer for receiving client solutions
            csolns_embed_grads = []
            csolns_hyper_grads = []

            for  c in selected_clients:
                # communicate the latest model
                c.set_embed_hyper(
                    server_embed_net=self.embed_net,
                    server_hyper_net=self.hyper_net,
                )

                # solve minimization locally
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                if soln[1][0] is None and soln[1][1] is None:
                    raise RuntimeError("Clients solution is None, Aborting execution.")
                # else: 
                #     # print(soln[1][0])
                #     print(soln[1][0].shape)

                # gather solutions from client
                csolns_embed_grads.append(soln[1][0])
                csolns_hyper_grads.append(soln[1][1])

            # update models
            self.aggregate(csolns_embed_grads, csolns_hyper_grads)

        # final test model
        self.eval_end()

    def set_client_model_test(
        self,
        client: PeFLLClient,
    ):
        client.set_embed_hyper(
            server_embed_net=self.embed_net,
            server_hyper_net=self.hyper_net,
            update_client_model=True,
        )

    @torch.no_grad()
    def aggregate(self, csolns_embed_grads, csolns_hyper_grads):

        # print("csolns_embed_grads:", csolns_embed_grads)
        # print("csolns_hyper_grads:", csolns_hyper_grads)

        self.embed_hyper_optimizer.zero_grad()

        for param, grads in zip(self.embed_net.parameters(), zip(*csolns_embed_grads)):
            if grads is None:
                print(grads)
                raise RuntimeError("Clients grads is None, Aborting execution.")
            else:
                for grad in grads:
                    if grad is None:
                        print(grad)
                        raise RuntimeError("Clients grad is None, Aborting execution.")
                    else: print(f"Tensor with shape {grad.shape}")
            grad = torch.stack(grads, dim=0).mean(dim=0).to(param.device).type_as(param)
            param.grad = grad

        for param, grads in zip(self.hyper_net.parameters(), zip(*csolns_hyper_grads)):
            grad = torch.stack(grads, dim=0).mean(dim=0).to(param.device).type_as(param)
            param.grad = grad

        self.embed_hyper_optimizer.step()
