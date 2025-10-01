import torch
from torch.nn.utils import clip_grad_norm_
from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
from collections import OrderedDict
from copy import deepcopy


class PeFLLClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_client_specific_params(
        self,
        embed_dim: int,
        embed_y: int,
        embed_num_kernels: int,
        embed_num_batches: int,
        hyper_embed_lr: float,
        hyper_hidden_dim: int,
        hyper_num_hidden_layers: int,
        clip_norm: float,
        embed_net: torch.nn.Module,
        hyper_net: torch.nn.Module,
    ):

        self.embed_dim = embed_dim
        self.embed_y = embed_y
        self.embed_num_kernels = embed_num_kernels
        self.embed_num_batches = embed_num_batches
        self.hyper_embed_lr = hyper_embed_lr
        self.hyper_hidden_dim = hyper_hidden_dim
        self.hyper_num_hidden_layers = hyper_num_hidden_layers
        self.clip_norm = clip_norm

        self.embed_net = deepcopy(embed_net).to(self.device)
        self.hyper_net = deepcopy(hyper_net).to(self.device)
        self.embed_hyper_optm = torch.optim.Adam(
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            lr=hyper_embed_lr,
        )
        self.embed_net.to("cpu")
        self.hyper_net.to("cpu")

    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """
        self.embed_net.to(self.device)
        self.hyper_net.to(self.device)

        model_params = self.update_model()

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                clip_grad_norm_(self.model.parameters(), self.clip_norm)

                self.optimizer.step()
                train_sample_size += len(labels)

        

        sol_embed, sol_hyper = self.update_embed_hyper(model_params)        
        
        self.embed_net.to("cpu")
        self.hyper_net.to("cpu")
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, (sol_embed, sol_hyper)), (bytes_w, comp, bytes_r)

    def get_batch_embedding(self):
        embedding = torch.zeros(self.embed_dim, device=self.device)
        size = 0

        for i, (x, y) in enumerate(self.trainloader):
            x, y = x.to(self.device), y.to(self.device)
            embedding += self.embed_net(x, y).sum(dim=0)
            size += len(x)
            if i + 1 == self.embed_num_batches:
                break
        embedding /= size
        embedding = (embedding - embedding.mean()) / embedding.std()
        return embedding

    def set_embed_hyper(
        self,
        server_embed_net: torch.nn.Module,
        server_hyper_net: torch.nn.Module,
        update_client_model: bool = False,
    ):
        self.embed_net.load_state_dict(server_embed_net.state_dict())
        self.hyper_net.load_state_dict(server_hyper_net.state_dict())

        if update_client_model:
            self.update_model()

    def update_model(self) -> OrderedDict:
        embedding_features = self.get_batch_embedding()
        model_params = self.hyper_net(embedding_features)
        # print(model_params.keys())
        # for k, _ in self.model.named_parameters():
        #     print(k)
        self.model.load_state_dict(model_params, strict=False)
        return model_params

    def update_embed_hyper(self, model_params) -> tuple[list, list]:
        num_embed_net_params = len(list(self.embed_net.parameters()))

        joint_grads = torch.autograd.grad(
            outputs=list(model_params.values()),
            inputs=list(self.embed_net.parameters())
            + list(self.hyper_net.parameters()),
            grad_outputs=[
                (param_old - param_new).detach()
                for param_new, param_old in zip(
                    self.model.parameters(), model_params.values()
                )
                if param_new.requires_grad
            ],
            allow_unused=True,
        )

        sol_embed = list(joint_grads[:num_embed_net_params])
        sol_hyper = list(joint_grads[num_embed_net_params:])
        print(type(sol_embed))

        return sol_embed, sol_hyper
