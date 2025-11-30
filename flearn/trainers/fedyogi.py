
import torch
from collections import OrderedDict
from flearn.trainers.server import BaseServer
from flearn.clients.fedavg import FedAvgClient
from flearn.utils.constants import CLASSES
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader


class FedYogiServer(BaseServer):
    def __init__(self, params):
        print("Using FedYogi (Adaptive Federated Optimization with Yogi) to Train")
        params["server_learning_rate"] = 0.0001  # ensure it's in params
        # FedYogi hyperparameters
        self.server_lr = params.get('server_learning_rate', 0.01)  # η
        self.beta1 = params.get('beta1', 0.9)
        self.beta2 = params.get('beta2', 0.99)
        self.tau = params.get('tau', 1e-3)
        self.eps = params.get('eps', 1e-12)

        super().__init__(params)

        self.num_classes = CLASSES[self.dataset]
        self.clients: list[FedAvgClient] = self.clients

        # Optimizer state
        self.m = None
        self.v = None
        self.t = 0

        for client in self.clients:
            client.num_classes = self.num_classes

    def train(self):
        print(f"Training with {self.clients_per_round} workers using FedYogi")
        print(f"Server LR: {self.server_lr}, Beta1: {self.beta1}, Beta2: {self.beta2}, Tau: {self.tau}")

        for i in trange(self.num_rounds, desc=self.desc):
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break

            selected_clients = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )

            csolns = []
            for c in tqdm(selected_clients, desc="Training Clients", leave=False):
                c.round = i
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )
                csolns.append(soln)

            self.latest_model = self.aggregate_yogi(csolns)
            self.client_model.load_state_dict(self.latest_model)

        self.eval_end()

    def set_client_model_test(self, client):
        client.model.load_state_dict(self.client_model.state_dict())

    @torch.no_grad()
    def aggregate_yogi(self, wsolns):
        self.t += 1

        device = next(self.client_model.parameters()).device
        dtype = next(self.client_model.parameters()).dtype
        current_state = self.client_model.state_dict()

        # Compute weighted average model delta Δ = Σ p_i (x_i - x)
        Delta = OrderedDict({k: torch.zeros_like(v, device=device, dtype=dtype)
                             for k, v in current_state.items()})
        total_weight = 0

        for w, client_state in wsolns:
            w = float(w or 1.0)
            total_weight += w
            for k in Delta:
                Delta[k] += w * (client_state[k].to(device, dtype) - current_state[k])

        if total_weight == 0:
            return OrderedDict({k: v.clone() for k, v in current_state.items()})

        for k in Delta:
            Delta[k] /= total_weight

        # Pseudo-gradient g_t = -Δ
        g = OrderedDict({k: -Delta[k] for k in Delta})

        # Init optimizer states
        if self.m is None:
            self.m = OrderedDict({k: torch.zeros_like(v) for k, v in g.items()})
            self.v = OrderedDict({k: torch.zeros_like(v) for k, v in g.items()})

        # Yogi update
        for k in g:
            grad = g[k]

            # 1st moment
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad

            # 2nd moment (Yogi)
            grad_sq = grad * grad
            v_prev = self.v[k]
            self.v[k] = v_prev - (1 - self.beta2) * grad_sq * torch.sign(v_prev - grad_sq)

        # Bias correction
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        # Update parameters
        new_state = OrderedDict()
        for k in current_state:
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            denom = torch.sqrt(v_hat) + self.tau + self.eps
            step = self.server_lr * (m_hat / denom)
            new_state[k] = (current_state[k] - step).clone()

        return new_state
