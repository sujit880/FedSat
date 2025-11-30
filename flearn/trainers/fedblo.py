# fedmap\_server.py (minimal structural update)
import torch
from collections import OrderedDict
from copy import deepcopy
from typing import List
from tqdm import trange

from flearn.clients.fedblo import FedBLOClient
from flearn.config.trainer_params import BLO_ARGS
from flearn.utils.constants import CLASSES, FedSatL_MU
from flearn.trainers.server import BaseServer


class FedBLOServer(BaseServer):
    def __init__(self, params):
        print('Using FedMAP to Train')
        
        params['endpoints'] = 20
        BLO_ARGS['lamda'] = 0.5
        BLO_ARGS['lamdav'] = 1.0
        BLO_ARGS['con_tau'] = 0.5        
        BLO_ARGS['con_mu'] = 0.1 
        BLO_ARGS['sprox_mu'] = 0.1
        BLO_ARGS['max_rl_steps'] = 100   
        params.update(BLO_ARGS)
        
        super().__init__(params)

        self.clients: list[FedBLOClient] = self.clients
        self.global_lr = 1.0
        self.num_classes = CLASSES[self.dataset]

        # RL feedback cache (filled when RL agg exposes weights; defaults to uniform)
        self.last_client_weights = {}

        # Initialize global model parameters
        self.global_params_dict = OrderedDict(self.client_model.state_dict())
        for param in self.global_params_dict.values():
            param.requires_grad = False

        # Initialize control variates (c_global) for SCAFFOLD
        self.c_global = OrderedDict(
            (key, torch.zeros_like(value, requires_grad=False, device="cpu"))
            for key, value in self.client_model.named_parameters()
        )

        # Initialize client parameters and learning rate schedulers
        for client in self.clients:
            client.lamda = self.lamda
            client.lamdav = self.lamdav
            BLO_ARGS.update({
                "c_global": deepcopy(self.c_global),
                "c_local": deepcopy(self.c_global),
                "tau": self.con_tau,
                "mu": self.con_mu,
                "cost_matrix": torch.ones(self.num_classes, self.num_classes, device=self.device),
                "clients_per_round": min(self.clients_per_round, len(self.clients)),
                "total_clients": len(self.clients),
                "prev_model": deepcopy(self.client_model),
                # === new knobs (safe defaults) for policy-conditioned local subgoal ===
                "temperature": 2.0,
                "lambda_rep": 0.5,
                "lambda_distill": 0.5,
                "lambda_prox": self.sprox_mu,
                "lambda_contrast": 0.5,
                "lambda_fair": 0.1,
                "trust_layers_prefix": ("layer3", "layer4", "fc", "resnet.fc"),
            })
            client.init_client_specific_params(**BLO_ARGS)
            client.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client.optimizer, self.num_rounds
            )

    def train(self):
        """Train the global model using federated learning rounds."""
        print(f"Training with {self.clients_per_round} clients per round.")

        for round_idx in trange(self.num_rounds, desc=self.desc):
            self.eval(round_idx, self.set_client_model_test)
            self.round = round_idx
            if self.loss_converged:
                break

            selected_clients: list[FedBLOClient] = self.select_clients(
                round_idx, num_clients=min(self.clients_per_round, len(self.clients))
            )

            # === provide RL feedback to clients (minimal, weight only) ===
            # prefer weights produced by RL aggregator in the previous round if available
            weight_fallback = 1.0
            if not self.last_client_weights:
                # initialize uniform
                for c in selected_clients:
                    self.last_client_weights[c.id] = weight_fallback

            for c in selected_clients:
                w = float(self.last_client_weights.get(c.id, weight_fallback))
                c.set_policy_feedback({"w": w, "adv": 0.0})

            client_solutions = []
            client_solutions_dict = {}

            for client in selected_clients:
                if self.loss == "CSN":
                    client.criterion.cost_reset(cost_matrix=client.test_and_cost_matrix_stats_t())

                client.set_model_params(self.global_params_dict)
                client.round = round_idx

                stats, soln = client.solve_inner_fedmap(
                    self.global_params_dict,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )

                client_solutions.append(soln)
                client_solutions_dict[client.id] = soln[1]

            # Aggregate global model parameters from client solutions
            if self.ddpg_aggregation:
                self.round = round_idx
                self.global_params_dict = self.drl_aggregate(client_solutions_dict)
                print(f"Last accuracy: {self.last_acc}")
                # If RL aggregator exposes per-client weights, store them for next round
                if hasattr(self, 'rl_client_weights') and isinstance(self.rl_client_weights, dict):
                    self.last_client_weights = {k: float(v) for k, v in self.rl_client_weights.items()}
            else:
                self.global_params_dict = self.aggregate(client_solutions)
                # fall back to uniform weights for next round
                self.last_client_weights = {c.id: 1.0 for c in selected_clients}

            self.client_model.load_state_dict(self.global_params_dict, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: FedBLOClient):
        client.set_model_params(self.global_params_dict)

    @torch.no_grad()
    def aggregate(self, client_solutions: List[OrderedDict]):
        total_weight = 0.0
        model_state_dict: OrderedDict = client_solutions[0][1]
        base = [torch.zeros_like(soln) for soln in model_state_dict.values()]

        for w, client_state_dict in client_solutions:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v

        averaged_soln = [v / max(total_weight, 1e-8) for v in base]
        averaged_state_dict = OrderedDict(zip(model_state_dict.keys(), averaged_soln))
        return averaged_state_dict
    
    
