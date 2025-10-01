import torch
from tqdm import trange
from typing import List
from copy import deepcopy
from collections import OrderedDict

from flearn.trainers.server import BaseServer
from flearn.clients.ditto import DittoClient
from flearn.utils.constants import CLASSES

# Fallback in case you have not added DITTO_ARGS to flearn.config.trainer_params yet
try:
    from flearn.config.trainer_params import DITTO_ARGS  # e.g. {"lam": 0.01}
except ImportError:
    DITTO_ARGS = {"lam": 0.01}


class FedDittoServer(BaseServer):
    """Federated Ditto server implementation.

    The server keeps a global model that is sent to each selected client
    at the beginning of every communication round.  Each client trains
    a personalised model with a proximal regularisation term to the
    global model.  The server aggregates the personalised models with
    a weighted average (FedAvg style) to update the global model.
    """

    def __init__(self, params):
        print("Using Ditto to Train")

        # expose Ditto‑specific hyper‑parameters to the trainer
        params.update(DITTO_ARGS)
        super().__init__(params)

        # type hint for IDEs / linters
        self.clients: List[DittoClient] = self.clients
        self.num_classes = CLASSES[self.dataset]

        # Propagate Ditto specific parameters to every client
        for client in self.clients:
            client.init_client_specific_params(lam=DITTO_ARGS["lam"])

    # ------------------------------------------------------------------
    # Training loop ----------------------------------------------------
    # ------------------------------------------------------------------
    def train(self):
        print(f"Training with {self.clients_per_round} workers ---")

        for i in trange(self.num_rounds, desc=self.desc):
            # Evaluate current global model ---------------------------------
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break

            # Select participants ------------------------------------------
            selected_clients: List[DittoClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )

            # Collect local solutions --------------------------------------
            csolns = []  # [(num_samples, state_dict), ...]
            client_solutions_dict = {}
            for client in selected_clients:
                # Send the latest global model
                client.set_model_params(self.latest_model)

                # Local training ------------------------------------------------
                soln, stats = client.solve_inner_ditto(
                    global_model=self.client_model,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                )

                # Gather --------------------------------------------------------
                csolns.append(soln)
                client_solutions_dict[client] = soln[1]

            # Aggregate to update the global model --------------------------
            self.latest_model = self.aggregate(csolns)

            # Optional: RL‑based aggregation --------------------------------
            if self.ddpg_aggregation:
                self.round = i
                client_solutions_dict[len(self.clients)] = self.latest_model
                self.latest_model = self.ddpg_aggregate(client_solutions_dict)
                print(f"Last accuracy: {self.last_acc}")

            # Broadcast weights to the BaseServer's reference model --------
            self.client_model.load_state_dict(self.latest_model)

        # Final evaluation --------------------------------------------------
        self.eval_end()

    # ------------------------------------------------------------------
    # Helpers -----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_client_model_test(self, client: DittoClient):
        """Utility for the evaluation routine – just load the weights."""
        client.set_model_params(self.latest_model)

    def aggregate(self, wsolns):
        """Weighted FedAvg aggregation implemented with PyTorch tensors."""
        total_weight = 0.0
        first_state_dict = wsolns[0][1]
        base = [torch.zeros_like(p) for p in first_state_dict.values()]

        for weight, client_state_dict in wsolns:
            total_weight += weight
            for idx, param_tensor in enumerate(client_state_dict.values()):
                base[idx] += weight * param_tensor

        averaged_tensors = [p / total_weight for p in base]
        averaged_state_dict = OrderedDict(zip(first_state_dict.keys(), averaged_tensors))
        return averaged_state_dict
