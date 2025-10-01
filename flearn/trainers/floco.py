from collections import OrderedDict

import numpy as np
from flearn.trainers.server import BaseServer
from flearn.config.trainer_params import FLOCO_ARGS
from flearn.clients.floco import FlocoClient
from tqdm import trange
import torch
from sklearn import decomposition


class FlocoServer(BaseServer):
    def __init__(self, params: dict):
        print("Using Floco to Train")
        # FLOCO_ARGS["tau"] = params["num_rounds"]-5
        params.update(FLOCO_ARGS)
        super().__init__(params)
        self.clients: list[FlocoClient] = self.clients
        self.projected_clients = None

        for client in self.clients:
            client.init_client_specific_params(**FLOCO_ARGS)
        print("Tau", self.tau)
    def train(self):
        """Train the model using the Floco algorithm."""
        print("Training with {} workers ---".format(self.clients_per_round))  # type: ignore

        for round_idx in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(round_idx, self.set_client_model_test)
            # if self.loss_converged:
            #     break

            selected_clients: list[FlocoClient] = self.select_clients(
                round_idx, num_clients=min(self.clients_per_round, len(self.clients))
            )

            for i in range(len(selected_clients)):
                selected_clients[i].set_parameters(
                    self.get_subregion_parameters(selected_clients[i].id) # check it
                )

            # print(self.tau, round_idx)
            if self.tau == round_idx:  # type: ignore
                print("Projecting gradients...")
                for _, c in enumerate(self.clients):
                    c.set_model_params(self.latest_model)

                    soln, stats = c.solve_inner(
                        num_epochs=self.num_epochs, batch_size=self.batch_size
                    )
                self.projected_clients = project_clients(
                    self.clients,
                    endpoints=self.endpoints, # type: ignore
                    return_diff=True,
                    global_model_params=self.latest_model
                )

            csolns = []
            client_solutions_dict = {}
            for _, c in enumerate(selected_clients):
                # communicate the latest model
                # c.set_model_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                # gather solutions from client
                csolns.append(soln)
                client_solutions_dict[c.id] = soln[1]

            # update models
            # self.latest_model = self.aggregate(csolns)
            if self.ddpg_aggregation:
                client_solutions_dict[len(self.clients)] = self.latest_model
                self.round = i
                self.latest_model = self.ddpg_aggregate(client_solutions_dict)
                print(f"Last accuracy: {self.last_acc}")
            else: self.latest_model = self.aggregate(csolns)
        self.eval_end()

    def get_subregion_parameters(self, client_id: int):
        return (
            None
            if self.projected_clients is None
            else (self.projected_clients[client_id], self.rho)
        )

    def set_client_model_test(self, client: FlocoClient):
        client.set_model_params(self.latest_model)

    def aggregate(self, wsolns):  # Weighted average using PyTorch
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


def project_clients(clients, endpoints, return_diff, global_model_params=None):
    """
    Project clients onto simplex using classifier gradients or differences from global model.
    
    Args:
        client_packages: Client training results
        endpoints: Number of simplex endpoints
        return_diff: Whether client packages contain diffs
        global_model_params: Global model parameters for computing differences
    """
    gradient_dict = {client.id: None for client in clients}
    
    for client in clients:
        # print(f"client_id: {client.id}, package: {package}")
        client_params = OrderedDict( {k: v for (k,v) in client.model.state_dict().items()})
        if global_model_params is not None:
            # Compute differences from global model
            gradients = []
            for k, v in client_params.items():
                if "fc._weights" in k and k in global_model_params:
                    diff = global_model_params[k] - v  # global - client
                    gradients.append(diff.cpu().numpy().flatten())
        else:
            # Use raw client parameters (fallback)
            gradients = [
                v.cpu().numpy().flatten()
                for k, v in client_params.items()
                if "fc._weights" in k
            ]
        
        gradient_dict[client.id] = np.concatenate(gradients)
    
    # Rest of the function remains the same
    client_stats = np.array(list(gradient_dict.values()))
    kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)
    
    # Find optimal projection
    lowest_log_energy = np.inf
    best_beta = None
    for i, z in enumerate(np.linspace(1e-4, 1, 1000)):
        betas = _project_client_onto_simplex(kappas, z=z)
        betas /= betas.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas
    return best_beta

def _project_client_onto_simplex(kappas, z):
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas

def _riesz_s_energy(simplex_points):
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow during gradient calculation
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / mutual_dist**2
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy