import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.fedavg import FedAvgClient
from flearn.utils.constants import CLASSES
from flearn.utils.aggregator import Aggregator
from copy import deepcopy

TOP_P = {
    "cifar10": 4, # 1, 3, 4, 5, 10
    "cifar": 4, # 1, 3, 4, 5, 10
    "cifar100": 15, # 3, 10, 20, 30 Best=15
    "fmnist": 4,
    "emnist": 10, # 6, 10 Best=10
    "svhn": 3,
    "mnist": 3,
    "femnist": 6,
    "synthetic": 3,
}  # Number of top struggling classes to prioritize
# Make LR aggregation and dataset-based
def get_lr_for_dataset(agg_method, dataset):
    # You can customize learning rates per aggregation method and dataset here
    dataset_lr = {
        "cifar10": {"fedadam": 1e-3, "fedavgm": 0.5},
        "cifar": {"fedadam": 1e-2, "fedavgm": 0.5},
        "cifar100": {"fedadam": 5e-4, "fedavgm": 0.5},
        "mnist": {"fedadam": 1e-2, "fedavgm": 1.0},
        "fmnist": {"fedadam": 1e-2, "fedavgm": 0.5},
        "emnist": {"fedadam": 1e-2, "fedavgm": 1.0},
        "femnist": {"fedadam": 1e-2, "fedavgm": 1.0},
    }
    return dataset_lr.get(dataset, {}).get(agg_method, 0.01)

class FedAvgServer(BaseServer):
    def __init__(self, params):
        print("Using Federated avg to Train")
        params['max_rl_steps'] = 100
        params["server_learning_rate"] = get_lr_for_dataset(params["agg"], params["dataset"]) if params["agg"] in {"fedadam", "fedavgm"} else 1.0
        params["server_momentum"] = 0.99
        params["top_p"] = TOP_P[params['dataset']] # int(CLASSES[params['dataset']]/2.5) # TOP_P[params['dataset']]
        params.update({"use_prev_global_model":  False})
        super().__init__(params)
        self.num_classes = CLASSES[self.dataset]
        for client in self.clients:
            client.num_classes = self.num_classes

        # choose aggregation method
        if self.agg is not None and not self.agg=="drl":
            self.aggregator = Aggregator(method=self.agg, model=self.client_model, lr=self.server_learning_rate, mu=self.server_momentum, top_p=self.top_p)
        # Check and print overall dataset stats (train + val) across all clients
        if hasattr(self, "clients") and self.clients:
            def subset_labels(subset) -> torch.Tensor:
                # subset is a torch.utils.data.Subset wrapping our pickled Dataset
                if hasattr(subset, "dataset") and hasattr(subset, "indices"):
                    base_ds = subset.dataset
                    idx = subset.indices
                    targets = getattr(base_ds, "targets", None)
                    if isinstance(targets, torch.Tensor):
                        return targets[idx].clone().detach().cpu().long()
                    elif isinstance(targets, (list, tuple, np.ndarray)):
                        return torch.as_tensor([targets[i] for i in idx], dtype=torch.long)
                # Fallback: iterate to collect labels (may miss last batch if drop_last=True)
                return torch.cat([y.detach().cpu().long() for _, y in DataLoader(subset, batch_size=256)], dim=0)

            total_train_samples = 0
            total_val_samples = 0
            train_class_counts = np.zeros(self.num_classes, dtype=np.int64)
            val_class_counts = np.zeros(self.num_classes, dtype=np.int64)

            for client in self.clients:
                # Accurate counts from underlying Subset lengths (not affected by drop_last)
                train_subset = client.trainloader.dataset
                val_subset = client.valloader.dataset
                total_train_samples += len(train_subset)
                total_val_samples += len(val_subset)

                # Per-class histogram for train
                tl = subset_labels(train_subset)
                binc = torch.bincount(tl, minlength=self.num_classes).numpy()
                train_class_counts += binc

                # Per-class histogram for val
                vl = subset_labels(val_subset)
                vbinc = torch.bincount(vl, minlength=self.num_classes).numpy()
                val_class_counts += vbinc

            print(f"Total training samples across all clients: {int(total_train_samples)}")
            print(f"Total validation samples across all clients: {int(total_val_samples)}")
            print(f"Train class distribution (global): {train_class_counts.tolist()}")
            print(f"Val class distribution (global):   {val_class_counts.tolist()}")

    def train(self):
        """Train using Federated Averaging"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)           
            if self.loss_converged: break

            selected_clients: list[FedAvgClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            csolns = []  # buffer for receiving client solutions
            client_solutions_dict = {}

            for _, c in enumerate(selected_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_model_params(self.latest_model)
                if self.agg == "elastic":
                    sensitivity = c.grad_sensitivity() # v2
                # solve minimization locally
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                # gather solutions from client
                if self.agg == "elastic":
                    # sensitivity = c.grad_sensitivity()
                    # compute delta
                    delta = OrderedDict()
                    for (name, p0), p1 in zip( self.client_model.named_parameters(), c.model.parameters()):
                        delta[name] = p0 - p1
                    csolns.append((soln[0], (delta, sensitivity)))  # (weight, (delta, sensitivity))
                elif self.agg == "prawgs" or self.agg == "prawgcs" or self.agg == "fedsat" or self.agg == "fedsatc":
                    # make sure the self.loss == CACS
                    assert (self.loss == "CACS" or self.loss == "CALC"), f"Client criterion must be CACS or CALC for {self.agg} aggregation."
                    struggling_score = c.criterion.compute_struggler_scores()
                    csolns.append((soln[0], (soln[1], struggling_score)))  # (weight, (soln[1], struggling_score))
                elif self.agg == "drl": 
                    client_solutions_dict[c.id] = soln[1]
                else:
                    csolns.append(soln)
            # update models
            if self.agg is not None and not self.agg=="drl":
                self.latest_model = self.aggregator.aggregate(csolns, self.latest_model)
            if self.agg=="drl":
                self.round = i
                if self.use_prev_global_model:
                    client_solutions_dict[len(self.clients)+1] = deepcopy(self.latest_model)
                self.latest_model = self.drl_aggregate(client_solutions_dict)
                # print(f"Last accuracy: {self.last_acc}")
                # If RL aggregator exposes per-client weights, store them for next round
                if hasattr(self, 'rl_client_weights') and isinstance(self.rl_client_weights, dict):
                    self.last_client_weights = {k: float(v) for k, v in self.rl_client_weights.items()}
            else:   self.latest_model = self.aggregate(csolns)
            self.client_model.load_state_dict(self.latest_model, strict=False)

        self.eval_end()

    def set_client_model_test(self, client: FedAvgClient):
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
