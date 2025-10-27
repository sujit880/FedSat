import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple

from flearn.trainers.server import BaseServer
from flearn.clients.ccvr import CCVRClient
from flearn.utils.constants import CLASSES
from flearn.utils.aggregator import Aggregator


def gaussian_virtual_features(mean: torch.Tensor, var: torch.Tensor, n: int) -> torch.Tensor:
    """
    Sample n virtual features from diagonal Gaussian N(mean, var). var is per-dim variance.
    mean/var: [D]
    returns [n, D]
    """
    D = mean.shape[0]
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    eps = torch.randn(n, D, device=mean.device, dtype=mean.dtype)
    return mean.unsqueeze(0) + eps * std.unsqueeze(0)


class CCVRServer(BaseServer):
    """
    CCVR (Classifier Calibration with Virtual Representations) implementation:
    - Normal FedAvg training
    - After aggregation each round, ask clients for per-class feature stats
    - Aggregate means/variances, synthesize virtual features per class
    - Calibrate classifier head using synthetic features (one or few epochs)
    """

    def __init__(self, params):
        print("Using CCVR to Train")
        # defaults
        params.setdefault("ccvr_samples_per_class", 128)
        params.setdefault("ccvr_epochs", 1)
        params.setdefault("ccvr_lr", 0.01)

        # Provide sensible server-side aggregation defaults like FedAvgServer
        def _get_lr_for_dataset(agg_method, dataset):
            dataset_lr = {
                "cifar10": {"fedadam": 1e-3, "fedavgm": 0.5},
                "cifar": {"fedadam": 1e-3, "fedavgm": 0.5},
                "cifar100": {"fedadam": 5e-4, "fedavgm": 0.5},
                "mnist": {"fedadam": 1e-2, "fedavgm": 1.0},
                "fmnist": {"fedadam": 1e-2, "fedavgm": 0.5},
                "emnist": {"fedadam": 1e-2, "fedavgm": 1.0},
                "femnist": {"fedadam": 1e-2, "fedavgm": 1.0},
            }
            return dataset_lr.get(dataset, {}).get(agg_method, 0.01)

        agg = params.get("agg", None)
        if agg in {"fedadam", "fedavgm", "fedyogi"}:
            params.setdefault("server_learning_rate", _get_lr_for_dataset(agg, params.get("dataset")))
        else:
            params.setdefault("server_learning_rate", 1.0)
        params.setdefault("server_momentum", 0.99)

        super().__init__(params)

        self.num_classes = CLASSES[self.dataset]
        # swap client class to CCVRClient (already created via TRAINERS map)
        self.clients: List[CCVRClient] = [c for c in self.clients]  # type: ignore

        # choose aggregation method if provided
        if getattr(self, "agg", None) is not None:
            self.aggregator = Aggregator(method=self.agg, model=self.client_model, lr=self.server_learning_rate, mu=self.server_momentum, top_p=getattr(self, "top_p", None))

    def set_client_model_test(self, client: CCVRClient):
        client.set_model_params(self.latest_model)

    def train(self):
        print("Training with {} workers ---".format(self.clients_per_round))

        for rnd in trange(self.num_rounds, desc=self.desc):
            self.eval(rnd, self.set_client_model_test)
            if self.loss_converged:
                break

            selected_clients: List[CCVRClient] = self.select_clients(
                rnd, num_clients=min(self.clients_per_round, len(self.clients))
            )

            csolns = []
            stats_payloads = []
            for c in selected_clients:
                c.set_model_params(self.latest_model)
                soln, stats, ccvr_stats = c.solve_inner_ccvr(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
                csolns.append(soln)
                stats_payloads.append(ccvr_stats)

            # aggregate model as FedAvg (or server agg if set)
            if hasattr(self, "aggregator") and self.aggregator is not None:
                self.latest_model = self.aggregator.aggregate(csolns, self.latest_model)
            else:
                self.latest_model = self.aggregate(csolns)

            # CCVR calibration: compute global per-class mean/var and retrain classifier
            self.client_model.load_state_dict(self.latest_model, strict=False)
            self.calibrate_classifier(stats_payloads)
            self.latest_model = self.client_model.state_dict()

        self.eval_end()

    def aggregate(self, wsolns):  # Weighted average like FedAvg
        total_weight = 0.0
        model_state_dict: OrderedDict = wsolns[0][1]
        base = [torch.zeros_like(soln) for soln in model_state_dict.values()]

        for w, client_state_dict in wsolns:
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v

        averaged_soln = [v / total_weight for v in base]
        averaged_state_dict = OrderedDict(zip(model_state_dict.keys(), averaged_soln))
        return averaged_state_dict

    @torch.no_grad()
    def aggregate_feature_stats(self, payloads: List[Dict[str, torch.Tensor]]):
        K = self.num_classes
        # feature dim might vary; assume consistent
        feat_dim = int(payloads[0]["feature_dim"]) if payloads else 1
        total_counts = torch.zeros(K, dtype=torch.long)
        total_sum = torch.zeros(K, feat_dim, dtype=torch.float32)
        total_sumsq = torch.zeros(K, feat_dim, dtype=torch.float32)
        for p in payloads:
            counts = p["counts"].to(total_counts.dtype)
            total_counts += counts
            # guard shape for clients without data
            fdim = int(p["feature_dim"]) if "feature_dim" in p else feat_dim
            if fdim != feat_dim:
                continue
            total_sum += p["sum"].to(total_sum.dtype)
            total_sumsq += p["sumsq"].to(total_sumsq.dtype)

        mean = torch.zeros(K, feat_dim, dtype=torch.float32)
        var = torch.ones(K, feat_dim, dtype=torch.float32)
        for c in range(K):
            n = int(total_counts[c].item())
            if n > 0:
                m = total_sum[c] / n
                v = torch.clamp(total_sumsq[c] / n - m * m, min=1e-6)
                mean[c] = m
                var[c] = v
        return total_counts, mean, var

    def calibrate_classifier(self, payloads: List[Dict[str, torch.Tensor]]):
        # compute global stats
        counts, mean, var = self.aggregate_feature_stats(payloads)
        K, D = mean.shape

        # build virtual dataset
        n_per = self.ccvr_samples_per_class
        X_list = []
        y_list = []
        device = self.device
        for c in range(K):
            if counts[c].item() == 0:
                continue
            feats = gaussian_virtual_features(mean[c].to(device), var[c].to(device), n_per)
            X_list.append(feats)
            y_list.append(torch.full((feats.size(0),), c, dtype=torch.long, device=device))
        if not X_list:
            return
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        # Freeze feature extractor, train classifier only
        self.client_model.train()
        if hasattr(self.client_model, 'get_representation_params_t'):
            for p in self.client_model.get_representation_params_t():
                p.requires_grad = False
        # find classifier parameters
        cls_params = []
        if hasattr(self.client_model, 'get_classifier_params_t'):
            cls_params = list(self.client_model.get_classifier_params_t())
        else:
            # fallback: parameters named 'fc.' or 'resnet.fc.'
            cls_params = [v for k, v in self.client_model.named_parameters() if k.startswith('fc.') or k.startswith('resnet.fc.')]
        if not cls_params:
            return

        optimizer = torch.optim.SGD(cls_params, lr=self.ccvr_lr, momentum=0.9)

        # simple epochs over virtual data in mini-batches
        bs = min(256, X.size(0))
        for _ in range(self.ccvr_epochs):
            perm = torch.randperm(X.size(0), device=device)
            Xp = X[perm]
            yp = y[perm]
            for i in range(0, Xp.size(0), bs):
                xb = Xp[i:i+bs]
                yb = yp[i:i+bs]
                logits = self.client_model.classifier(xb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
