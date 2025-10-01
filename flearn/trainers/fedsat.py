import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from typing import List, Dict
from copy import deepcopy
from collections import OrderedDict
import random

from flearn.clients.fedsat import FedSatLClient
from flearn.config.trainer_params import SCAFFOLD_ARGS
from flearn.utils.constants import CLASSES
from flearn.trainers.server import BaseServer

SIGMA = {"mnist": (0.7, 0.8), "cifar": (0.5, 0.7), "cifar100": (0.5, 0.7), "tinyimagenet": (0.5, 0.7)}
BETA_C = {"mnist": 0.2, "cifar": 0.05, "cifar100": 0.05, "tinyimagenet": 0.05}
G_BETA = {"mnist": (1.1, 1.5), "cifar": (1.1, 1.5), "cifar100": (1.1, 1.5), "tinyimagenet": (1.1, 1.5)}


class FedSatLServer(BaseServer):
    """
    Modification summary:
    - Accepts *client-side* prioritization payloads (E and ACC vectors) from each client.
    - Computes priority class PC_t and per-client aggregation weights ϑ_{t,k} using Eq. (21)-style dependence
      on ACC and E (normalized), then normalizes weights and aggregates deltas.
    - Keeps the rest of the training design unchanged.
    """

    def __init__(self, params):
        print('Using FedSatL to Train')
        params.update(SCAFFOLD_ARGS)
        params["lamdav"] = 1.0
        params["alpha"] = 0.9
        params["beta"] = 0.05
        params["margin"] = 0.3
        # params["scale"] = 1.5

        super().__init__(params)
        self.clients: list[FedSatLClient] = self.clients
        self.global_lr = 1.0
        self.num_classes = CLASSES[self.dataset]
        self.global_params_dict = deepcopy(OrderedDict(self.client_model.named_parameters()))
        for params in self.global_params_dict.values():
            params.requires_grad = False
        self.c_global = OrderedDict((key, torch.zeros_like(value, requires_grad=False, device="cpu")) for (key, value) in self.client_model.named_parameters())

        # Set attributes for all clients
        for client in self.clients:
            client.lamdav = self.lamdav
            client.alpha = self.alpha
            # client.beta = self.beta
            if client.loss == "CSN":
                # client.criterion.scale = self.scale
                client.criterion.beta2 = self.beta
            if client.loss == "CAPA":
                # client.criterion.scale = self.scale
                client.criterion.lam = self.alpha
                client.criterion.mu = self.beta
                client.criterion.margin = self.margin
            SCAFFOLD_ARGS["c_global"] = deepcopy(self.c_global)
            SCAFFOLD_ARGS["c_local"] = deepcopy(self.c_global)
            SCAFFOLD_ARGS["cost_matrix"] = torch.ones(CLASSES[self.dataset], CLASSES[self.dataset], device=self.device)
            SCAFFOLD_ARGS["clients_per_round"] = min(self.clients_per_round, len(self.clients))
            SCAFFOLD_ARGS["total_clients"] = len(self.clients)
            SCAFFOLD_ARGS["prev_model"] = deepcopy(self.client_model)
            SCAFFOLD_ARGS["num_classes"] = self.num_classes
            client.init_client_specific_params(**SCAFFOLD_ARGS)
            client.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client.optimizer, self.num_rounds)

    def train(self):
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            self.eval(i, self.set_client_model_test)
            if self.loss_converged:
                break

            selected_clients: list[FedSatLClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )

            y_deltas = {}
            priority_payloads: Dict[FedSatLClient, Dict[str, torch.Tensor]] = {}
            client_solutions = []

            for c in selected_clients:
                # if self.loss == "CSN":
                #     # Refresh client-side cost stats if used by the loss
                #     c.criterion.cost_reset(cost_matrix=c.test_and_cost_matrix_stats_t())
                c.round = i

                # === Modified return signature: (y_delta, priority_payload, stats_and_soln)
                y_delta, (stats, soln) = c.solve_inner_fedsat_t(
                    self.global_params_dict,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                )
                priority_payload = self.compute_prioritization_payload(client=c, alpha=c.alpha)
                y_deltas[c] = list(y_delta.values())
                priority_payloads[c] = priority_payload  # contains 'E' and 'ACC'                
                client_solutions.append(soln)
            # print("priority_payloads", priority_payloads)

            # Build per-client scalar weights from client-side prioritization payloads
            weights = self.calculate_weights_from_priorities(priority_payloads)
            total_weights = 0.0
            for w,_ in client_solutions: 
                total_weights += w 
            # act_weights = [w+soln[0]/total_weights for w, soln in zip(weights.values(), client_solutions)] #avgw
            act_weights = [soln[0]/total_weights for w, soln in zip(weights.values(), client_solutions)] #fedavgw
            # act_weights = [w+(1/len(client_solutions)) for w, soln in zip(weights.values(), client_solutions)] #aw1
            # print("weights", list(weights.values()))
            # print("fedavg_weights", act_weights)

            # client_soln = [((w,soln[0]), soln[1]) for w, soln in zip(weights.values(), client_solutions)]
            client_soln = [(w, soln[1]) for w, soln in zip(act_weights, client_solutions)]
            # self.global_params_dict = self.aggregate_soln(client_soln)
            # self.global_params_dict = self.aggregate(client_solutions)
            self.global_params_dict = self.aggregate(client_soln)

            # Aggregate deltas using these weights
            # y_delta_cache = [y_deltas[c] for c in selected_clients]
            # statistical_weight_cache = [weights[c] for c in selected_clients]
            # self.global_params_dict = self.aggregate_sens_t(
            #     y_delta_cache=y_delta_cache, statistical_weight_cache=statistical_weight_cache
            # )

        self.eval_end()

    def set_client_model_test(self, client: FedSatLClient):
        client.set_model_params(self.global_params_dict)

    @torch.no_grad()
    def compute_prioritization_payload(self, client: FedSatLClient, alpha: float = 0.3, beta: float = 0.2):
        """
        Robust per-class metrics on the client:
        - Works even when some classes are absent locally (T_i = 0).
        - ACC in [0,1], no >1 values.
        - E = alpha*FNR_norm + beta*FPR_norm, normalized over valid classes only.
        Returns CPU float32 tensors: {"E": [m], "ACC": [m]}
        """
        device = client.device
        client.model.eval()

        # Infer number of classes
        m = CLASSES[self.dataset]

        # Accumulators
        total = torch.zeros((), dtype=torch.float64, device=device)
        T  = torch.zeros(m, dtype=torch.float64, device=device)   # true count per class
        TP = torch.zeros(m, dtype=torch.float64, device=device)   # true positives
        Pred = torch.zeros(m, dtype=torch.float64, device=device) # predicted-as count (Zhat)

        # loader = getattr(self, "testloader", None) or getattr(self, "valloader", None) or self.trainloader
        if self.test_subset is None:
            self.test_subset, _ = self.get_representative_subset(test_loader=self.test_loader, num_samples=1024)
        for xb, yb in self.test_subset:
            xb, yb = xb.to(device), yb.to(device)
            logits = client.model(xb)
            preds = torch.argmax(logits, dim=1)
            total += yb.numel()
            for i in range(m):
                T[i]   += torch.sum(yb == i)
                Pred[i] += torch.sum(preds == i)
                TP[i]  += torch.sum((yb == i) & (preds == i))

        FN = T - TP
        FP = Pred - TP
        TN = total - (TP + FP + FN)

        # Valid classes are those with at least one true sample
        valid_T = T > 0

        # ACC_i = TP/(TP+FN) = TP/T, for valid classes; else 0
        ACC = torch.zeros(m, dtype=torch.float64, device=device)
        ACC[valid_T] = TP[valid_T] / T[valid_T]

        # FNR_i = FN/(TP+FN) for valid classes; else NaN -> set to 0 later
        FNR = torch.zeros(m, dtype=torch.float64, device=device)
        FNR[valid_T] = FN[valid_T] / (TP[valid_T] + FN[valid_T])

        # FPR_i = FP/(FP+TN) for classes with negatives (total - T_i > 0)
        has_neg = (total - T) > 0
        FPR = torch.zeros(m, dtype=torch.float64, device=device)
        # FPR[has_neg] = FP[has_neg] / torch.clamp((FP[has_neg] + TN[has_neg]), min=eps)
        # FPR[valid_T] = (Pred[valid_T] - TP[valid_T]) / (T.sum() + T[valid_T])
        # FPR[has_neg] = (Pred[has_neg] - TP[has_neg]) / (T.sum() - T[has_neg])
        FPR[has_neg] = FP[has_neg] / (FP[has_neg] + TN[has_neg])
        E = alpha * FNR + (1-alpha) * FPR

        # eps = random.uniform(1e-3, 1e-4)
        # E += eps
        # ACC += eps

        # print(f"[{self.id}]-TP:{TP}, \n[{self.id}]-T:{T}, \n{self.id}]-ACC:{ACC}, \n{self.id}]-E:{E}")

        return {
            "E":   E.detach().to("cpu", dtype=torch.float32),
            "ACC": ACC.detach().to("cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def calculate_weights_from_priorities(self, payloads: Dict[FedSatLClient, Dict[str, torch.Tensor]]) -> Dict[FedSatLClient, float]:
        """
        Given payloads per client with tensors 'E' (prioritization score vector) and 'ACC' (class-wise accuracy),
        compute:
          1) Priority class PC_t = argmax_i sum_k E_{i,t,k}
          2) Per-client weight ϑ_{t,k} ∝ [sum_i ACC_{i,t,k}] * max(1, ACC_{PC_t,t,k} * ν) / sum_i [ E_{i,t,k} * (ACC_{i,t,k} * ν) ]
             (then L1-normalize weights across selected clients)
        """
        # Stack client vectors (align shapes)
        clients = list(payloads.keys())
        if not clients:
            return {}
        E_list = [payloads[c]["E"].to(torch.float64) for c in clients]
        ACC_list = [payloads[c]["ACC"].to(torch.float64) for c in clients]
        m = E_list[0].numel()

        # 1) Priority class
        sum_E = torch.stack(E_list, dim=0).sum(dim=0)  # [m]
        PC_t = int(torch.argmax(sum_E).item())

        # 2) Per-client ϑ computation (Eq. 21-style)
        nu = 100.0
        eps = 1e-9
        raw_weights = []
        for c, E_vec, ACC_vec in zip(clients, E_list, ACC_list):
            acc_sum = ACC_vec.sum()
            acc_pc = ACC_vec[PC_t]
            num = acc_sum * torch.maximum(torch.tensor(1.0, dtype=torch.float64), acc_pc * nu)
            denom = (E_vec * (ACC_vec * nu)).sum()
            denom = torch.clamp(denom, min=eps)
            w = (num / denom).item()
            raw_weights.append(w)
        weights_np = np.array(raw_weights, dtype=np.float64)
        weights_np = np.maximum(weights_np, 0)
        if weights_np.sum() == 0:
            weights_np += 1.0
        weights_np = weights_np / weights_np.sum()

        return {c: float(w) for c, w in zip(clients, weights_np.tolist())}

    @torch.no_grad()
    def aggregate_sens_t(
        self,
        y_delta_cache: List[List[torch.Tensor]],  # per-client list of parameter deltas
        statistical_weight_cache: List[float],    # per-client scalar weights
    ):
        counts = torch.tensor(statistical_weight_cache, device=self.device, dtype=torch.float32)
        if counts.numel() == 0 or counts.sum() <= 0:
            raise ValueError("statistical_weight_cache is empty or sums to zero.")
        weights = counts / counts.sum()

        client_soln = [(w, deltas) for w, deltas in zip(weights, y_delta_cache)]
        aggregated_delta = self.aggregate(client_soln)

        name_to_param = dict(self.client_model.named_parameters())
        for name, delta in aggregated_delta.items():
            if name not in name_to_param:
                continue
            p = name_to_param[name]
            p.add_(self.global_lr * delta.to(device=p.device, dtype=p.dtype))

        self.global_params_dict = deepcopy(OrderedDict(self.client_model.named_parameters()))
        return self.global_params_dict

    def aggregate(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors
        # Initialize base with zeros tensors with the same size as the first solution's parameters'
        model_state_dict: OrderedDict = wsolns[0][1]
        base = [torch.zeros_like(soln) for soln in model_state_dict.values()]

        for w, client_state_dict in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += (w * v).to(base[i].dtype)

        # Divide each aggregated tensor by the total weight to compute the average
        averaged_soln = [v / total_weight for v in base]
        averaged_state_dict = OrderedDict(zip(model_state_dict.keys(), averaged_soln))

        return averaged_state_dict
    
    def aggregate_soln(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        total_weight_cs = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors
        # Initialize base with zeros tensors with the same size as the first solution's parameters'
        model_state_dict: OrderedDict = wsolns[0][1]
        base = OrderedDict((k,torch.zeros_like(v)) for k, v in model_state_dict.items())
        for k in base.keys():
            for w, _ in wsolns:  # w is the number of local samples
                # print("w", w)
                total_weight += w[1]
                total_weight_cs += w[0]
            for w, client_state_dict in wsolns:
                if k.startswith('fc.') or k.startswith('resnet.fc.')  or k.startswith('linear'):
                    # base[k] += (w[0] * client_state_dict[k]/total_weight_cs).to(base[k].dtype)
                    base[k] += (w[0] * client_state_dict[k]).to(base[k].dtype)
                else:
                    # base[k] += (w[0] * client_state_dict[k]).to(base[k].dtype)
                    base[k] += (w[0] * client_state_dict[k].data).to(base[k].dtype)
        # Divide each aggregated tensor by the total weight to compute the average
        for k in base.keys():
            if k.startswith('fc.') or k.startswith('resnet.fc.')  or k.startswith('linear'):
                base[k] = base[k]/total_weight_cs
            else:
                base[k] = base[k]/total_weight_cs
        return base
