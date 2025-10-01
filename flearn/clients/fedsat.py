from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple
import random

class FedSatLClient(BaseClient):
    """
    Modification summary:
    - Keeps original training design.
    - After local training, compute client-side prioritization score Et,k along with
      FNR, FPR, ACC (class-wise) using the client's *test* (or val) loader.
    - Returns (y_delta, priority_payload, stats_and_soln) to the server, where
      priority_payload = {"E": tensor[m], "ACC": tensor[m]}.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = 0.3
        self.beta = 0.2
        self.lamdav = 1.0

    def init_client_specific_params(
        self,
        c_global: OrderedDict[str, torch.Tensor],
        c_local: OrderedDict[str, torch.Tensor],
        cost_matrix: torch.Tensor,
        clients_per_round: int,
        total_clients: int,
        prev_model: torch.nn.Module,
        num_classes: int,
        **kwargs,
    ) -> None:     
        self.num_classes = num_classes
        self.c_global: OrderedDict[str, torch.Tensor] = c_global
        for val in self.c_global.values():
            val.requires_grad = False
        self.c_local: OrderedDict[str, torch.Tensor] = c_local
        for val in self.c_local.values():
            val.requires_grad = False
        self.cost_matrix = cost_matrix
        self.clients_per_round = clients_per_round
        self.total_clients = total_clients
        self.prev_params_dict: OrderedDict[str, torch.Tensor] = None
        self.mse_loss = torch.nn.MSELoss()
        self.prev_model = prev_model
        for param in self.prev_model.parameters():
            param.requires_grad = False
        self.prev_model.eval()

    def solve_inner(self, num_epochs=1, batch_size=10):
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    @torch.no_grad()
    def _infer_num_classes(self) -> int:
        # Prefer cost_matrix shape when available (always set by server).
        if hasattr(self, "criterion") and hasattr(self.criterion, "cost_matrix"):
            cm = getattr(self.criterion, "cost_matrix", None)
            if isinstance(cm, torch.Tensor) and cm.dim() == 2 and cm.shape[0] == cm.shape[1]:
                return int(cm.shape[0])
        # Fallback: probe a small batch
        loader = getattr(self, "testloader", None) or getattr(self, "valloader", None) or self.trainloader
        for xb, _ in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            return int(logits.shape[-1])
        return 0

    @torch.no_grad()
    def compute_prioritization_payload_(self, alpha: float = 0.3, beta: float = 0.2) -> Dict[str, torch.Tensor]:
        """Compute class-wise FNR, FPR, ACC and prioritization score E (client-side).
        Returns a dict with tensors on CPU: {"E": [m], "ACC": [m]}.
        Safe against zero-division via eps.
        """
        m = self._infer_num_classes()
        # eps = 1e-8
        eps = random.uniform(1e-2, 1e-3)
        device = self.device

        T = torch.zeros(m, dtype=torch.float64, device=device)      # per-class true count |T_i|
        TP = torch.zeros(m, dtype=torch.float64, device=device)     # per-class true positive |TP_i|
        Zhat = torch.zeros(m, dtype=torch.float64, device=device)   # per-class predicted as i |Z^_i|

        # loader = getattr(self, "testloader", None) or getattr(self, "valloader", None) or self.trainloader
        self.model.eval()
        for xb, yb in self.valloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = self.model(xb)
            preds = torch.argmax(logits, dim=1)
            # accumulate
            for i in range(m):
                T[i] += torch.sum(yb == i)
                Zhat[i] += torch.sum(preds == i)
                TP[i] += torch.sum((yb == i) & (preds == i))

        # Metrics (vectorized)
        # ACC = TP / (T + eps)
        ACC = torch.zeros(m, dtype=torch.float64, device=device)
        for i in range(len(ACC)):
            ACC[i] = eps if T[i] == 0 else eps + (TP[i] / T[i])
        # FNR = (T - TP) / (TP + eps)  # per manuscript Eq. (12) style (safe) 
        FNR = torch.zeros(m, dtype=torch.float64, device=device)
        for i in range(len(FNR)):
            FNR[i] = 0 if TP[i] == 0 else (T[i] -TP[i]) / TP[i]
        denom_fp = torch.clamp(torch.sum(T) - T, min=eps)
        FPR = torch.clamp(Zhat - TP, min=0.0) / denom_fp

        # Normalize by per-vector maxima
        FNRn = FNR / torch.clamp(FNR.max(), min=eps)
        FPRn = FPR / torch.clamp(FPR.max(), min=eps)
        E = alpha * FNRn + beta * FPRn

        # Move to CPU for light payload
        return {"E": E.detach().to("cpu", dtype=torch.float32),
                "ACC": ACC.detach().to("cpu", dtype=torch.float32)}

    @torch.no_grad()
    def compute_prioritization_payload(self, alpha: float = 0.3, beta: float = 0.2):
        """
        Robust per-class metrics on the client:
        - Works even when some classes are absent locally (T_i = 0).
        - ACC in [0,1], no >1 values.
        - E = alpha*FNR_norm + beta*FPR_norm, normalized over valid classes only.
        Returns CPU float32 tensors: {"E": [m], "ACC": [m]}
        """
        device = self.device
        self.model.eval()

        # Infer number of classes
        m = None
        if hasattr(self, "criterion") and hasattr(self.criterion, "cost_matrix"):
            cm = getattr(self.criterion, "cost_matrix", None)
            if isinstance(cm, torch.Tensor) and cm.dim() == 2 and cm.shape[0] == cm.shape[1]:
                m = int(cm.shape[0])
        if m is None:
            # Fallback: probe a batch
            loader_probe = getattr(self, "testloader", None) or getattr(self, "valloader", None) or self.trainloader
            for xb, _ in loader_probe:
                m = int(self.model(xb.to(device)).shape[-1])
                break
        if m is None:
            # No data at all on this client
            return {"E": torch.zeros(0), "ACC": torch.zeros(0)}

        # Accumulators
        total = torch.zeros((), dtype=torch.float64, device=device)
        T  = torch.zeros(m, dtype=torch.float64, device=device)   # true count per class
        TP = torch.zeros(m, dtype=torch.float64, device=device)   # true positives
        Pred = torch.zeros(m, dtype=torch.float64, device=device) # predicted-as count (Zhat)

        # loader = getattr(self, "testloader", None) or getattr(self, "valloader", None) or self.trainloader
        for xb, yb in self.valloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = self.model(xb)
            preds = torch.argmax(logits, dim=1)
            total += yb.numel()
            for i in range(m):
                T[i]   += torch.sum(yb == i)
                Pred[i] += torch.sum(preds == i)
                TP[i]  += torch.sum((yb == i) & (preds == i))

        # Compute FP, FN, TN
        # T +=1
        # Pred += 1
        # TP += 1
        FN = T - TP
        FP = Pred - TP
        # FP += 1
        TN = total - (TP + FP + FN)
        # TN += 1

        # Valid classes are those with at least one true sample
        valid_T = T > 0
        eps = 1e-6

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
        FPR[has_neg] = (Pred[has_neg] - TP[has_neg]) / (T.sum() - T[has_neg])

        # Normalize FNR/FPR over classes that are actually defined
        # def safe_norm(vec, mask):
        #     if mask.any():
        #         vmax = torch.max(vec[mask])
        #         return vec / torch.clamp(vmax, min=eps)
        #     else:
        #         return torch.zeros_like(vec)

        # FNRn = safe_norm(FNR, valid_T)          # only where T_i > 0
        # FPRn = safe_norm(FPR, has_neg)          # only where negatives exist

        # E = alpha * FNRn + (1-alpha) * FPRn

        E = alpha * FNR + (1-alpha) * FPR

        eps = random.uniform(1e-3, 1e-4)
        E += eps
        ACC += eps

        # print(f"[{self.id}]-TP:{TP}, \n[{self.id}]-T:{T}, \n{self.id}]-ACC:{ACC}, \n{self.id}]-E:{E}")

        return {
            "E":   E.detach().to("cpu", dtype=torch.float32),
            "ACC": ACC.detach().to("cpu", dtype=torch.float32),
        }

    def solve_inner_fedsat_t(
        self,
        global_parameters: OrderedDict[str, torch.Tensor],
        num_epochs=1,
        batch_size=10,
    ) -> Tuple[OrderedDict, Dict[str, torch.Tensor], Tuple[Tuple[int, int, int], Tuple[int, OrderedDict]]]:
        """Local update with control variates + send prioritization payload.
        Returns: (y_delta, priority_payload, ((bytes_w, comp, bytes_r), (num_samples, soln)))
        """
        bytes_w = graph_size(self.model)
        train_sample_size = 0

        global_model = deepcopy(self.prev_model)
        global_model.load_state_dict(global_parameters, strict=False)
        global_model.eval()
        coef = (1 / num_epochs) * self.lr

        prev_params_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
            (key, value.clone().detach()) for (key, value) in self.prev_model.named_parameters()
        )
        for val in prev_params_dict.values():
            val.requires_grad = False

        with torch.no_grad():
            for k in self.c_global.keys():
                self.c_global[k] += 0.5 * coef * (
                    global_parameters[k].data.clone() - prev_params_dict[k].data.clone()
                ).to(self.c_global[k].device) + 0.5 * self.c_local[k]

        self.model.train()
        for epoch in range(num_epochs):
            if self.loss == "PSL":
                K = self.num_classes
                conf = torch.zeros(K, K, dtype=torch.long, device=self.device)
            for inputs, labels in self.trainloader:
                if len(labels) <= 1:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = self.model.get_representation_features(inputs)
                logits = self.model.classifier(features)
                # ====== CAPA: update confusion counters from this batch (no_grad) ======
                if self.loss == "CAPA":
                    K = self.num_classes
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=1)                         # (B,K)
                        onehot_y = F.one_hot(labels, num_classes=K).float()          # (B,K)
                        # soft confusion: add prob row to the true-class row
                        self.conf_N += onehot_y.T @ probs                            # (K,K)
                        self.pred_q += probs.sum(dim=0)                              # (K,)
                        self.label_y += onehot_y.sum(dim=0)                          # (K,)

                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                for (k, param) in self.model.named_parameters():
                    if k.startswith("fc.") or k.startswith("resnet.fc."):
                        if param.grad is not None:
                            param.grad.data += self.lamdav * (self.c_global[k].data - self.c_local[k].data).to(self.device)
                self.optimizer.step()
                train_sample_size += len(labels)
                if self.loss == "PSL":
                    with torch.no_grad():
                        self.confusion_bincount(conf, logits, labels)
            if self.loss == "PSL":
                self.criterion.ema_update_from_confusion(
                        conf, alpha=self.alpha,      # EMA speed (lower later in training)
                        w_min=1.0, w_max=2.0, # cost range; keep narrow for stability
                        tau=10.0, gamma=1.0   # smoothing & emphasis
                    )
            # Option A (client-only CAPA): refresh W,U every epoch using local counts
            if self.loss == "CAPA":
                with torch.no_grad():
                    W, U = self.build_W_U(self.conf_N, self.pred_q, self.label_y,
                                    beta=1.0, gamma=2.0, eps=1e-6)
                    self.criterion.update_weights(W, U)
                # (optional) decay/EMA the counters instead of hard reset:
                decay = 0.9
                self.conf_N *= decay
                self.pred_q *= decay
                self.label_y *= decay

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)

        # update local control variate & compute y_delta
        self.model.eval()
        with torch.no_grad():
            y_delta: OrderedDict[str, torch.tensor] = OrderedDict()
            for k, v in self.model.named_parameters():
                y_i = v.data.clone()
                x_i = global_parameters[k].data.clone().detach()
                y_delta[k] = (y_i - x_i).to(y_i.data.dtype)

            for k, y_del in self.c_local.items():
                self.c_local[k] += coef * y_del.clone().detach().to(self.c_local[k].device) - self.c_global[k].data.clone()

        self.prev_model.load_state_dict(global_model.state_dict())
        self.model.train()

        # === NEW: compute client-side prioritization payload ===
        # priority_payload = self.compute_prioritization_payload(alpha=self.alpha, beta=self.beta)

        # return y_delta, priority_payload, ((bytes_w, comp, bytes_r), (self.num_samples, soln))
        return y_delta, ((bytes_w, comp, bytes_r), (self.num_samples, soln))
    
    @torch.no_grad()
    def confusion_bincount(self, conf: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
        """
        Vectorized in-place accumulation of confusion counts.
        conf: [K,K] long
        logits: [B,K]
        targets: [B]
        """
        K = logits.size(1)
        preds = logits.argmax(dim=1)
        idx = targets * K + preds
        counts = torch.bincount(idx, minlength=K*K).reshape(K, K)
        conf += counts

    @torch.no_grad()
    def build_W_U(self, conf_counts, pred_counts, label_counts, beta=1.0, gamma=2.0, eps=1e-6):
        """
        conf_counts N[a,b] is your (soft or hard) confusion matrix as counts.
        """
        K = conf_counts.size(0)
        C = conf_counts.clone().float()
        C.fill_diagonal_(0.0)
        C = C / (C.sum(dim=1, keepdim=True) + eps)                      # row-normalized rates

        W = (C + eps).pow(beta); W.fill_diagonal_(0.0)
        W = W / (W.sum(dim=1, keepdim=True) + eps)

        pred = pred_counts.float(); pred = pred / (pred.sum() + eps)
        lab  = label_counts.float(); lab  = lab  / (label_counts.sum() + eps)
        over = torch.clamp(pred - lab, min=0.0).pow(gamma)
        U = over / (over.sum() + eps)                                   # possibly all zeros early
        return W, U