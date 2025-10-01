from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

# --- add near top of fedmap_client.py ---
import torch.nn as nn
import math

class ResidualAdapter(nn.Module):
    """LoRA/Adapter-style low-rank residual; zero-init 'up' keeps it identity at start."""
    def __init__(self, d_feat: int, r: int = 16):
        super().__init__()
        r = max(1, min(r, d_feat // 2))
        self.down = nn.Linear(d_feat, r, bias=False)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_feat, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # important: start as no-op

    def forward(self, s):
        return self.up(self.act(self.down(s)))


class FedBLOClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # RL feedback (filled by server each round)
        self.rl_feedback = {"w": 0.5, "adv": 0.0}

    # lightweight setter to avoid changing call-sites
    def set_policy_feedback(self, feedback: dict):
        self.rl_feedback.update({"w": float(feedback.get("w", 0.5)),
                                 "adv": float(feedback.get("adv", 0.0))})

    def init_client_specific_params(
        self,
        c_global: OrderedDict[str, torch.Tensor],
        c_local: OrderedDict[str, torch.Tensor],
        tau: float,
        mu: float,
        cost_matrix: torch.Tensor,
        clients_per_round: int,
        total_clients: int,
        prev_model: torch.nn.Module,
        temperature: float = 2.0,
        lambda_rep: float = 0.5,
        lambda_distill: float = 0.5,
        lambda_prox: float = 0.1,
        lambda_contrast: float = 0.5,
        lambda_fair: float = 0.1,
        trust_layers_prefix: tuple = ("layer3", "layer4", "fc", "resnet.fc"),
        **kwargs,
    ) -> None:
        self.lamda = self.lamda
        self.lamdav = self.lamdav
        self.v_rec = self._set_requires_grad_false(deepcopy(c_global))
        self.v_local = self._set_requires_grad_false(deepcopy(c_local))
        self.tau = tau
        self.mu = mu
        self.cost_matrix = cost_matrix
        self.clients_per_round = clients_per_round
        self.total_clients = total_clients
        self.prev_model = deepcopy(prev_model)
        for param in self.prev_model.parameters():
            param.requires_grad = False
        self.prev_model.eval()

        # === new hyper-params for policy-conditioned local subgoal ===
        self.temperature = temperature
        self.lambda_rep = lambda_rep
        self.lambda_distill = lambda_distill
        self.lambda_prox = lambda_prox
        self.lambda_contrast = lambda_contrast
        self.lambda_fair = lambda_fair
        self.trust_layers_prefix = trust_layers_prefix

        # tiny gate (3 -> 3): inputs = [w, feat_drift, logit_kl]; outputs = gates for rep/logit/prox
        self.pcp_gate = torch.nn.Linear(3, 4).to(self.device)
        # after creating self.pcp_gate
        self.optimizer.add_param_group({
            "params": self.pcp_gate.parameters(),
            "lr": self.lr  # or a smaller LR like 0.5*self.lr
        })
        # Infer representation dim once
        # with torch.no_grad():
        #     _dummy = torch.zeros(2, *next(iter(self.trainloader))[0].shape[1:]).to(self.device)
        #     _feat  = self.model.get_representation_features(_dummy)
        #     d_feat = _feat.shape[-1]
        d_feat = self.model.get_feature_dim()

        self.adapter = ResidualAdapter(d_feat, r=min(16, d_feat // 4)).to(self.device)
        self.optimizer.add_param_group({"params": self.adapter.parameters()})
        # weights for new losses (you can pass via kwargs if you prefer)
        self.gamma_kd   = kwargs.get("gamma_kd", 0.5)
        self.delta_orth = kwargs.get("delta_orth", 0.1)
        self.kd_T       = kwargs.get("kd_T", 2.0)

        self.mse_loss = torch.nn.MSELoss()
        self.global_features_prototype = OrderedDict()
        self.average_features_prototype = OrderedDict()

    def _set_requires_grad_false(self, params: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        for val in params.values():
            val.requires_grad = False
        return params

    def solve_inner(self, num_epochs=1, batch_size=10):
        bytes_w = graph_size(self.model)
        train_sample_size = 0

        self.model.train()
        self.adapter.train()
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

    def solve_inner_fedmap(
        self,
        global_parameters: OrderedDict[str, torch.Tensor],
        num_epochs=1,
        batch_size=10
    ):
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        coef = (1 / max(num_epochs, 1)) * self.lr

        global_model = deepcopy(self.model)
        global_model.load_state_dict(global_parameters, strict=False)
        global_model.eval()


        # Build class prototypes once per round
        self.build_class_prototypes(global_model=global_model)
        # rep_epochs = int(num_epochs*0.8) #v5,v6
        rep_epochs = int(num_epochs*0.6) #v53
        fine_tune_epochs = num_epochs - rep_epochs
        # Local training with policy-conditioned subgoal
        self.model.train()
        # for epoch in range(num_epochs): #v4
        for epoch in range(rep_epochs): #v5,v6
            for inputs, labels in self.trainloader:
                if len(labels) <= 1:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # current model pass
                features = self.model.get_representation_features(inputs)
                # personalized residual (kept private)
                p = self.adapter(features)                                     # (B, D)
                z = features + p                                               # (B, D)

                logits   = self.model.classifier(z)                     # used for CE and InfoNCE
                # logits_s = self.model.classifier(features.clone().detach())            # detached to keep KD target stable (v4)
                logits_s = self.model.classifier(features) #(v41)
                # --- KD: make shared s predictive like personalized z --- 
                kdT = self.kd_T
                kd = F.kl_div(
                    F.log_softmax(logits_s / kdT, dim=1),
                    F.softmax(logits.detach() / kdT, dim=1),
                    reduction="batchmean",
                ) * (kdT * kdT)
                # --- Orthogonal residual: encourage p to live off the shared subspace ---
                # s_n = F.normalize(features, dim=1) #not v53 means comment ot
                # p_n = F.normalize(p, dim=1)
                # orth = (s_n * p_n).pow(2).sum(dim=1).mean()
                # # logits = self.model.classifier(features)
                ce_loss = self.criterion(logits, labels)

                # teacher/global pass
                with torch.no_grad():
                    g_features = global_model.get_representation_features(inputs)
                    g_logits = global_model.classifier(g_features)

                # === build gates from RL feedback + drifts ===
                with torch.no_grad():
                    # feature drift (L2) and logit KL (T)
                    feat_drift = (features - g_features).pow(2).sum(dim=1).mean()
                    T = self.temperature
                    p_t = F.log_softmax(logits / T, dim=1)
                    p_s = F.softmax(g_logits / T, dim=1)
                    logit_kl = F.kl_div(p_t, p_s, reduction="batchmean") * (T * T)
                    w = torch.tensor(float(self.rl_feedback.get("w", 0.5)), device=self.device)
                    phi = torch.stack([w, feat_drift.detach(), logit_kl.detach()])
                gates = torch.sigmoid(self.pcp_gate(phi))
                m_rep, m_logit, m_kd, m_orth = gates[0], gates[1], gates[2], gates[3]

                # === (1) logit distillation ===
                distill_loss = logit_kl  # already computed with T^2 scaling

                # === (2) representation alignment to prototypes (with per-class fallback) ===
                target_features = g_features.clone().detach()
                for i, y in enumerate(labels.tolist()):
                    # mix "correct-only" and "all-sample" prototypes via m_rep
                    pc = self.global_features_prototype.get(y, None)
                    pa = self.average_features_prototype.get(y, None)
                    # if pc is not None and pa is not None: #v4,v5,v6
                    #     target_features[i] = (m_rep * pc + (1.0 - m_rep) * pa)
                    if pc is not None and pa is not None: #v52
                        target_features[i] = pc
                    elif pa is not None:
                        target_features[i] = pa
                # rep_loss = self.mse_loss(features, target_features) #v4,v5,v6
                rep_loss = (1-F.cosine_similarity(target_features, features)).mean() #v52

                # === (3) confusion-aware InfoNCE (top-k negatives per sample) === v4,v5,v6
                # positives = class prototype for y; negatives = top-3 alternative class prototypes by cosine sim
                con_loss = 0.0
                with torch.no_grad():
                    # stack available prototypes for quick sim (C_avail, D)
                    proto_items = sorted(self.average_features_prototype.items(), key=lambda kv: kv[0])
                    if len(proto_items) > 0:
                        proto_ids = [c for c, _ in proto_items]
                        proto_mat = torch.stack([p for _, p in proto_items]).to(self.device)  # (C_avail, D)
                        proto_norm = F.normalize(proto_mat, dim=1)
                if len(self.average_features_prototype) > 0:
                    f_norm = F.normalize(features, dim=1)  # (B,D)
                    sims = f_norm @ proto_norm.t()          # (B, C_avail)
                    # for each sample, pick pos = its class proto if present, negatives = top-3 others
                    pos_scores, neg_logits_list, valid_mask = [], [], []
                    for i, y in enumerate(labels.tolist()):
                        if len(proto_items) == 0:
                            continue
                        # indices in proto_ids
                        try:
                            y_idx = proto_ids.index(y)
                        except ValueError:
                            # class proto not available → skip from contrastive (will rely on rep_loss)
                            valid_mask.append(False)
                            continue
                        valid_mask.append(True)
                        s = sims[i]
                        pos_scores.append(s[y_idx:y_idx+1])  # shape (1,)
                        # mask out true class
                        neg_s = s.clone()
                        neg_s[y_idx] = -1e9
                        k = min(3, neg_s.numel() - 1)
                        topk = torch.topk(neg_s, k=k).values  # (k,)
                        neg_logits_list.append(topk)
                    if len(pos_scores) > 0:
                        pos = torch.stack(pos_scores)  # (Bv,1)
                        # pad negatives to same k (in case some rows have <k)
                        maxk = max(x.numel() for x in neg_logits_list)
                        padded = []
                        for x in neg_logits_list:
                            if x.numel() < maxk:
                                pad = torch.full((maxk - x.numel(),), -1e9, device=self.device)
                                x = torch.cat([x, pad], dim=0)
                            padded.append(x)
                        neg = torch.stack(padded)  # (Bv, maxk)
                        logits_con = torch.cat([pos, neg], dim=1) / max(self.tau, 1e-6)
                        labels_con = torch.zeros(logits_con.size(0), dtype=torch.long, device=self.device)
                        con_loss = F.cross_entropy(logits_con, labels_con)
                # scale with original mu as a floor to preserve prior behavior
                con_loss = self.mu * con_loss

                # Combine losses with learned gates
                loss = ce_loss \
                       + self.lambda_rep * m_rep * rep_loss \
                       + self.lambda_distill * m_logit * distill_loss \
                       + self.lambda_contrast * con_loss \
                       + self.gamma_kd * m_kd * kd \
                    #    + self.delta_orth * m_orth * orth

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                # self.finetune_fedblo(num_epochs=1, batch_size=batch_size) #v6
                train_sample_size += len(labels)

        self.finetune_fedblo(num_epochs=fine_tune_epochs, batch_size=batch_size) #v52
        # self.solve_inner(num_epochs=fine_tune_epochs, batch_size=batch_size) #v5
        # Update control variates
        self.model.eval()
        with torch.no_grad():
            y_delta: OrderedDict[str, torch.tensor] = OrderedDict()
            for k, v in self.model.named_parameters():
                y_i = v.data.clone()
                x_i = global_parameters[k].data.clone().detach()
                y_delta[k] = (y_i - x_i).to(y_i.data.dtype)

            for k in self.v_local.keys():
                self.v_local[k] += coef * y_delta[k].clone().detach().to(self.v_local[k].device) - self.v_rec[k].data.clone()

        self.prev_model.load_state_dict(global_model.state_dict())
        self.model.eval()

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)

        return (bytes_w, comp, bytes_r), (self.num_samples, soln)

    def finetune_fedblo(self, num_epochs=1, batch_size=10): #v5
        for (k,v) in self.model.named_parameters():
            if k.startswith('fc.') or k.startswith('resnet.fc.') or k.startswith('linear.') or k.startswith('resnet.linear.'):
                v.requires_grad = True
            else: v.requires_grad = False

        self.solve_inner(num_epochs, batch_size)

        for layer in self.model.parameters():
            layer.requires_grad = True
    
    @torch.no_grad()
    def build_class_prototypes(self, global_model):
        device = self.device
        global_model.eval()

        correct_sum = defaultdict(lambda: None)
        correct_count = defaultdict(int)
        total_sum = defaultdict(lambda: None)
        total_count = defaultdict(int)

        for inputs, labels in self.trainloader:
            if labels.numel() == 0:
                continue
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = global_model.get_representation_features(inputs)
            logits = global_model.classifier(features)
            preds = logits.argmax(dim=1)

            for c in labels.unique():
                c = c.item()
                idx = (labels == c)
                if not idx.any():
                    continue
                feats_c = features[idx]
                batch_sum = feats_c.sum(dim=0)
                if total_sum[c] is None:
                    total_sum[c] = batch_sum.clone()
                else:
                    total_sum[c] += batch_sum
                total_count[c] += idx.sum().item()

            correct_mask = (preds == labels)
            if correct_mask.any():
                labels_corr = labels[correct_mask]
                feats_corr = features[correct_mask]
                for c in labels_corr.unique():
                    c = c.item()
                    idx = (labels_corr == c)
                    feats_c = feats_corr[idx]
                    batch_sum = feats_c.sum(dim=0)
                    if correct_sum[c] is None:
                        correct_sum[c] = batch_sum.clone()
                    else:
                        correct_sum[c] += batch_sum
                    correct_count[c] += idx.sum().item()

        self.global_features_prototype = {}
        for c, cnt in correct_count.items():
            if cnt > 0:
                self.global_features_prototype[c] = (correct_sum[c] / cnt).detach()

        self.average_features_prototype = {}
        for c, cnt in total_count.items():
            if cnt > 0:
                self.average_features_prototype[c] = (total_sum[c] / cnt).detach()