from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

class FedMRLClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # RL feedback (filled by server each round)
        self.rl_feedback = {"w": 0.5, "adv": 0.0}
        self.avg_rl_weight = 0.5
        self.adv_gain = 0.5
        self.version = "contrastive"

    # lightweight setter to avoid changing call-sites
    def set_policy_feedback(self, feedback: dict):
        w = float(feedback.get("w", 0.5))
        adv = float(feedback.get("adv", 0.0))
        adv = max(-1.0, min(1.0, adv))
        self.rl_feedback.update({"w": w, "adv": adv})
        avg_w = feedback.get("avg_w", None)
        if avg_w is not None:
            avg_val = float(avg_w)
            # clamp to valid probability range
            self.avg_rl_weight = max(0.0, min(1.0, avg_val))

    def init_client_specific_params(
        self,
        tau: float,
        mu: float,
        **kwargs,
    ) -> None:
        self.tau = tau
        self.mu = mu
        self.adv_gain = kwargs.get("adv_gain", self.adv_gain)
        self.version = kwargs.get("version", self.version)

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
                logits   = self.model.classifier(features)
                ce_loss = self.criterion(logits, labels)

                # teacher/global pass (frozen)
                with torch.no_grad():
                    g_features = global_model.get_representation_features(inputs)

                avg_w_val = float(self.avg_rl_weight)
                adv_val = max(-1.0, min(1.0, float(self.rl_feedback.get("adv", 0.0))))

                # === (2) misclassification alignment weighted by RL feedback ===
                if self.version=="contrastive":
                    curr_predicted = torch.argmax(logits, dim=1)
                    z_curr, z_target, z_pred = [], [], []
                    for tl, pl, f_local, f_global in zip(labels, curr_predicted, features, g_features):
                        if tl != pl:
                            target_proto = self.global_features_prototype.get(tl.item(), f_global.clone().detach())
                            pred_proto = self.global_features_prototype.get(pl.item(), f_local.clone().detach())
                            if isinstance(target_proto, torch.Tensor):
                                target_proto = target_proto.to(self.device)
                            if isinstance(pred_proto, torch.Tensor):
                                pred_proto = pred_proto.to(self.device)
                            z_curr.append(f_local)
                            z_target.append(target_proto)
                            z_pred.append(pred_proto)

                    align_loss = 0.0
                    if z_curr:
                        z_curr = torch.stack(z_curr)
                        z_target = torch.stack(z_target)
                        z_predicted = torch.stack(z_pred)
                        sim_target_curr = F.cosine_similarity(z_curr, z_target, dim=-1)
                        sim_pred_curr = F.cosine_similarity(z_curr, z_predicted, dim=-1)
                        logits_con = torch.cat((sim_target_curr.reshape(-1, 1),
                                                sim_pred_curr.reshape(-1, 1)), dim=1) / max(self.tau, 1e-6)
                        labels_con = torch.zeros(z_curr.size(0), device=self.device).long()
                        align_loss = F.cross_entropy(logits_con, labels_con)
                    # Combine losses with RL-aware alignment weight (no learned gate)
                    align_scale = max(0.0, 1.0 - avg_w_val - self.adv_gain * adv_val)
                    loss = ce_loss
                    if isinstance(align_loss, torch.Tensor) and align_loss.numel() > 0:
                        loss += self.mu * align_scale * align_loss
                elif self.version=="mse":
                    curr_predicted = torch.argmax(logits, dim=1)
                    z_curr, z_target = [], []
                    for tl, pl, f_local, f_global in zip(labels, curr_predicted, features, g_features):
                        if tl != pl:
                            target_proto = self.global_features_prototype.get(tl.item(), f_global.clone().detach())
                            if isinstance(target_proto, torch.Tensor):
                                target_proto = target_proto.to(self.device)
                            z_curr.append(f_local)
                            z_target.append(target_proto)

                    align_loss = 0.0
                    if z_curr:
                        z_curr = torch.stack(z_curr)
                        z_target = torch.stack(z_target)
                        align_loss = F.mse_loss(z_curr, z_target)
                    # Combine losses with RL-aware alignment weight (no learned gate)
                    align_scale = 0.2 * max(0.0, 1.0 - avg_w_val)
                    loss = ce_loss
                    if isinstance(align_loss, torch.Tensor) and align_loss.numel() > 0:
                        loss += self.mu * align_loss
                else:
                    pass
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                train_sample_size += len(labels)

        self.finetune(num_epochs=fine_tune_epochs, batch_size=batch_size) 
        self.model.eval()

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)

        return (bytes_w, comp, bytes_r), (self.num_samples, soln)

    def finetune(self, num_epochs=1, batch_size=10): #v5
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