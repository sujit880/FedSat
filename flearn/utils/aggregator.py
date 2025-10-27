# flearn/aggregation.py
import torch
from collections import OrderedDict

class Aggregator:
    def __init__(self, method="fedavg", model=None, lr=0.1, beta1=0.9, beta2=0.99, eps=1e-3, mu=0.9, tau=0.5, top_p=3):
        self.method = method.lower()
        self.m = None
        self.v = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.lr = lr
        self.control_variate = None
        self.model: torch.nn.Module = model
        self.mu = mu
        self.tau = tau
        self.top_p = top_p  # Number of top struggling classes to prioritize

        # Set optimizer if model is provided and method is fedavgm or fedadam
        self.optimizer = None
        if self.model is not None:
            if self.method == "fedavgm":
                # self.lr = 1.0  # lr is handled by optimizer
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mu)
            elif self.method == "fedadam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps)

    @torch.no_grad()
    def aggregate(self, wsolns, global_state):
        """
        wsolns: list of tuples (weight, state_dict) from clients
        global_state: OrderedDict (previous global model state_dict)
        """
        total_weight = sum([w for w, _ in wsolns])
        base = [torch.zeros_like(v) for v in global_state.values()]

        if self.method == "mean":   # simple mean (ignores weights)
            for _, state in wsolns:
                for i, v in enumerate(state.values()):
                    base[i] += v
            averaged = [v / len(wsolns) for v in base]

        elif self.method == "fedavg":
            for w, state in wsolns:
                for i, v in enumerate(state.values()):
                    base[i] += w * v
            averaged = [v / total_weight for v in base]

        elif self.method == "fedadam":  # FedAdam (adaptive)
            self.model.load_state_dict(global_state, strict=False)
            # Only aggregate floating-point parameters
            param_names = [name for name, param in self.model.named_parameters()]
            grads = [torch.zeros_like(global_state[name]) for name in param_names]
            for w, state in wsolns:
                for i, name in enumerate(param_names):
                    grads[i] += w * (state[name] - global_state[name]) / total_weight
            # Assign gradients and step optimizer
            for param, g in zip(self.model.parameters(), grads):
                param.grad = -g  # negative to move towards the average
            self.optimizer.step()
            # Build new global state: update parameters, keep buffers unchanged
            new_state = OrderedDict()
            for k in global_state.keys():
                if k in param_names:
                    # Use updated parameter
                    new_state[k] = self.model.state_dict()[k].clone()
                else:
                    # Keep buffer as is
                    new_state[k] = global_state[k].clone()
            averaged = list(new_state.values())

        elif self.method == "fedyogi":  # FedYogi (like Adam but uses Yogi update)
            self.t += 1
            # Only compute gradients for floating-point parameters
            param_names = [name for name, param in self.model.named_parameters() if param.dtype.is_floating_point]
            grads = [torch.zeros_like(global_state[name]) for name in param_names]
            for w, state in wsolns:
                for i, name in enumerate(param_names):
                    grads[i] += w * (state[name] - global_state[name]) / total_weight

            if self.m is None:
                self.m = [torch.zeros_like(g) for g in grads]
                self.v = [torch.zeros_like(g) for g in grads]

            # Build new state with updated parameters and unchanged buffers
            new_state = OrderedDict()
            param_idx = 0
            for k, v in global_state.items():
                if k in param_names:
                    g = grads[param_idx]
                    self.m[param_idx] = self.beta1 * self.m[param_idx] + (1 - self.beta1) * g
                    self.v[param_idx] = self.v[param_idx] - (1 - self.beta2) * (g * g) * torch.sign(self.v[param_idx] - g * g)
                    m_hat = self.m[param_idx] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[param_idx] / (1 - self.beta2 ** self.t)
                    new_state[k] = v - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                    param_idx += 1
                else:
                    # Keep buffer unchanged
                    new_state[k] = v.clone()
            averaged = list(new_state.values())

        elif self.method == "fedavgm":  # FedAvg with server momentum using SGD optimizer
            self.model.load_state_dict(global_state, strict=False)
            # Only compute gradients for floating-point parameters
            param_names = [name for name, param in self.model.named_parameters()]
            grads = [torch.zeros_like(global_state[name]) for name in param_names]
            for w, state in wsolns:
                for i, name in enumerate(param_names):
                    grads[i] += w * (state[name] - global_state[name]) / total_weight
            # Assign gradients and step optimizer
            for param, g in zip(self.model.parameters(), grads):
                param.grad = -g  # negative to move towards the average
            self.optimizer.step()
            # Build new global state: update parameters, keep buffers unchanged
            new_state = OrderedDict()
            for k in global_state.keys():
                if k in param_names:
                    # Use updated parameter
                    new_state[k] = self.model.state_dict()[k].clone()
                else:
                    # Keep buffer as is
                    new_state[k] = global_state[k].clone()
            averaged = list(new_state.values())

        elif self.method == "elastic":
            self.model.load_state_dict(global_state, strict=False)
            # wsolns: list of (weight, state, sensitivity)
            weights = torch.tensor([w for w, _ in wsolns], device=next(self.model.parameters()).device)
            weights = weights / weights.sum()
            sensitivity_list = [s for _, (_, s) in wsolns]
            stacked_sensitivity = torch.stack(sensitivity_list, dim=-1)
            aggregated_sensitivity = torch.sum(stacked_sensitivity * weights, dim=-1)

            max_sensitivity = stacked_sensitivity.max(dim=-1)[0]
            max_sensitivity = torch.where(max_sensitivity == 0, torch.tensor(1e-7, device=max_sensitivity.device), max_sensitivity)
            zeta = 1 + self.tau - aggregated_sensitivity / max_sensitivity  
            state_list = [list(state.values()) for _, (state, _) in wsolns]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*state_list)
            ]

            for param, coef, diff in zip(self.model.parameters(), zeta, aggregated_delta):
                param.data -= coef * diff

            averaged = [p.data.clone() for p in self.model.parameters()]

        elif self.method == "prawgs":  # Personalized Re-weighted Aggregation with Struggling Classes
            self.model.load_state_dict(global_state, strict=False)
            # wsolns: list of (weight, state, struggler_score)
            weights = torch.tensor([w for w, _ in wsolns], device=next(self.model.parameters()).device)
            weights = weights / weights.sum()
            struggler_list = [s for _, (_, s) in wsolns]  # each s is a tensor of shape (num_classes,)
            # Compute global struggler score (mean across clients)
            global_struggler = torch.stack(struggler_list, dim=0).mean(dim=0)
            # Pick top-p struggling classes
            p = min(self.top_p, global_struggler.numel())  # or set p as you wish
            top_p_classes = torch.topk(global_struggler, p).indices.tolist()

            class_models = []
            for cls in top_p_classes:
                state_list = [list(state.values()) for _, (state, _) in wsolns]
                # For class-specific params (e.g., final layer), aggregate with struggler weighting
                class_state = OrderedDict((k, v.clone()) for k, v in global_state.items())
                for i, (name, param) in enumerate(class_state.items()):
                    # If param is class-specific (e.g., [num_classes, ...])
                    if param.ndim > 1 and param.shape[0] == global_struggler.shape[0]:
                        agg_row = torch.zeros_like(param[cls])
                        total_weight = 0.0
                        for w, (state, s) in wsolns:
                            score = s[cls]
                            agg_row += w * score * state[name][cls]
                            total_weight += w * score
                        if total_weight > 0:
                            param[cls] = agg_row / total_weight
                    elif param.ndim == 1 and param.shape[0] == global_struggler.shape[0]:
                        agg_val = 0.0
                        total_weight = 0.0
                        for w, (state, s) in wsolns:
                            score = s[cls]
                            agg_val += w * score * state[name][cls]
                            total_weight += w * score
                        if total_weight > 0:
                            param[cls] = agg_val / total_weight
                    else:
                        # For other params, do normal weighted average
                        param.data.zero_()
                        total_weight = 0.0
                        for w, (state, _) in wsolns:
                            param.data += w * state[name]
                            total_weight += w
                        if total_weight > 0:
                            param.data = (param.data / total_weight).to(param.data.dtype)
                class_models.append(class_state)
            # Average the p class-specialized models to obtain the new global model
            averaged = []
            for i, k in enumerate(global_state.keys()):
                stacked = torch.stack([cm[k] for cm in class_models], dim=0)
                if not torch.is_floating_point(stacked):
                    stacked = stacked.float()
                mean_param = stacked.mean(dim=0)
                # Optionally cast back to original dtype:
                orig_dtype = global_state[k].dtype
                if mean_param.dtype != orig_dtype:
                    mean_param = mean_param.to(orig_dtype)
                averaged.append(mean_param)

        elif self.method == "prawgcs":  # Personalized Re-weighted Aggregation with Class Competence
            self.model.load_state_dict(global_state, strict=False)
            # wsolns: list of (weight, (state, struggler_score))
            weights = torch.tensor([w for w, _ in wsolns], device=next(self.model.parameters()).device)
            weights = weights / weights.sum()
            struggler_list = [s for _, (_, s) in wsolns]  # each s is a tensor of shape (num_classes,)
            global_struggler = torch.stack(struggler_list, dim=0).mean(dim=0)
            p = min(self.top_p, global_struggler.numel())
            top_p_classes = torch.topk(global_struggler, p).indices.tolist()

            class_models = []
            for cls in top_p_classes:
                state_list = [list(state.values()) for _, (state, _) in wsolns]
                class_state = OrderedDict((k, v.clone()) for k, v in global_state.items())
                for i, (name, param) in enumerate(class_state.items()):
                    if param.ndim > 1 and param.shape[0] == global_struggler.shape[0]:
                        agg_row = torch.zeros_like(param[cls])
                        total_weight = 0.0
                        for w, (state, s) in wsolns:
                            competence = 1.0 - s[cls]
                            agg_row += w * competence * state[name][cls]
                            total_weight += w * competence
                        if total_weight > 0:
                            param[cls] = agg_row / total_weight
                    elif param.ndim == 1 and param.shape[0] == global_struggler.shape[0]:
                        agg_val = 0.0
                        total_weight = 0.0
                        for w, (state, s) in wsolns:
                            competence = 1.0 - s[cls]
                            agg_val += w * competence * state[name][cls]
                            total_weight += w * competence
                        if total_weight > 0:
                            param[cls] = agg_val / total_weight
                    else:
                        param.data.zero_()
                        total_weight = 0.0
                        for w, (state, _) in wsolns:
                            param.data += w * state[name]
                            total_weight += w
                        if total_weight > 0:
                            param.data = (param.data / total_weight).to(param.data.dtype)
                class_models.append(class_state)
            averaged = []
            for i, k in enumerate(global_state.keys()):
                stacked = torch.stack([cm[k] for cm in class_models], dim=0)
                if not torch.is_floating_point(stacked):
                    stacked = stacked.float()
                mean_param = stacked.mean(dim=0)
                # Optionally cast back to original dtype:
                orig_dtype = global_state[k].dtype
                if mean_param.dtype != orig_dtype:
                    mean_param = mean_param.to(orig_dtype)
                averaged.append(mean_param)

        elif self.method == "fedsat":  # Federated SAT (focus on struggling classes)
            self.model.load_state_dict(global_state, strict=False)
            # wsolns: list of (weight, (state, struggler_score))
            struggler_list = [s for _, (_, s) in wsolns]  # each s is a tensor of shape (num_classes,)
            global_struggler = torch.stack(struggler_list, dim=0).mean(dim=0)
            p = min(self.top_p, global_struggler.numel())
            top_p_classes = torch.topk(global_struggler, p).indices.tolist()

            class_models = []
            for cls in top_p_classes:
                class_state = OrderedDict((k, v.clone()) for k, v in global_state.items())
                # Compute class-specific client weights (normalize to sum to 1)
                client_weights = torch.tensor([(1.0-s[cls]) for _, (_, s) in wsolns], device=next(self.model.parameters()).device)
                if client_weights.sum() > 0:
                    client_weights = client_weights / client_weights.sum()
                else:
                    client_weights = torch.ones_like(client_weights) / len(client_weights)
                # Aggregate all parameters using these weights
                for i, (name, param) in enumerate(class_state.items()):
                    param.data.zero_()
                    for cw, (_, (state, _)) in zip(client_weights, wsolns):
                        param.data += (cw * state[name]).to(param.data.dtype)
                class_models.append(class_state)
            # Average the p class-specialized models to obtain the new global model
            averaged = []
            for i, k in enumerate(global_state.keys()):
                stacked = torch.stack([cm[k] for cm in class_models], dim=0)
                if not torch.is_floating_point(stacked):
                    stacked = stacked.float()
                mean_param = stacked.mean(dim=0)
                # Optionally cast back to original dtype:
                orig_dtype = global_state[k].dtype
                if mean_param.dtype != orig_dtype:
                    mean_param = mean_param.to(orig_dtype)
                averaged.append(mean_param)

        elif self.method == "fedsatc":  # Federated SAT with Class Competence
            self.model.load_state_dict(global_state, strict=False)
            # wsolns: list of (weight, (state, struggler_score))
            struggler_list = [s for _, (_, s) in wsolns]  # each s is a tensor of shape (num_classes,)
            global_struggler = torch.stack(struggler_list, dim=0).mean(dim=0)
            p = min(self.top_p, global_struggler.numel())
            top_p_classes = torch.topk(global_struggler, p).indices.tolist()

            class_models = []
            for cls in top_p_classes:
                class_state = OrderedDict((k, v.clone()) for k, v in global_state.items())
                # Multiply client struggler score with original client weight for stability
                client_weights = torch.tensor(
                    [w * s[cls] for w, (_, s) in wsolns],
                    device=next(self.model.parameters()).device
                )
                if client_weights.sum() > 0:
                    client_weights = client_weights / client_weights.sum()
                else:
                    client_weights = torch.ones_like(client_weights) / len(client_weights)
                # Aggregate all parameters using these weights
                for i, (name, param) in enumerate(class_state.items()):
                    param.data.zero_()
                    for cw, (_, (state, _)) in zip(client_weights, wsolns):
                        param.data += (cw * state[name]).to(param.data.dtype)
                class_models.append(class_state)
            # Average the p class-specialized models to obtain the new global model
            averaged = []
            for i, k in enumerate(global_state.keys()):
                stacked = torch.stack([cm[k] for cm in class_models], dim=0)
                if not torch.is_floating_point(stacked):
                    stacked = stacked.float()
                mean_param = stacked.mean(dim=0)
                # Optionally cast back to original dtype:
                orig_dtype = global_state[k].dtype
                if mean_param.dtype != orig_dtype:
                    mean_param = mean_param.to(orig_dtype)
                averaged.append(mean_param)

        else:
            raise NotImplementedError(f"Aggregation method {self.method} not supported")

        return OrderedDict(zip(global_state.keys(), averaged))
