from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
from collections import OrderedDict
from copy import deepcopy

class SCAFFOLDClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_client_specific_params(
        self,
        c_local: OrderedDict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        # Ensure c_local is copied to the client's device and ordered
        # to match model.named_parameters(). This avoids device / ordering
        # mismatches when applying control-variate corrections.
        self.c_local: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in c_local.items():
            # clone to avoid accidental shared storage and move to client device
            self.c_local[k] = v.clone().to(self.device)
            self.c_local[k].requires_grad = False


    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """

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
    
    def solve_inner_scaffold(self,c_global: OrderedDict[str, torch.Tensor], 
                            global_parameters: OrderedDict[str, torch.Tensor],
                            num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        self.model.load_state_dict(global_parameters, strict=False)
        # Precompute coef = 1/(K * eta) where K is total local steps (num_epochs * batches_per_epoch)
        # We compute it here once (static for this local solve) to avoid recomputing inside loops.
        try:
            batches_per_epoch = len(self.trainloader)
            if batches_per_epoch <= 0:
                batches_per_epoch = 1
        except Exception:
            batches_per_epoch = 1
        try:
            lr_val = float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            lr_val = float(self.lr) if hasattr(self, "lr") else float(getattr(self, "learning_rate", 1e-3))
        tau = max(1, int(num_epochs) * int(batches_per_epoch))
        coef = 1.0 / (tau * lr_val)

        self.model.train()
        for epoch in range(num_epochs): 
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                # Apply control-variate correction to all trainable parameters.
                # Ensure G and c_i are aligned to the parameter device/dtype.
                for (name, param) in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    # get server and client controls by name to avoid ordering issues
                    G = c_global[name]
                    c_i = self.c_local[name]
                    # move controls to the parameter grad device if needed
                    if G.device != param.grad.device:
                        G = G.to(param.grad.device)
                    if c_i.device != param.grad.device:
                        c_i = c_i.to(param.grad.device)
                    # Paper: add (c_i - c) to the local gradient (client minus server).
                    # Previously this used (G - c_i) which is server - client and can push
                    # updates in the opposite direction causing divergence / NaNs.-> it also doing same as above
                    param.grad.data.add_(G - c_i)
                self.optimizer.step()
                train_sample_size += len(labels)

        self.model.eval()               
        with torch.no_grad():
            y_delta = OrderedDict()
            c_delta = OrderedDict()   
            c_plus = OrderedDict()
            # compute y_delta (difference of model before and after training)
            for k, v in self.model.named_parameters():
                y_i = v.data.clone()
                x_i = global_parameters[k].data.clone()
                y_delta[k] = y_i - x_i

            # coef was precomputed before local training loop

            # compute c_plus using explicit key-based lookup to preserve ordering
            for k in self.c_local.keys():
                G = c_global[k]
                c_l = self.c_local[k]
                y_del = y_delta[k]
                # move tensors to a common device (use G device as target)
                if y_del.device != G.device:
                    y_del = y_del.to(G.device)
                # SCAFFOLD paper: c_plus = c_l - G + (1/(tau*eta)) * y_delta
                c_plus[k] = c_l - G + coef * (-y_del) # y_del = y_i - x; -y_del = x - y_i to match paper formula

            # compute c_delta
            for k, c_p, c_l in zip(c_plus.keys(), c_plus.values(), self.c_local.values()):
                c_delta[k] = c_p.data - c_l.data
            for k in self.c_local.keys():
                self.c_local[k] = c_plus[k]
        self.model.train()
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), (deepcopy(y_delta), deepcopy(c_delta))
    