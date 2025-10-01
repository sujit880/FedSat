from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
import torch.nn.functional as F

class FedAvgClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            if self.loss == "PSL":
                K = self.num_classes
                conf = torch.zeros(K, K, dtype=torch.long, device=self.device)            
            if self.loss == "DBCC":
                self.criterion.set_epoch(epoch=epoch)
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                self.optimizer.zero_grad()
                if self.loss == "DBCC":
                    feats = self.model.get_representation_features(inputs)
                    logits = self.model.classifier(feats)
                    if hasattr(self.model, "resnet"):
                        if hasattr(self.model.resnet, "fc"):
                            class_w = self.model.resnet.fc.weight  
                    elif hasattr(self.model, "fc"):
                        class_w = self.model.fc.weight
                    elif hasattr(self.model, "linear"):
                        class_w = self.model.linear.weight                      
                    loss = self.criterion(logits, labels, feats, class_w)
                else:                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                if self.loss == "PSL":
                    with torch.no_grad():
                        self.confusion_bincount(conf, outputs, labels)
                # ====== CAPA: update confusion counters from this batch (no_grad) ======
                if self.loss == "CAPA":
                    K = self.num_classes
                    with torch.no_grad():
                        probs = torch.softmax(outputs, dim=1)      # soft predictions (B,K)
                        # preds = torch.argmax(outputs, dim=1)     # hard predictions (B,K)
                        # probs = F.one_hot(preds, num_classes=K).float()
                        onehot_y = F.one_hot(labels, num_classes=K).float()          # (B,K)
                        # soft confusion: add prob row to the true-class row
                        self.conf_N += onehot_y.T @ probs                            # (K,K)
                        self.pred_q += probs.sum(dim=0)                              # (K,)
                        self.label_y += onehot_y.sum(dim=0)                          # (K,)
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)
            if self.loss == "PSL":
                self.criterion.ema_update_from_confusion(
                        conf, alpha=0.1,      # EMA speed (lower later in training)
                        w_min=1.0, w_max=2.0, # cost range; keep narrow for stability
                        tau=10.0, gamma=1.0   # smoothing & emphasis
                    )
            # (client-only CAPA): refresh W,U every epoch using local counts
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
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

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