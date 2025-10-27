import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class LabelCalibratedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        label_distrib=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        tau=1.0,
    ):
        super(LabelCalibratedCrossEntropyLoss, self).__init__()
        self.label_distrib = label_distrib
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.tau = tau

    def forward(self, logit, y_logit):
        cal_logit = torch.exp(
            logit
            - (
                self.tau
                * torch.pow(self.label_distrib, -1 / 4).expand((logit.shape[0], -1))
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=y_logit.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]
    
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        reduction="mean",
        ignore_index=-100
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(true_dist * log_probs, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class ClassBalancedCELoss(nn.Module):
    def __init__(self, beta=0.9999, reduction="mean", ignore_index=-100):
        super(ClassBalancedCELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        labels_one_hot = F.one_hot(targets, num_classes).float()

        effective_num = 1.0 - torch.pow(self.beta, labels_one_hot.sum(0))
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * num_classes

        weights = weights.to(logits.device)
        ce_loss = F.cross_entropy(
            logits, targets, weight=weights, reduction=self.reduction,
            ignore_index=self.ignore_index
        )
        return ce_loss

class DBLoss(nn.Module):
    def __init__(self, K, alpha=1.0, ema_m=0.99, prior_ema=True, device=None):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.ema_m = ema_m
        self.prior_ema = prior_ema
        self.device = device

        # EMA priors (labels) and confusion (rows: true, cols: predicted)
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))  # small symmetric prior
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))

    @torch.no_grad()
    def update_emas(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # update label prior
        batch_counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        if self.prior_ema:
            self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (batch_counts + 1e-2)) #v1
        else:
            self.ema_label_counts.add_(batch_counts + 1e-2)

        # update confusion with hard preds, but smoothed
        onehot_y = F.one_hot(targets, num_classes=self.K).float()
        onehot_p = F.one_hot(preds,   num_classes=self.K).float()
        # soften the update a bit by mixing hard & soft
        conf_update = 0.7 * (onehot_y.T @ onehot_p) + 0.3 * (onehot_y.T @ probs)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + 1e-2))

    def forward(self, logits, targets):
        """
        logits: (B, K) linear classifier outputs (before softmax)
        feats:  (B, D) penultimate features
        class_weights: (K, D) classifier weight vectors
        """
        # 1) Update EMA stats (no grad)
        with torch.no_grad():
            self.update_emas(logits.detach(), targets.detach())
            # priors = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8) #v1
            with torch.no_grad(): #v2
                C = self.ema_conf.clone()
                # normalize each row to get confusion probabilities
                priors = C / (C.sum(dim=1, keepdim=True) + self.tiny)
            log_prior = priors.log()  # (K,)

        # 2) Logit-adjusted CE
        # la_logits = logits + self.alpha * log_prior.unsqueeze(0)  # stable, diagonal adjustment v1
        la_logits = logits + self.alpha * log_prior[targets]  # stable, diagonal adjustment v2
        ce = F.cross_entropy(la_logits, targets, reduction='mean')
        return ce
 
@torch.no_grad()
def confusion_bincount(conf: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
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
def confusion_to_cost(conf: torch.Tensor,
                      w_min: float = 1.0, w_max: float = 2.0,
                      tau: float = 10.0, gamma: float = 1.0) -> torch.Tensor:
    """
    Convert a confusion matrix to a bounded, row-normalized cost matrix.
    - Zero diagonal (no penalty for correct predictions).
    - Row-normalize with smoothing (tau).
    - Optional exponent gamma to emphasize frequent confusions.
    - Rescale off-diagonals to [w_min, w_max].
    """
    K = conf.size(0)
    row_sums = conf.sum(dim=1, keepdim=True) + tau
    R = conf.float() / row_sums
    R.fill_diagonal_(0.0)

    if gamma != 1.0:
        R = R.clamp_min(1e-12).pow(gamma)

    maxv = R.max().clamp_min(1e-12)
    C_new = w_min + (w_max - w_min) * (R / maxv)
    C_new.fill_diagonal_(0.0)
    return C_new

class CAPALoss(torch.nn.Module): #v1
    def __init__(self, W, U, lam=0.5, mu=0.1, tau=2.0, margin=0.5, kappa=6.0, use_gate=True):
        """
        W: (K,K) tensor with rows summing to 1, diag=0
        U: (K,)   tensor summing to 1 (can be all zeros initially)
        """
        super().__init__()
        self.register_buffer("W", W.float())
        self.register_buffer("U", U.float())
        self.lam = lam
        self.mu = mu
        self.tau = tau
        self.margin = margin
        self.kappa = kappa
        self.use_gate = use_gate

    def forward(self, logits, targets):
        # print(f"lam: {self.lam}, mu: {self.mu}, margin: {self.margin}")
        # logits: (B,K), targets: (B,)
        B, K = logits.size()
        probs = F.softmax(logits, dim=1)

        # ----- CE -----
        ce = F.cross_entropy(logits, targets, reduction='none')

        # ----- Pairwise confusion term -----
        z_a = logits.gather(1, targets.view(-1,1))                     # (B,1)
        # (B,K) margins: z_b - z_a + m
        margins = logits - z_a + self.margin
        pair_mask = torch.ones_like(margins, dtype=torch.bool)
        pair_mask[torch.arange(B), targets] = False                    # exclude b=a

        # select W rows per target
        W_rows = self.W[targets]                                       # (B,K)
        # softplus(τ * margin)
        splus = F.softplus(self.tau * margins)
        pair_term = (W_rows * splus).masked_select(pair_mask).view(B, K-1).sum(dim=1)
        # print(f"pair_term1: {pair_term}")
        if self.use_gate:
            # α = sigmoid(kappa * (max_{b≠a} z_b - z_a))
            z_wo_a = logits.clone()
            z_wo_a[torch.arange(B), targets] = -1e9
            z_max_other = z_wo_a.max(dim=1, keepdim=True).values
            gate = torch.sigmoid(self.kappa * (z_max_other - z_a)).squeeze(1) # defalt Kappa(k) = 6.0
        else:
            gate = torch.ones(B, device=logits.device)

        pair_term = gate * pair_term
        # print(f"pair_term2: {pair_term}")

        # ----- Anti-bias term -----
        U = self.U.view(1, -1).expand(B, K)                            # (B,K)
        anti_mask = pair_mask                                          # exclude target class
        anti_bias = (U * probs).masked_select(anti_mask).view(B, K-1).sum(dim=1)#'''

        # loss = (1-self.lam)*ce + self.lam * pair_term + self.mu * anti_bias
        loss = ce + self.lam * pair_term + self.mu * anti_bias
        # loss = pair_term + self.lam * ce + self.mu * anti_bias
        return loss.mean()
    
    @torch.no_grad()
    def update_weights(self, W, U):
        self.W.copy_(W.to(self.W.device, dtype=torch.float32))
        self.U.copy_(U.to(self.U.device, dtype=torch.float32))

class CACSLoss_rough(nn.Module):
    """
    Confusion-Aware Cost-Sensitive Multiclass Logistic
    L(z,y) = -z_y + log sum_j exp(z_j + Delta[y,j])
    where Delta[y,j] >= 0 encodes pairwise cost from EMA confusion.
    Optional: add Bayes prior bias b_j = beta * log pi_j to logits.
    """
    def __init__(self, K, device=None, m0=0.1, alpha=0.5, ema_m=0.995, warmup_steps=0,
                 prior_beta=0.0, conf_beta=0.5, min_floor=1e-3, lmu=0.9, cmu=0.01):
        super().__init__()
        self.K = K
        self.m0 = m0
        self.alpha = alpha
        self.ema_m = ema_m
        self.warmup_steps = warmup_steps
        self.prior_beta = prior_beta
        self.conf_beta = conf_beta 
        self.min_floor = min_floor
        self.lmu = lmu 
        self.cmu = cmu 
        self.device = device
        self.symmetrize = True 
        self.row_norm = "softmax" # "l1" or "softmax"

        # EMA stats
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))       # soft confusion
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))  # for priors
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def _update_emas_4(self, logits, targets):
        # self.ema_m = 0.95 # comment out for v7
        P = F.softmax(logits, dim=1)                                # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()          # (B,K)
        conf_update = Y.T @ P                                       # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.min_floor))

        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (counts + self.min_floor))
        
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)

    @torch.no_grad()
    def _build_Delta(self):
        # symmetric, row-normalized confusion graph in [0,1], zero diag
        C = self.ema_conf.clone()
        A = 0.5 * (C + C.T)
        A = A / (A.sum(dim=1, keepdim=True) + self.tiny)
        A = A - torch.diag(torch.diag(A))
        if A.max() > 0:
            A = A / (A.max() + self.tiny)
        # Delta = self.m0 + self.alpha * A #v1,2
        Delta = self.m0 + A #v1,2
        # ensure zero diagonal
        Delta.fill_diagonal_(0.0) #v 1,2
        return Delta  # (K,K) # v2
    @torch.no_grad()
    def _build_Delta_1(self):
        # symmetric, row-normalized confusion graph in [0,1], zero diag
        C = self.ema_conf.clone()
        A = 0.5 * (C + C.T)
        A = A / (A.sum(dim=1, keepdim=True) + self.tiny)
        A = A - torch.diag(torch.diag(A))
        if A.max() > 0:
            A = A / (A.max() + self.tiny)
        return A  # (K,K) # V1

    
    @torch.no_grad() # v6_c_5_p_5
    def _build_margin_matrix(self):
        """
        From EMA confusion -> normalized pairwise margin matrix M in [0,1], zero diagonal.
        """
        C = self.ema_conf.clone()

        # Symmetrize if desired so margins reflect mutual confusion
        if self.symmetrize:
            C = 0.5 * (C + C.T)

        # Row-normalize to probabilities
        if self.row_norm == "softmax":
            C = torch.softmax(C, dim=1)
        else:  # "l1" (default)
            C = C / (C.sum(dim=1, keepdim=True) + self.tiny)

        # Zero diagonal (we never subtract margin from the target logit)
        M = C - torch.diag(torch.diag(C))
        # Normalize M to [0,1] scale (robust)
        M = M / (M.max() + self.tiny)

        return M
    
    @torch.no_grad() # v52=lpi.log * pi.exp       
    def _log_prior_5(self):
        lpi = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)        
        C = self.ema_conf.clone()
        C = C + 1e-3
        pi = (C / (C.sum())).clamp_min(self.tiny)
        return lpi.log() * pi.exp()  # (K,) # v5=pi.log,v51=pi.log * pi.exp

    def forward(self, logits, targets):
        with torch.no_grad():
            self._update_emas_4(logits.clone().detach(), targets.clone().detach()) # v5, v52_p_5

        z = logits
        # Optional Bayes logit adjustment
        if self.prior_beta > 0.0:
            # z = z + self.prior_beta * self._log_prior_5()[targets] # v4, v52_p_5 #comment out for v9 that remove delta addition
            z_plus = z + self.prior_beta * self._log_prior_5()[targets] # v9

        # Build Delta after warmup ##comment out for v9 that remove delta addition
        # if self.steps.item() >= self.warmup_steps and (self.m0 > 0.0 or self.alpha > 0.0):
        #     with torch.no_grad():
        #         Delta = self._build_Delta_1()                   # (K,K) # v1=self._build_Delta_1() # v2=self._build_Delta()
        #     D_y = Delta[targets]                              # (B,K)
        # else:
        #     # no cost yet
        #     D_y = torch.zeros_like(z)

        # Cost-augmented softmax log-loss:
        # L = -z_y + logsumexp(z_j + Delta[y,j])
        # z_plus = z + D_y #comment out for v9 that remove delta addition
        lse = torch.logsumexp(z_plus, dim=1)                  # (B,)
        zy = z.gather(1, targets.view(-1,1)).squeeze(1)       # (B,)
        log_loss = (lse - zy).mean() 
        # Confusion-aware matrix shift after warmup  # v6_c_5_p_5
        conf_loss = 0.0
        if self.steps.item() >= self.warmup_steps and self.conf_beta > 0.0:
            with torch.no_grad():
                M = self._build_margin_matrix()          # (K, K)

            margin_rows = M[targets]                     # (B, K)
            # Ensure zero on the target class index
            margin_rows[torch.arange(targets.size(0), device=targets.device), targets] = 0.0

            # Apply calibrated subtraction to logits
            callogits = logits - self.conf_beta * margin_rows

            # Standard CE on calibrated logits
            conf_loss = F.cross_entropy(callogits, targets, reduction='mean')

        return self.lmu * log_loss + self.cmu * conf_loss
    
    @torch.no_grad()
    def compute_struggler_scores(self):
        """
        Compute class-wise struggler scores using EMA confusion and label counts.
        Returns a tensor of size (K,) with values in [0,1].
        High score = class is struggling.
        """
        C = self.ema_conf.clone()
        counts = self.ema_label_counts.clone().clamp_min(self.tiny)

        # Misclassifications per class (row sum excluding diagonal)
        errors = C.sum(dim=1) - torch.diag(C)

        # Struggler score = error rate for each class
        struggler = (errors / counts).clamp(0.0, 1.0)

        return struggler  # (K,)

class CACSLoss(nn.Module):
    """
    Confusion-Aware Cost-Sensitive Multiclass Logistic
    L(z,y) = -z_y + log sum_j exp(z_j + Delta[y,j])
    where Delta[y,j] >= 0 encodes pairwise cost from EMA confusion.
    Optional: add Bayes prior bias b_j = beta * log pi_j to logits.
    """
    def __init__(self, K, device=None, ema_m=0.995, warmup_steps=0,
                 prior_beta=0.0, conf_beta=0.5, min_floor=1e-3, lmu=0.9, cmu=0.01, **kwargs):
        super().__init__()
        self.K = K
        self.ema_m = ema_m
        self.warmup_steps = warmup_steps
        self.prior_beta = prior_beta
        self.conf_beta = conf_beta 
        self.min_floor = min_floor
        self.lmu = lmu 
        self.cmu = cmu 
        self.device = device
        self.symmetrize = True 
        self.row_norm = "softmax" # "l1" or "softmax"

        # EMA stats
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))       # soft confusion
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))  # for priors
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def _update_emas(self, logits, targets):
        # self.ema_m = 0.95 # comment out for v7
        P = F.softmax(logits, dim=1)                                # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()          # (B,K)
        conf_update = Y.T @ P                                       # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.min_floor))

        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (counts + self.min_floor))
        
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)

    @torch.no_grad()
    def _build_Delta(self):
        # symmetric, row-normalized confusion graph in [0,1], zero diag
        C = self.ema_conf.clone()
        A = 0.5 * (C + C.T)
        A = A / (A.sum(dim=1, keepdim=True) + self.tiny)
        A = A - torch.diag(torch.diag(A))
        if A.max() > 0:
            A = A / (A.max() + self.tiny)
        return A  # (K,K)

    
    @torch.no_grad()
    def _build_margin_matrix(self):
        """
        From EMA confusion -> normalized pairwise margin matrix M in [0,1], zero diagonal.
        """
        C = self.ema_conf.clone()

        # Symmetrize if desired so margins reflect mutual confusion
        if self.symmetrize:
            C = 0.5 * (C + C.T)

        # Row-normalize to probabilities
        if self.row_norm == "softmax":
            C = torch.softmax(C, dim=1)
        else:  # "l1" (default)
            C = C / (C.sum(dim=1, keepdim=True) + self.tiny)

        # Zero diagonal (we never subtract margin from the target logit)
        M = C - torch.diag(torch.diag(C))
        # Normalize M to [0,1] scale (robust)
        M = M / (M.max() + self.tiny)

        return M
    
    @torch.no_grad()    
    def _log_prior(self):
        lpi = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)        
        C = self.ema_conf.clone()
        C = C + 1e-3
        pi = (C / (C.sum())).clamp_min(self.tiny)
        return lpi.log() * pi.exp()  # (K,) 

    def forward(self, logits, targets):
        with torch.no_grad():
            self._update_emas(logits.clone().detach(), targets.clone().detach()) # v5, v52_p_5

        z = logits
        # Optional Bayes logit adjustment
        if self.prior_beta > 0.0:
            z = z + self.prior_beta * self._log_prior()[targets] 

        # Build Delta after warmup 
        if self.steps.item() >= self.warmup_steps:
            with torch.no_grad():
                Delta = self._build_Delta()                   # (K,K) 
            D_y = Delta[targets]                              # (B,K)
        else:
            # no cost yet
            D_y = torch.zeros_like(z)

        # Cost-augmented softmax log-loss:
        # L = -z_y + logsumexp(z_j + Delta[y,j])
        z_plus = z + D_y
        lse = torch.logsumexp(z_plus, dim=1)                  # (B,)
        zy = z.gather(1, targets.view(-1,1)).squeeze(1)       # (B,)
        log_loss = (lse - zy).mean() 
        # Confusion-aware matrix shift after warmup
        conf_loss = 0.0
        if self.steps.item() >= self.warmup_steps and self.conf_beta > 0.0:
            with torch.no_grad():
                M = self._build_margin_matrix()          # (K, K)

            margin_rows = M[targets]                     # (B, K)
            # Ensure zero on the target class index
            margin_rows[torch.arange(targets.size(0), device=targets.device), targets] = 0.0

            # Apply calibrated subtraction to logits
            callogits = logits - self.conf_beta * margin_rows

            # Standard CE on calibrated logits
            conf_loss = F.cross_entropy(callogits, targets, reduction='mean')

        return self.lmu * log_loss + self.cmu * conf_loss
    
    @torch.no_grad()
    def compute_struggler_scores(self):
        """
        Compute class-wise struggler scores using EMA confusion and label counts.
        Returns a tensor of size (K,) with values in [0,1].
        High score = class is struggling.
        """
        C = self.ema_conf.clone()
        counts = self.ema_label_counts.clone().clamp_min(self.tiny)

        # Misclassifications per class (row sum excluding diagonal)
        errors = C.sum(dim=1) - torch.diag(C)

        # Struggler score = error rate for each class
        struggler = (errors / counts).clamp(0.0, 1.0)

        return struggler  # (K,)


class CACSLoss_LC(CACSLoss):
    """
    CACS with Label Calibration (CACS-LC)
    Combines dynamic confusion-aware Δ[y,j] with global label calibration scaling.
    """
    def __init__(self, K, label_distrib=None, tau=1.0, **kwargs):
        super().__init__(K, **kwargs)
        self.tau = tau
        # ensure label_distrib is a tensor on the right device
        if label_distrib is not None:
            label_distrib = torch.tensor(label_distrib, dtype=torch.float32, device=self.device)
        else:
            label_distrib = torch.ones(K, device=self.device)
        self.register_buffer("label_distrib", label_distrib)

    def forward(self, logits, targets):
        with torch.no_grad():
            # update EMAs inherited from CACSLoss
            self._update_emas(logits.clone().detach(), targets.clone().detach())

        # ----- Label Calibration (LCCE) -----
        calib = self.tau * torch.pow(self.label_distrib, -0.25)
        z = logits - calib.unsqueeze(0)  # (B, K)

        # ----- Apply Bayesian bias correction if any -----
        if self.prior_beta > 0.0:
            z = z + self.prior_beta * self._log_prior()[targets]

        # ----- Build Delta from EMA Confusion -----
        if self.steps.item() >= self.warmup_steps:
            with torch.no_grad():
                Delta = self._build_Delta()
            D_y = Delta[targets]
        else:
            D_y = torch.zeros_like(z)

        # ----- Confusion-aware cost-sensitive log loss -----
        z_plus = z + D_y
        lse = torch.logsumexp(z_plus, dim=1)
        zy = z.gather(1, targets.view(-1, 1)).squeeze(1)
        log_loss = (lse - zy).mean()

        # ----- Optional confusion-regularized CE -----
        conf_loss = 0.0
        if self.steps.item() >= self.warmup_steps and self.conf_beta > 0.0:
            with torch.no_grad():
                M = self._build_margin_matrix()
            margin_rows = M[targets]
            margin_rows[torch.arange(targets.size(0), device=targets.device), targets] = 0.0
            callogits = z - self.conf_beta * margin_rows
            conf_loss = F.cross_entropy(callogits, targets, reduction='mean')

        return self.lmu * log_loss + self.cmu * conf_loss

class LabelCalibratedCE_CACS(nn.Module):
    """
    LCCE enhanced with CACS-style confusion adaptivity (LC-CACS)
    """
    def __init__(self, K, label_distrib=None, tau=1.0, lambda_conf=0.5, ema_m=0.995, device=None):
        super().__init__()
        self.K = K
        self.tau = tau
        self.lambda_conf = lambda_conf
        self.device = device
        # EMA confusion
        self.register_buffer("ema_conf", torch.ones(K, K, device=device))
        self.register_buffer("tiny", torch.tensor(1e-8, device=device))
        if label_distrib is not None:
            label_distrib = torch.tensor(label_distrib, dtype=torch.float32, device=device)
        else:
            label_distrib = torch.ones(K, device=device)
        self.register_buffer("label_distrib", label_distrib)
        self.ema_m = ema_m

    @torch.no_grad()
    def _update_conf(self, logits, targets):
        P = F.softmax(logits, dim=1)
        Y = F.one_hot(targets, num_classes=self.K).float()
        conf_update = Y.T @ P
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.tiny))

    @torch.no_grad()
    def _margin_matrix(self):
        C = 0.5 * (self.ema_conf + self.ema_conf.T)
        C = C / (C.sum(dim=1, keepdim=True) + self.tiny)
        M = C - torch.diag(torch.diag(C))
        M = M / (M.max() + self.tiny)
        return M

    def forward(self, logits, targets):
        with torch.no_grad():
            self._update_conf(logits.clone().detach(), targets.clone().detach())
            M = self._margin_matrix()

        # ----- Base LCCE correction -----
        calib = self.tau * torch.pow(self.label_distrib, -0.25)
        z = logits - calib.unsqueeze(0)

        # ----- Confusion adaptive correction -----
        conf_corr = self.lambda_conf * M[targets]
        z = z - conf_corr

        # ----- Calibrated CE -----
        exp_z = torch.exp(z)
        y_z = torch.gather(exp_z, dim=1, index=targets.unsqueeze(1))
        loss = -torch.log(y_z / exp_z.sum(dim=1, keepdim=True))
        return loss.mean()

class CADBLoss(nn.Module):
    def __init__(self, K, alpha=0.5, ema_m=0.995, warmup_steps=0,
                 prior_beta=0.0, conf_beta=0.5, min_floor=1e-3, lmu=0.9, cmu=0.01, device=None):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.ema_m = ema_m
        self.warmup_steps = warmup_steps
        self.prior_beta = prior_beta
        self.conf_beta = conf_beta 
        self.min_floor = min_floor
        self.lmu = lmu 
        self.cmu = cmu 
        self.device = device
        self.symmetrize = True 
        self.row_norm = "softmax" # "l1" or "softmax"

        # EMA stats
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))       # soft confusion
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))  # for priors
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def _update_emas_4(self, logits, targets):
        # self.ema_m = 0.95 # comment out for v7
        P = F.softmax(logits, dim=1)                                # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()          # (B,K)
        conf_update = Y.T @ P                                       # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.min_floor))

        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (counts + self.min_floor))
        
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)

    @torch.no_grad()
    def _build_Delta(self):
        # symmetric, row-normalized confusion graph in [0,1], zero diag
        C = self.ema_conf.clone()
        A = 0.5 * (C + C.T)
        A = A / (A.sum(dim=1, keepdim=True) + self.tiny)
        A = A - torch.diag(torch.diag(A))
        if A.max() > 0:
            A = A / (A.max() + self.tiny)
        Delta = self.m0 + self.alpha * A #v1,2
        # ensure zero diagonal
        Delta.fill_diagonal_(0.0) #v 1,2
        return Delta  # (K,K) # v1,2

    
    @torch.no_grad() # v6_c_5_p_5
    def _build_margin_matrix(self):
        """
        From EMA confusion -> normalized pairwise margin matrix M in [0,1], zero diagonal.
        """
        C = self.ema_conf.clone()

        # Symmetrize if desired so margins reflect mutual confusion
        if self.symmetrize:
            C = 0.5 * (C + C.T)

        # Row-normalize to probabilities
        if self.row_norm == "softmax":
            C = torch.softmax(C, dim=1)
        else:  # "l1" (default)
            C = C / (C.sum(dim=1, keepdim=True) + self.tiny)

        # Zero diagonal (we never subtract margin from the target logit)
        M = C - torch.diag(torch.diag(C))
        # Normalize M to [0,1] scale (robust)
        M = M / (M.max() + self.tiny)

        return M

    @torch.no_grad()
    def update_emas(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # update label prior
        batch_counts = torch.bincount(targets, minlength=self.K).float().to(self.device) # confusion matrix of logits and targets
        if self.prior_ema:
            self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (batch_counts + 1e-2)) #v1
        else:
            self.ema_label_counts.add_(batch_counts + 1e-2)

        # update confusion with hard preds, but smoothed
        onehot_y = F.one_hot(targets, num_classes=self.K).float()
        onehot_p = F.one_hot(preds,   num_classes=self.K).float()
        # soften the update a bit by mixing hard & soft
        conf_update = 0.7 * (onehot_y.T @ onehot_p) + 0.3 * (onehot_y.T @ probs)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + 1e-2))

    def forward(self, logits, targets):
        """
        logits: (B, K) linear classifier outputs (before softmax)
        feats:  (B, D) penultimate features
        class_weights: (K, D) classifier weight vectors
        """
        # 1) Update EMA stats (no grad)
        with torch.no_grad():
            self.update_emas(logits.detach(), targets.detach())
            priors = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8) #v1
            # with torch.no_grad(): #v2
            #     C = self.ema_conf.clone()
            #     # normalize each row to get confusion probabilities
            #     priors = C / (C.sum(dim=1, keepdim=True) + self.tiny)
            log_prior = priors.log()  # (K,)

        # 2) Logit-adjusted CE
        la_logits = logits + self.alpha * log_prior.unsqueeze(0)  # stable, diagonal adjustment
        ce = F.cross_entropy(la_logits, targets, reduction='mean')
        return ce
 
class contrastive_separation_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(contrastive_separation_loss, self).__init__()
        self.margin = margin

    def forward(self, resnet_features, far_features):
        # Normalize features to unit sphere
        resnet_norm = F.normalize(resnet_features, p=2, dim=1)
        far_norm = F.normalize(far_features, p=2, dim=1)
        
        # Compute cosine similarities (range [-1, 1])
        similarities = torch.mm(resnet_norm, far_norm.t())
                
        # Convert similarity to distance (range [0, 2])
        distances = 1 - similarities

        loss = F.relu(self.margin - distances)
        
        return loss.mean()
    
class contrastive_together_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(contrastive_together_loss, self).__init__()
        self.margin = margin

    def forward(self, resnet_features, far_features):
        # Normalize features to unit sphere
        resnet_norm = F.normalize(resnet_features, p=2, dim=1)
        far_norm = F.normalize(far_features, p=2, dim=1)
        
        # Compute cosine similarities (range [-1, 1])
        similarities = torch.mm(resnet_norm, far_norm.t())
                
        loss = F.relu(similarities)
        
        return loss.mean()


class RestrictedSoftmaxLoss(nn.Module):
    """
    FedRS: Federated Learning with Restricted Softmax
    Paper: Luo et al., KDD 2021
    
    Only computes softmax over classes present in client's local data.
    This helps with label distribution skew in federated learning.
    """
    def __init__(self, num_classes, local_classes=None, reduction="mean", ignore_index=-100):
        super(RestrictedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.local_classes = local_classes
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def set_local_classes(self, local_classes):
        """Update the local classes for this client"""
        self.local_classes = torch.tensor(local_classes, dtype=torch.long)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K) logits from classifier
            targets: (B,) ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        if self.local_classes is None:
            # Fall back to standard cross-entropy if local classes not set
            return F.cross_entropy(logits, targets, reduction=self.reduction, ignore_index=self.ignore_index)
        
        batch_size, num_classes = logits.size()
        device = logits.device
        
        # Ensure local_classes is on the same device
        if self.local_classes.device != device:
            self.local_classes = self.local_classes.to(device)
        
        # Create mask for local classes
        mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
        mask[self.local_classes] = True
        
        # Mask out non-local classes by setting their logits to very negative value
        masked_logits = logits.clone()
        masked_logits[:, ~mask] = -1e9  # Effectively -inf
        
        # Compute cross-entropy with masked logits
        loss = F.cross_entropy(masked_logits, targets, reduction=self.reduction, ignore_index=self.ignore_index)
        
        return loss


def get_loss_fun(loss):
    if loss == "CE":
        return torch.nn.CrossEntropyLoss
    if loss == "MSE":
        return torch.nn.MSELoss
    if loss == "CL":
        return LabelCalibratedCrossEntropyLoss
    if loss == "CAPA":
        return CAPALoss
    if loss == "FL":
        return FocalLoss
    if loss == "LS":
        return LabelSmoothingCrossEntropy
    if loss == "CB":
        return ClassBalancedCELoss
    if loss == "DB":
        return DBLoss
    if loss == "CACS":
        return CACSLoss
    if loss == "CALC":
        return CACSLoss_LC
    if loss == "LCCA":
        return LabelCalibratedCE_CACS
    if loss == "RS":
        return RestrictedSoftmaxLoss

