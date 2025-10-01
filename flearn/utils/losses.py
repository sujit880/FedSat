import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CustomCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        alpha=1.0,
    ):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

        # Additional custom term
        custom_term = torch.mean(torch.pow(torch.abs(input - target), self.alpha))

        return ce_loss + custom_term


# Example usage
# Assuming input and target are your model's predictions and ground truth labels respectively
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)

# criterion = CustomCrossEntropyLoss(alpha=0.5)  # You can specify your alpha value here
# loss = criterion(input, target)
# print(loss)


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
    
class MSLCrossEntropyLoss_1(nn.Module): # V1 Miss classification aware cross entropy loss
    def __init__(
        self,
        label_distrib=None,
        conf_N=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        tau=1.0,
    ):
        super().__init__()
        self.label_distrib = label_distrib
        self.conf_N = conf_N
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.tau = tau

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)     # hard predictions
            onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            onehot_p = F.one_hot(preds, num_classes=num_classes).float()
            # self.conf_N += onehot_y.T @ onehot_p     # now it's the true hard confusion matrix
            self.conf_N +=onehot_y @ logits #soft
            ##############################################
            mis_matrix = copy.deepcopy(self.conf_N)
            # diag_indices = torch.arange(mis_matrix.size(0))
            # mis_matrix[diag_indices, diag_indices] = 0.0
            ##############################################
            col_sum = mis_matrix.sum(dim=0)
            col_sum[col_sum == 0.0] = 1e-8
            # print(col_sum)
            miss_classification_distrib = col_sum #/ col_sum.sum()
        # print(f"label_distrib: {self.label_distrib} \nmsl_distrib: {miss_classification_distrib}")
        cl_logits = self.tau * torch.pow(self.label_distrib, -1 / 4).expand((logits.shape[0], -1))
        msl_logits = self.tau * torch.pow(miss_classification_distrib, -1 / 4).expand((logits.shape[0], -1))
        # small epsilon to prevent division by zero
        ##################################################################################
        # eps = 1e-6
        # row_distrib = mis_matrix[targets] + eps   # pick rows corresponding to targets
        # row_distrib = row_distrib / row_distrib.sum(dim=1, keepdim=True)  # normalize
        # msl_logits = self.tau * torch.pow(row_distrib, -1/4)
        # # msl_logits = self.tau * (-torch.log(row_distrib))
        ###################################################################################

        # print(f"label_distrib: {cl_logits} \nmsl_distrib: {msl_logits}")
        cal_logit = torch.exp(
            logits - msl_logits
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logits.shape[0]

class MSLCrossEntropyLoss_2(nn.Module):  # v2 reduce loss but not training
    """
    Misclassification-aware calibration loss (per-class temperature scaling).
    Uses the confusion matrix to define class-specific temperatures that 
    soften/sharpen logits before cross-entropy.
    """
    def __init__(
        self,
        label_distrib=None,
        conf_N=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        tau=1.0,
    ):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        self.register_buffer("conf_N", conf_N)

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        # === Update confusion matrix ===
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            onehot_p = F.one_hot(preds, num_classes=num_classes).float()
            self.conf_N += onehot_y.T @ onehot_p  # update confusion stats

        # === Build per-class temperature from confusion ===
        mis_matrix = self.conf_N.clone()
        diag_indices = torch.arange(mis_matrix.size(0))
        mis_matrix[diag_indices, diag_indices] = 0.0  # ignore correct preds

        eps = 1e-6
        row_distrib = mis_matrix[targets] + eps
        row_distrib = row_distrib / row_distrib.sum(dim=1, keepdim=True)

        # Confusion -> temperature scaling
        # If class is often confused, temperature > 1 (softens predictions)
        # If class is confident, temperature ~ 1 (keeps sharpness)
        conf_factor = -torch.log(row_distrib)   # safer than power transform
        temperature = 1.0 + self.tau * conf_factor  # shape [B, C]

        # Apply per-class temperature scaling
        scaled_logits = logits / temperature

        # === Standard cross-entropy ===
        loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)
        return loss

class MSLCrossEntropyLoss_3(nn.Module): #v3
    """
    MaxEnt Confusion-Constrained Calibration (MECC):
    CE trained on a calibrated distribution q obtained by an I-projection of p
    onto linear constraints defined by target confusion (phi) and class prior (pi).

    z' = z - Lambda[y] - mu
    q = softmax(z')

    Dual variables (Lambda, mu) are updated by stochastic dual ascent each batch
    to match the confusion/prior constraints (in expectation).
    """

    def __init__(
        self,
        num_classes: int,
        prior_target: torch.Tensor = None,   # shape [C] (pi); if None, use uniform
        conf_ema_decay: float = 0.99,        # EMA for running confusion counts
        dirichlet_alpha: float = 1.0,        # prior for smoothing phi rows
        dual_lr_lambda: float = 0.05,        # step for Lambda updates
        dual_lr_mu: float = 0.02,            # step for mu updates
        enforce_uniform_prior: bool = False, # else use dataset running prior
        reduction: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.C = num_classes
        self.reduction = reduction
        self.conf_ema_decay = conf_ema_decay
        self.dirichlet_alpha = dirichlet_alpha
        self.dual_lr_lambda = dual_lr_lambda
        self.dual_lr_mu = dual_lr_mu
        self.enforce_uniform_prior = enforce_uniform_prior

        # Running confusion EMA (counts), used to set target phi rows
        self.register_buffer("conf_ema", torch.zeros(self.C, self.C, device=device))

        # Running class frequency EMA for prior target if not uniform
        self.register_buffer("prior_ema", torch.ones(self.C, device=device))

        # Dual variables (NOT learned by backprop; updated by dual ascent)
        self.register_buffer("Lambda", torch.zeros(self.C, self.C, device=device))  # row y, col c
        self.register_buffer("mu", torch.zeros(self.C, device=device))

        # Optional fixed prior target
        if prior_target is not None:
            prior_target = prior_target.to(device)
            self.register_buffer("pi_target_fixed",
                                 prior_target / (prior_target.sum() + 1e-12))
        else:
            self.pi_target_fixed = None

        # For gauge-fixing / identifiability: keep diagonals at 0.
        self.Lambda.data.fill_diagonal_(0.0)

    @torch.no_grad()
    def _targets_from_ema(self):
        """
        Build confusion target phi (rows) from EMA with Dirichlet smoothing,
        and prior target pi from either fixed prior, uniform, or running EMA.
        """
        # Confusion targets per row (exclude diagonal mass)
        counts = self.conf_ema.clone()
        # Dirichlet smoothing to avoid zeros
        counts = counts + self.dirichlet_alpha
        # Zero out diagonal before normalizing off-diagonals
        diag = torch.arange(self.C, device=counts.device)
        counts[:, diag] = 0.0
        # Normalize rows; if a row is all zero, make it uniform off-diagonal
        row_sums = counts.sum(dim=1, keepdim=True) + 1e-12
        phi = counts / row_sums  # shape [C, C], off-diagonals sum to 1 per row (effectively)
        phi[:, diag] = 0.0  # make sure diagonal stays 0 in the target

        # Prior target
        if self.pi_target_fixed is not None:
            pi = self.pi_target_fixed
        elif self.enforce_uniform_prior:
            pi = torch.full((self.C,), 1.0 / self.C, device=counts.device)
        else:
            pi = self.prior_ema / (self.prior_ema.sum() + 1e-12)

        return phi, pi

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, C], unnormalized model scores z
        targets: [B], ground-truth ints in [0..C-1]
        """
        device = logits.device
        B, C = logits.shape
        assert C == self.C, "num_classes mismatch"

        # --- Step 1: Update running statistics (no grad) ---
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # hard preds for EMA only
            onehot_y = F.one_hot(targets, num_classes=C).float()
            onehot_p = F.one_hot(preds,   num_classes=C).float()
            batch_conf = onehot_y.T @ onehot_p  # [C, C]

            # EMA update
            d = self.conf_ema_decay
            self.conf_ema.mul_(d).add_(batch_conf, alpha=(1 - d))
            # Prior EMA from targets
            self.prior_ema.mul_(d).add_(onehot_y.sum(dim=0), alpha=(1 - d))

        # --- Step 2: Build targets phi (confusion) and pi (prior) ---
        with torch.no_grad():
            phi, pi_target = self._targets_from_ema()  # [C,C], [C] V1

            '''q = logits.exp()  # [B, C] V2.... causing nan

            # (a) Confusion constraints: for each y, match off-diagonal mass profile
            # For efficiency, do batched row updates using masked avgs
            for y in torch.unique(targets):
                mask = (targets == y)
                if mask.sum() == 0:
                    continue
                q_y = q[mask]  # [By, C]
                # empirical off-diagonal mean distribution for this row
                emp = q_y.mean(dim=0)  # [C]
                emp_y = emp.clone()
                emp_y[y] = 0.0
                s = emp_y.sum().clamp_min(1e-12)
                emp_y = emp_y / s

                # target off-diagonal row
                tgt = phi[y]  # [C], diagonal is 0 by construction

                # gradient (constraint violation): emp_y - tgt
                grad_row = emp_y - tgt

                # dual ascent step on Lambda row
                self.Lambda[y].add_( self.dual_lr_lambda * grad_row )

                # keep diagonal at 0 (gauge fixing + don't calibrate self-conf)
                self.Lambda[y, y] = 0.0

                # optional centering to remove row-wise mean (identifiability)
                self.Lambda[y] -= self.Lambda[y].mean()

            # (b) Prior constraint: match model marginal q to pi_target
            emp_marg = q.mean(dim=0)  # [C]
            grad_mu = emp_marg - pi_target
            self.mu.add_( self.dual_lr_mu * grad_mu )
            # center mu to remove additive ambiguity
            self.mu -= self.mu.mean() #'''

        # --- Step 3: Calibrated logits via dual variables ---
        # z' = z - Lambda[y] - mu
        # Gather row Lambda[y] for each sample
        Lambda_rows = self.Lambda[targets]                     # [B, C]
        mu_row = self.mu.view(1, C).expand(B, C)               # [B, C]
        # z_prime = logits - Lambda_rows - mu_row #v1,v2
        z_prime = logits - Lambda_rows #v3

        # --- Step 4: Cross-entropy on calibrated distribution ---
        logq = F.log_softmax(z_prime, dim=-1)
        if self.reduction == "mean":
            ce = F.nll_loss(logq, targets, reduction="mean")
        elif self.reduction == "sum":
            ce = F.nll_loss(logq, targets, reduction="sum")
        else:
            ce = F.nll_loss(logq, targets, reduction="none").mean()

        # --- Step 5: Dual ascent to enforce constraints (no backprop to theta) ---
        with torch.no_grad(): #v1
            q = logq.exp()  # [B, C]

            # (a) Confusion constraints: for each y, match off-diagonal mass profile
            # For efficiency, do batched row updates using masked avgs
            for y in torch.unique(targets):
                mask = (targets == y)
                if mask.sum() == 0:
                    continue
                q_y = q[mask]  # [By, C]
                # empirical off-diagonal mean distribution for this row
                emp = q_y.mean(dim=0)  # [C]
                emp_y = emp.clone()
                emp_y[y] = 0.0
                s = emp_y.sum().clamp_min(1e-12)
                emp_y = emp_y / s

                # target off-diagonal row
                tgt = phi[y]  # [C], diagonal is 0 by construction

                # gradient (constraint violation): emp_y - tgt
                grad_row = emp_y - tgt

                # dual ascent step on Lambda row
                self.Lambda[y].add_( self.dual_lr_lambda * grad_row )

                # keep diagonal at 0 (gauge fixing + don't calibrate self-conf)
                self.Lambda[y, y] = 0.0

                # optional centering to remove row-wise mean (identifiability)
                self.Lambda[y] -= self.Lambda[y].mean()

            # (b) Prior constraint: match model marginal q to pi_target
            emp_marg = q.mean(dim=0)  # [C]
            grad_mu = emp_marg - pi_target
            self.mu.add_( self.dual_lr_mu * grad_mu )
            # center mu to remove additive ambiguity
            self.mu -= self.mu.mean() #'''

        return ce
    
class MSLCrossEntropyLoss_4(nn.Module): # V4 Miss classification aware cross entropy loss
    def __init__(
        self,
        label_distrib=None,
        conf_N=None,
        ignore_index=-100,
        reduction="mean",
        tau=1.0,
    ):
        super().__init__()
        self.label_distrib = label_distrib
        self.conf_N = conf_N
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.tau = tau

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            preds = torch.argmax(probs, dim=1)   # use probs, not log probs
            onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            onehot_p = F.one_hot(preds, num_classes=num_classes).float()
            self.conf_N += onehot_y.T @ onehot_p      # proper confusion counts
            col_sum = self.conf_N.sum(dim=0)
            col_sum[col_sum == 0.0] = 1e-8
            # col_sum = col_sum / col_sum.sum()
            
            # msl_bias = self.tau * F.softmax(torch.pow(col_sum, -0.25), dim=0) # v4s

        # class-dependent bias
        msl_bias = self.tau * torch.pow(col_sum, -0.25)   # shape [B,C] v4
        #############################################
        # with torch.no_grad():
        #     row_distrib = self.conf_N[targets] + 1e-8   # pick rows corresponding to targets v41
        #     row_distrib = row_distrib / row_distrib.sum(dim=1, keepdim=True)  # normalize v42
        # msl_bias = self.tau * torch.pow(row_distrib, -1/4)
        ########################################################
        # mcal_logits = logits - msl_bias

        # mcal_logits = torch.exp(
        #     probs - msl_bias.expand((logits.shape[0], -1))
        # )

        mcal_logits = probs - msl_bias.expand((logits.shape[0], -1)) #v4s

        # stable CE
        # log_probs = F.log_softmax(mcal_logits, dim=-1)
        ########
        # log_probs = torch.log(mcal_logits)        
        # loss = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(1))
        # return loss.mean()
        ########
        y_logits = torch.gather(mcal_logits, dim=-1, index=targets.unsqueeze(1))
        loss = -torch.log(y_logits / mcal_logits.sum(dim=-1, keepdim=True))
        
        # print(f"conf: {self.conf_N} \ncolsum: {col_sum} \nnmsl_distrib: {msl_bias} \nlogits: {logits[0]} \nmcal_logits: {mcal_logits[0]} \ny_logits: {y_logits[0]} \nloss: {loss[0]}")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("⚠️ Loss became NaN or Inf!")
            print("Logits:", logits)
            print("Probs:", probs)
            print("ConfM", self.conf_N)
            print("Col_sum", col_sum)
            print("Msl_bias", msl_bias)
            print("Mcalogits:", mcal_logits)
            print("Loss", loss)
            print("Targets:", targets)
            raise ValueError

        return loss.sum() / logits.shape[0]

class MSLCrossEntropyLoss(nn.Module): #v5
    def __init__(self, label_distrib=None, conf_N=None, ignore_index=-100, reduction="mean", tau=1.0):
        super().__init__()
        self.conf_N = conf_N
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.tau = tau

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        with torch.no_grad():
            ################################################################# v5
            # probs = F.softmax(logits, dim=1)
            # onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            # self.conf_N += onehot_y.T @ probs   # use soft probs, not argmax
            ################################################################### v51            
            # probs = F.softmax(logits, dim=1)
            # preds = torch.argmax(probs, dim=1)   # use probs, not log probs
            # onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            # onehot_p = F.one_hot(preds, num_classes=num_classes).float()
            # self.conf_N += onehot_y.T @ onehot_p      # proper confusion counts
            ####################################################################   v52     
            preds = torch.argmax(logits, dim=1)   # use probs, not log probs
            onehot_y = F.one_hot(targets, num_classes=num_classes).float()
            onehot_p = F.one_hot(preds, num_classes=num_classes).float()
            self.conf_N += onehot_y.T @ onehot_p      # proper confusion counts
            ####################################################################
            col_sum = self.conf_N.sum(dim=0)
            col_sum[col_sum == 0.0] = 1e-8
            col_normalized = col_sum / col_sum.max() # v51, v52, v56
            # col_mask = torch.clamp(col_sum, min=1e-8, max=1) #v55

        # class-dependent bias               
        msl_bias = self.tau * torch.pow(col_normalized, -0.25)
        # msl_bias = msl_bias / msl_bias.max() # v54, 55
        # msl_bias = torch.clamp(msl_bias, max=0.5) #v54
        # msl_bias = col_mask * torch.clamp(msl_bias, max=0.5) #v55
        msl_bias = msl_bias.unsqueeze(0).expand_as(logits)

        # apply bias in logit space
        cal_logits = logits - msl_bias

        # stable log-softmax
        # log_probs = F.log_softmax(cal_logits, dim=-1) # v5, v51        
        # log_probs = F.softmax(cal_logits, dim=-1) # v52   
        log_probs = torch.exp(cal_logits) # v53, v56

        # negative log likelihood
        # loss = F.nll_loss(log_probs, targets, reduction=self.reduction,
        #                   ignore_index=self.ignore_index)
        loss = F.cross_entropy(log_probs, targets)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Logits:", logits)
            print("ConfM", self.conf_N)
            print("Col_sum", col_sum)
            print("Msl_bias", msl_bias)
            print("Mcalogits:", log_probs)
            print("Loss", loss)
            print("Targets:", targets)
            raise ValueError

        return loss

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

class CostSensitiveCrossEntropyLoss(nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        if self.is_train:
            gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
            gather_costs = self.cost_matrix[targets, self.predicted]
            # Cost-sensitive loss computation
            loss = -torch.sum(gather_log_probs * gather_costs) / batch_size
            # print(f"\ngathered cost: {gather_costs}, loss: {loss}")
            self.is_train = False
        else:
            log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -log_probs.mean()
        # print(f"loss: {loss}")
        return loss

    def set_train(self):
        self.is_train = True

    def set_cost_matrix(self, cost_matrix, predicted):
        self.is_train = True
        self.cost_matrix = cost_matrix
        self.predicted = predicted

class CostSensitiveCrossEntropyLossN_(torch.nn.Module):
    def __init__(self, cost_matrix, scale=1.0):
        super().__init__()
        self.is_train = False
        self.global_cost_matrix = None
        self.cost_matrix = cost_matrix
        self.initial_cost_matrix = copy.deepcopy(cost_matrix)
        self.scale = scale


    def forward(self, outputs, targets):
        if torch.isnan(outputs).any():
            raise RuntimeError("NaN found in outputs")
      
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        _, predicted = torch.max(log_probs, 1)

        # Update the cost matrix based on the predictions
        for i, (target, pred) in enumerate(zip(targets, predicted)):
            self.cost_matrix[target][pred] += 1

        cost_matrix = copy.deepcopy(self.cost_matrix)
        # cost_matrix = torch.pow(cost_matrix, 1/4)  # Making higher values closer to 1

        if torch.isnan(cost_matrix).any():
            raise RuntimeError("NaN occurred in cost_matrix")

        # Ignore the diagonal elements (correct predictions) making zero
        cost_matrix = cost_matrix * (1 - torch.eye(cost_matrix.shape[0], device=cost_matrix.device))           
        max_value = torch.max(cost_matrix)
        cost_matrix = (cost_matrix / max_value) * (self.scale)
        cost_matrix = cost_matrix + 1 # Adding 1 to make no penalty for correct class

        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        gather_costs = cost_matrix[targets, predicted] 
        # print(gather_costs)

        if self.global_cost_matrix is not None:
            # print("In global")
            # Calculate total predictions for each class (sum of columns)
            class_predictions = self.global_cost_matrix.sum(dim=0)  # Sum over rows to get total predictions per class
            mean_predictions = class_predictions.mean()
            
            under_predicted_classes = class_predictions < mean_predictions * 0.5  # Using 0.5 as threshold for under-prediction
            
            for i, target in enumerate(targets):
                # if over_predicted_classes[target]:
                #     gather_costs[i] = gather_costs[i] * self.gbeta1
                if under_predicted_classes[target]:
                    gather_costs[i] = gather_costs[i] * 1.2
            temp_loss = gather_log_probs * gather_costs
            loss = -temp_loss.mean()
            # loss += 0.05 * kl_loss # torch.nn.functional.mse_loss(outputs, target_outputs)
        else:
            temp_loss = gather_log_probs * gather_costs
            loss = -temp_loss.mean()
        if torch.isnan(loss).any():
            if torch.isnan(outputs).any():
                print("\noutputs:",outputs)
            if torch.isnan(targets).any():
                print("\ntargets:",targets)
            if torch.isnan(gather_costs).any():
                print("\ngather_costs:",gather_costs)
            if torch.isnan(gather_log_probs).any():
                print("gather_log_probs:", gather_log_probs)
            raise RuntimeError("NaN occurred in loss")

        return loss

    def add_global_cost(self, global_cost_matrix):
        self.global_cost_matrix = global_cost_matrix    
    
    def cost_reset(self, cost_matrix= None):
        if cost_matrix is not None:
            self.cost_matrix = cost_matrix
        else: self.cost_matrix = self.initial_cost_matrix

class CostSensitiveCrossEntropyLossN(torch.nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False
        self.global_cost_matrix = None
        self.cost_matrix = cost_matrix
        self.initial_cost_matrix = copy.deepcopy(cost_matrix)
        self.beta = 2.0
        self.beta1 = 1.0
        self.beta2 = 2.0


    def forward(self, outputs, targets):
        if torch.isnan(outputs).any():
            raise RuntimeError("NaN found in outputs")
      
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        _, predicted = torch.max(log_probs, 1)

        # Update the cost matrix based on the predictions
        for i, (target, pred) in enumerate(zip(targets, predicted)):
            self.cost_matrix[target][pred] += 1

        cost_matrix = copy.deepcopy(self.cost_matrix)
        # cost_matrix = torch.pow(cost_matrix, 1/4)  # Making higher values closer to 1

        if torch.isnan(cost_matrix).any():
            raise RuntimeError("NaN occurred in cost_matrix")

        # Ignore the diagonal elements (correct predictions) making zero
        cost_matrix = cost_matrix * (1 - torch.eye(cost_matrix.shape[0], device=cost_matrix.device))
        cost_matrix = cost_matrix + 1 # Adding 1 to remove division by zero possibility

        # Get the min and max values of the cost_matrix
        min_value = torch.min(cost_matrix)
        max_value = torch.max(cost_matrix)

        # Scale the cost_matrix to the range [self.beta1, self.beta2]
        cost_matrix = self.beta1 + ((cost_matrix - min_value) * (self.beta2 - self.beta1) / (max_value - min_value))
        # cost_matrix = 1 + ((cost_matrix - min_value) * (self.beta2 - 1) / max_value)
        
        # # Normalize the cost_matrix to the range [self.beta1, self.beta2]
        # cost_matrix = cost_matrix * (self.beta2 / max_value)
        # cost_matrix = torch.clamp(cost_matrix, min=self.beta1, max=self.beta2)

        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        gather_costs = cost_matrix[targets, predicted] 
        # print(gather_costs)

        if self.global_cost_matrix is not None:
            # print("In global")
            # Calculate total predictions for each class (sum of columns)
            class_predictions = self.global_cost_matrix.sum(dim=0)  # Sum over rows to get total predictions per class
            mean_predictions = class_predictions.mean()
            
            # Identify over-predicted and under-predicted classes
            over_predicted_classes = class_predictions > mean_predictions
            under_predicted_classes = class_predictions < mean_predictions * 0.5  # Using 0.5 as threshold for under-prediction
            
            for i, target in enumerate(targets):
                # if over_predicted_classes[target]:
                #     gather_costs[i] = gather_costs[i] * self.gbeta1
                if under_predicted_classes[target]:
                    gather_costs[i] = gather_costs[i] * 1.2
            temp_loss = gather_log_probs * gather_costs
            loss = -temp_loss.mean()
            # loss += 0.05 * kl_loss # torch.nn.functional.mse_loss(outputs, target_outputs)
        else:
            temp_loss = gather_log_probs * gather_costs
            loss = -temp_loss.mean()
        if torch.isnan(loss).any():
            raise RuntimeError("NaN occurred in loss")

        return loss

# ---------- Helpers ----------

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


def _check_cost(C: torch.Tensor):
    if C.dim() != 2 or C.size(0) != C.size(1):
        raise ValueError("cost_matrix must be square [K,K].")
    if not torch.all(torch.isfinite(C)):
        raise ValueError("cost_matrix contains non-finite values.")


# ---------- Loss ----------

class MisclassificationAwarePairwiseLoss(nn.Module):
    """
    L = λ * CrossEntropy
        + (1-λ) * [ α * ExpectedCost  + (1-α) * PairwiseMargin ]

    ExpectedCost:     sum_j C[y, j] * p(j)
    PairwiseMargin:   sum_{j != y} C[y, j] * softplus(m + z_j - z_y)

    Notes:
    - No argmax inside forward -> smooth gradients.
    - The cost matrix C can be updated between epochs (EMA from confusion).
    - Keep diag(C)=0. Off-diagonals encode how bad y->j mistakes are.
    """

    def __init__(self,
                 cost_matrix: torch.Tensor,
                 lam: float = 0.5,
                 alpha: float = 0.5,
                 margin: float = 0.0,
                 normalize_pairwise: bool = True,
                 focal_gamma: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        _check_cost(cost_matrix)
        self.register_buffer("C", cost_matrix.clone().float())  # [K,K]
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.normalize_pairwise = bool(normalize_pairwise)
        self.focal_gamma = float(focal_gamma)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    @torch.no_grad()
    def set_cost_matrix(self, cost_matrix: torch.Tensor):
        """Hard-set the cost matrix (e.g., after your own preprocessing)."""
        _check_cost(cost_matrix)
        self.C.copy_(cost_matrix.float())

    @torch.no_grad()
    def ema_update_from_confusion(self, conf: torch.Tensor,
                                  alpha: float = 0.1,
                                  w_min: float = 1.0, w_max: float = 2.0,
                                  tau: float = 10.0, gamma: float = 1.0):
        """
        Build C_new from confusion and blend: C <- (1-alpha)C + alpha*C_new
        Call this BETWEEN epochs/loops, not inside forward.
        """
        C_new = confusion_to_cost(conf, w_min=w_min, w_max=w_max, tau=tau, gamma=gamma).to(self.C.device)
        self.C.mul_(1.0 - alpha).add_(alpha * C_new)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ---- Anchor: standard CE ----
        ce = F.cross_entropy(logits, targets)

        # ---- Expected-cost term (distribution-aware) ----
        p = torch.softmax(logits, dim=1)                 # [B,K]
        C_rows = self.C[targets]                         # [B,K]
        exp_cost = (p * C_rows).sum(dim=1)               # [B]
        if self.focal_gamma > 0.0:
            p_y = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            exp_cost = exp_cost * ((1.0 - p_y).clamp_min(1e-8) ** self.focal_gamma)

        # Reduce
        if self.reduction == "mean":
            exp_cost = exp_cost.mean()
        else:
            exp_cost = exp_cost.sum()

        # ---- Pairwise margin term (logit-vs-each-wrong) ----
        B, K = logits.shape
        z_y = logits.gather(1, targets.unsqueeze(1))     # [B,1]
        # m + (z_j - z_y)
        margins = self.margin + (logits - z_y)           # [B,K]
        pair_pen = F.softplus(margins)                   # [B,K], smooth hinge

        # zero diagonal
        diag_mask = torch.zeros_like(pair_pen)
        diag_mask[torch.arange(B, device=logits.device), targets] = 1.0
        pair_pen = pair_pen * (1.0 - diag_mask)

        # weight by row of C for each target
        weights = C_rows                                 # [B,K]
        pair_term = (weights * pair_pen).sum(dim=1)      # [B]

        # Optional normalization to keep scale stable across K
        if self.normalize_pairwise and K > 1:
            pair_term = pair_term / (K - 1)

        if self.reduction == "mean":
            pair_term = pair_term.mean()
        else:
            pair_term = pair_term.sum()

        # ---- Combine ----
        aux = self.alpha * exp_cost + (1.0 - self.alpha) * pair_term
        loss = self.lam * ce + (1.0 - self.lam) * aux
        return loss

class MisclassificationAwarePairwiseLoss1(nn.Module):
    def __init__(self,
                 cost_matrix: torch.Tensor,
                 lam: float = 0.9,        # favor CE more
                 alpha: float = 0.9,      # favor ExpectedCost more
                 margin: float = 0.1,     # small positive margin
                 normalize_pairwise: bool = True,
                 focal_gamma: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        _check_cost(cost_matrix)
        self.register_buffer("C", cost_matrix.clone().float())
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.normalize_pairwise = bool(normalize_pairwise)
        self.focal_gamma = float(focal_gamma)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction=self.reduction)

        # Expected-cost term
        p = torch.softmax(logits, dim=1)          # [B,K]
        C_rows = self.C[targets]                  # [B,K]
        exp_cost = (p * C_rows).sum(dim=1)        # [B]
        if self.focal_gamma > 0.0:
            p_y = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            exp_cost = exp_cost * ((1.0 - p_y).clamp_min(1e-8) ** self.focal_gamma)
        exp_cost = exp_cost.mean() if self.reduction == "mean" else exp_cost.sum()

        # Pairwise margin term
        B, K = logits.shape
        z_y = logits.gather(1, targets.unsqueeze(1))
        margins = self.margin + (logits - z_y)
        pair_pen = F.softplus(margins)

        diag_mask = torch.zeros_like(pair_pen)
        diag_mask[torch.arange(B, device=logits.device), targets] = 1.0
        pair_pen = pair_pen * (1.0 - diag_mask)

        pair_term = (C_rows * pair_pen).sum(dim=1)
        if self.normalize_pairwise and K > 1:
            pair_term = pair_term / (K - 1)
        pair_term = pair_term.mean() if self.reduction == "mean" else pair_term.sum()

        # Balance auxiliary terms
        aux = self.alpha * exp_cost + (1.0 - self.alpha) * pair_term
        # Normalize to CE scale
        aux = aux / (aux.detach().mean() + 1e-8) * ce.detach().mean()

        loss = self.lam * ce + (1.0 - self.lam) * aux
        return loss

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

class MCAPALoss(torch.nn.Module):
    def __init__(self, device, K, lam=0.5, mu=0.01, mtau=0.5, tau=2.0, margin=0.05, use_gate=True):
        """
        W: (K,K) tensor with rows summing to 1, diag=0
        U: (K,)   tensor summing to 1 (can be all zeros initially)
        """
        super().__init__()
        # safe initial weights: uniform off-diagonal for W, zeros for U
        self.device = device
        self.W = (torch.ones(K, K, device=self.device) - torch.eye(K, device=self.device)) / (K - 1)
        self.U = torch.zeros(K, device=self.device)
        # soft confusion trackers (float, with small prior)
        prior = 0.5
        self.conf_N = torch.full((K, K), prior, device=self.device)
        self.pred_q = torch.full((K,),     prior, device=self.device)
        self.label_y = torch.full((K,),    prior, device=self.device)
        self.lam = lam
        self.mu = mu
        self.mtau = mtau
        self.tau = tau
        self.margin = margin
        self.use_gate = use_gate

    def forward(self, logits, targets):
        # print(f"lam: {self.lam}, mu: {self.mu}, margin: {self.margin}")
        # logits: (B,K), targets: (B,)
        B, K = logits.size()
        probs = F.softmax(logits, dim=1)

        # ----- MSLCE -----
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)      # soft predictions (B,K)
            preds = torch.argmax(logits, dim=1)     # hard predictions (B,K)
            onehot_p = F.one_hot(preds, num_classes=K).float()
            onehot_y = F.one_hot(targets, num_classes=K).float()          # (B,K)
            # soft confusion: add prob row to the true-class row
            self.conf_N += onehot_y.T @ onehot_p                            # (K,K)
            self.pred_q += probs.sum(dim=0)                              # (K,)
            self.label_y += onehot_y.sum(dim=0)                          # (K,)
            col_sum = self.conf_N.sum(dim=0)
            col_sum = col_sum / (self.label_y + 1) # ration PredL/TL
            col_sum[col_sum == 0.0] = 1e-8
            col_normalized = col_sum / col_sum.max() 

        # class-dependent bias               
        msl_bias = self.mtau * torch.pow(col_normalized, -0.25)
        msl_bias = msl_bias / msl_bias.max() 
        # nmsl_bias = col_normalized * torch.clamp(msl_bias, max=0.5) #v1
        # msl_logits = nmsl_bias.unsqueeze(0).expand_as(logits) # v1
        msl_bias = msl_bias * (self.label_y  / (self.label_y.sum() + 1)) # v2
        # msl_bias = torch.clamp(msl_bias, max=0.5) # v2 comment out for v3
        msl_logits = msl_bias.unsqueeze(0).expand_as(logits) # v2

        # apply bias in logit space
        cal_logits = logits - msl_logits    
        log_probs = F.softmax(cal_logits, dim=-1) 
        ce = F.cross_entropy(log_probs, targets, reduction='none')

        # ----- Pairwise confusion term -----
        self.W, self.U = self.build_W_U(self.conf_N, self.pred_q, self.label_y,
                                    beta=1.0, gamma=2.0, eps=1e-6)
        z_a = logits.gather(1, targets.view(-1,1))                     # (B,1)
        # (B,K) margins: z_b - z_a + m
        margins = logits - z_a + self.margin
        pair_mask = torch.ones_like(margins, dtype=torch.bool)
        pair_mask[torch.arange(B), targets] = False                    # exclude b=a

        # select W rows per target
        W_rows = self.W[targets]                                       # (B,K)
        # softplus(τ * margin)
        splus = F.softplus(self.tau * margins) # default 2.0
        pair_term = (W_rows * splus).masked_select(pair_mask).view(B, K-1).sum(dim=1)

        # ----- Anti-bias term -----
        U = self.U.view(1, -1).expand(B, K)                            # (B,K)
        anti_mask = pair_mask                                          # exclude target class
        anti_bias = (U * probs).masked_select(anti_mask).view(B, K-1).sum(dim=1)#'''

        loss = (1-self.lam)*ce + self.lam * pair_term + self.mu * anti_bias
        # loss = ce + self.lam * pair_term + self.mu * anti_bias
        # loss = pair_term + self.lam * ce + self.mu * anti_bias
        return loss.mean()


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

class MCALoss(torch.nn.Module):
    def __init__(self, device, K, lam=0.5, mu=0.01, mtau=0.5, tau=2.0, margin=0.05, use_gate=True):
        """
        W: (K,K) tensor with rows summing to 1, diag=0
        U: (K,)   tensor summing to 1 (can be all zeros initially)
        """
        super().__init__()
        # safe initial weights: uniform off-diagonal for W, zeros for U
        self.device = device
        self.W = (torch.ones(K, K, device=self.device) - torch.eye(K, device=self.device)) / (K - 1)
        self.U = torch.zeros(K, device=self.device)
        # soft confusion trackers (float, with small prior)
        prior = 0.5
        self.conf_N = torch.full((K, K), prior, device=self.device)
        self.pred_q = torch.full((K,),     prior, device=self.device)
        self.label_y = torch.full((K,),    prior, device=self.device)
        self.lam = lam
        self.mu = mu
        self.mtau = mtau
        self.tau = tau
        self.margin = margin
        self.use_gate = use_gate

    def forward(self, logits, targets):
        # print(f"lam: {self.lam}, mu: {self.mu}, margin: {self.margin}")
        # logits: (B,K), targets: (B,)
        B, K = logits.size()
        probs = F.softmax(logits, dim=1)

        # ----- MSLCE -----
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)      # soft predictions (B,K)
            preds = torch.argmax(logits, dim=1)     # hard predictions (B,K)
            onehot_p = F.one_hot(preds, num_classes=K).float()
            onehot_y = F.one_hot(targets, num_classes=K).float()          # (B,K)
            # soft confusion: add prob row to the true-class row
            # self.conf_N += onehot_y.T @ onehot_p                            # (K,K)
            self.conf_N += onehot_y.T @ probs                            # (K,K)  v4
            self.pred_q += onehot_p.sum(dim=0)                              # (K,)
            self.label_y += onehot_y.sum(dim=0)                          # (K,)
            col_sum = self.conf_N.sum(dim=0)
            col_sum = col_sum / (self.label_y + 1) # ration PredL/TL
            col_sum[col_sum == 0.0] = 1e-8
        self.W, self.U = self.build_W_U(self.conf_N, self.pred_q, self.label_y, beta=1.0, gamma=2.0, eps=1e-6)
        # class-dependent bias               
        # msl_bias = self.mtau * torch.pow(col_sum, -0.25)
        msl_bias = torch.pow(col_sum, -0.25)
        msl_bias = msl_bias / msl_bias.max() 
        msl_logits = msl_bias.unsqueeze(0).expand_as(logits) 
        # select W rows per target
        W_rows = self.W[targets]                                       # (B,K)
        msl_logits = W_rows * msl_logits # v1,2,7
        # msl_logits = 0.5 * (W_rows * msl_logits + msl_bias * (self.label_y  / (self.label_y.sum() + 1))) # v3,4,5,6

        # apply bias in logit space
        cal_logits = logits - msl_logits    
        ce = F.cross_entropy(cal_logits, targets, reduction='none') #v1,7
        # log_probs = F.softmax(cal_logits, dim=-1) #v2,3,4,5,6
        # ce = F.cross_entropy(log_probs, targets, reduction='none') #v2,3,4,5,6

        # ----- Pairwise confusion term -----       
        z_a = logits.gather(1, targets.view(-1,1))                     # (B,1)
        # (B,K) margins: z_b - z_a + m
        margins = logits - z_a + self.margin
        pair_mask = torch.ones_like(margins, dtype=torch.bool)
        pair_mask[torch.arange(B), targets] = False                    # exclude b=a
        # softplus(τ * margin)
        splus = F.softplus(self.tau * margins) # default 2.0
        pair_term = (W_rows * splus).masked_select(pair_mask).view(B, K-1).sum(dim=1)

        # ----- Anti-bias term -----
        # U = self.U.view(1, -1).expand(B, K)                            # (B,K)
        # anti_mask = pair_mask                                          # exclude target class
        # anti_bias = (U * probs).masked_select(anti_mask).view(B, K-1).sum(dim=1)#'''

        # loss = (1-self.lam)*ce.mean() + self.lam * pair_term.mean() + self.mu * anti_bias.mean() #v5
        # loss = ce.mean() + self.lam * pair_term.mean() + self.mu * anti_bias.mean() 
        # loss = ce.mean() + self.lam * pair_term.mean() #v2,3,4
        # loss = (1-self.lam)*ce.mean() + self.lam * pair_term.mean() #v5
        # loss =  pair_term.mean() #v6
        # loss = ce.mean() #v7
        # loss = ce.mean() + pair_term.mean() #v72
        # loss = ce.mean() + 0.25 * pair_term.mean() #v72_25
        loss = ce.mean() + 0.025 * pair_term.mean() #v72B_025
        return loss


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

class DBCCLoss(nn.Module):
    def __init__(self, K, alpha=1.0, lambda_cc=0.1, topM=3, ema_m=0.99, warmup_epochs=2, 
                 rare_temp=0.07, head_temp=0.10, prior_ema=True, device=None):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.lambda_cc = lambda_cc
        # self.topM = topM # v1,2 =3
        self.topM = int((K/10)*topM) # v3C0_1
        self.ema_m = ema_m
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.prior_ema = prior_ema
        self.device = device

        # EMA priors (labels) and confusion (rows: true, cols: predicted)
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))  # small symmetric prior
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))

        # per-class temperature (rarer -> lower temp -> stronger push)
        self.rare_temp = rare_temp
        self.head_temp = head_temp

    @torch.no_grad()
    def update_emas(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # update label prior
        batch_counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        if self.prior_ema:
            self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (batch_counts + 1e-2))
        else:
            self.ema_label_counts.add_(batch_counts + 1e-2)

        # update confusion with hard preds, but smoothed
        onehot_y = F.one_hot(targets, num_classes=self.K).float()
        onehot_p = F.one_hot(preds,   num_classes=self.K).float()
        # soften the update a bit by mixing hard & soft
        conf_update = 0.7 * (onehot_y.T @ onehot_p) + 0.3 * (onehot_y.T @ probs)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + 1e-2))

    def forward(self, logits, targets, feats, class_weights):
        """
        logits: (B, K) linear classifier outputs (before softmax)
        feats:  (B, D) penultimate features
        class_weights: (K, D) classifier weight vectors
        """
        B, K = logits.shape
        device = logits.device

        # 1) Update EMA stats (no grad)
        with torch.no_grad():
            self.update_emas(logits.detach(), targets.detach())
            priors = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)
            log_prior = priors.log()  # (K,)

        # 2) Logit-adjusted CE
        la_logits = logits + self.alpha * log_prior.unsqueeze(0)  # stable, diagonal adjustment
        ce = F.cross_entropy(la_logits, targets, reduction='mean')

        # early warmup -> only CE
        if self.epoch < self.warmup_epochs or self.lambda_cc <= 0:
            return ce

        # 3) Confusion-focused InfoNCE
        # print("Confusion focused") # v2
        v2 = False
        if v2:
            with torch.no_grad():
                C = self.ema_conf.clone()
                # normalize each row to get confusion probabilities
                C = C / (C.sum(dim=1, keepdim=True) + self.tiny)

                # pick top-M confusing negatives per target class
                top_neg_idx = torch.topk(C[targets], k=min(self.topM, K-1), dim=1).indices  # (B, M)

                # per-class temperature: rarer classes get slightly lower temp => stronger separation
                freq = priors[targets]  # (B,)
                temps = torch.where(freq < priors.median(), 
                                    torch.full_like(freq, self.rare_temp),
                                    torch.full_like(freq, self.head_temp))

                # weights for negatives from confusion probs (stop-grad, row-normalized)
                neg_w = torch.gather(C[targets], dim=1, index=top_neg_idx)  # (B, M)
                neg_w = neg_w / (neg_w.sum(dim=1, keepdim=True) + self.tiny)

            # cosine similarities
            h = F.normalize(feats, dim=1)                # (B, D)
            W = F.normalize(class_weights, dim=1)        # (K, D)
            pos_sim = (h * W[targets]).sum(dim=1)        # (B,)
            neg_sim = torch.einsum('bd,bmd->bm', h, W[top_neg_idx])  # (B, M)

            # InfoNCE with per-sample temperature
            pos_term = (pos_sim / temps).exp()           # (B,)
            neg_term = ((neg_sim / temps.unsqueeze(1)).exp() * neg_w).sum(dim=1)  # (B,)
            cc = -torch.log( (pos_term + 1e-12) / (pos_term + neg_term + 1e-12) ).mean()

            return ce + self.lambda_cc * cc
        v4=True # v4C0_1m0_2a0_5
        if v4:
            # Build adaptive margins from confusion graph
            m0 = 0.2 # margin
            alpha = 0.5 # scale
            with torch.no_grad():
                A = 0.5 * (self.ema_conf + self.ema_conf.T)  # symmetrized confusion
                A = A / (A.max() + 1e-8)                     # normalize [0,1]
                M = m0 + alpha * A                           # (K,K) adaptive margins

            # logits: (B,K), feats normalized, W normalized
            cos_logits = F.linear(F.normalize(feats), F.normalize(class_weights))

            # Apply adaptive margin: subtract M[y,j] from each negative logit
            margin_mat = M[targets]  # (B,K)
            margin_mat[torch.arange(B), targets] = 0.0
            caal_logits = cos_logits - margin_mat

            caal_loss = F.cross_entropy(caal_logits, targets)
            return ce + self.lambda_cc * caal_loss


    def step_epoch(self):
        self.epoch += 1
    
    def set_epoch(self, epoch): #v2 and call in fedvag client
        self.epoch = epoch

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
        la_logits = logits + self.alpha * log_prior.unsqueeze(0)  # stable, diagonal adjustment
        ce = F.cross_entropy(la_logits, targets, reduction='mean')
        return ce
    
class CBregLoss(nn.Module):
    def __init__(self, K, lambda_diag=0.05, lambda_off=0.05, ema_m=0.995, device=None):
        super().__init__()
        self.K = K
        self.lambda_diag = lambda_diag
        self.lambda_off  = lambda_off
        self.ema_m = ema_m
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))  # mild prior
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))

    @torch.no_grad()
    def _update_conf(self, logits, targets):
        P = F.softmax(logits, dim=1)                        # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()  # (B,K)
        C_update = Y.T @ P                                  # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (C_update + 1e-3))

    def forward(self, logits, targets):
        with torch.no_grad():
            self._update_conf(logits.detach(), targets.detach())
            C = self.ema_conf / (self.ema_conf.sum(dim=1, keepdim=True) + self.tiny)  # row-stochastic

        ce = F.cross_entropy(logits, targets, reduction='mean')

        # confusion Bregman penalties
        diag = torch.clip(torch.diag(C), min=1e-6)
        diag_term = -torch.log(diag).mean()                         # ↑ trace(C)
        off = C - torch.diag(torch.diag(C))
        off_term = (off**2).sum() / (self.K*(self.K-1))             # ↓ off-diagonal mass

        loss = ce + self.lambda_diag*diag_term + self.lambda_off*off_term
        return loss

class CBregLoss(nn.Module):
    def __init__(self, K, lambda_diag=0.05, lambda_off=0.05, ema_m=0.995, device=None):
        super().__init__()
        self.K = K
        self.lambda_diag = lambda_diag
        self.lambda_off  = lambda_off
        self.ema_m = ema_m
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))  # mild prior
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))

    @torch.no_grad()
    def _update_conf(self, logits, targets):
        P = F.softmax(logits, dim=1)                        # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()  # (B,K)
        C_update = Y.T @ P                                  # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (C_update + 1e-3))

    def forward(self, logits, targets):
        with torch.no_grad():
            self._update_conf(logits.detach(), targets.detach())
            C = self.ema_conf / (self.ema_conf.sum(dim=1, keepdim=True) + self.tiny)  # row-stochastic

        ce = F.cross_entropy(logits, targets, reduction='mean')

        # confusion Bregman penalties
        diag = torch.clip(torch.diag(C), min=1e-6)
        diag_term = -torch.log(diag).mean()                         # ↑ trace(C)
        off = C - torch.diag(torch.diag(C))
        off_term = (off**2).sum() / (self.K*(self.K-1))             # ↓ off-diagonal mass

        loss = ce + self.lambda_diag*diag_term + self.lambda_off*off_term
        return loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CALBLoss(nn.Module):
    """
    Confusion-Aware Logit Bias (CALB)
    - Learns an EMA confusion matrix C (rows: true class, cols: predicted)
    - Builds a pairwise margin matrix M from C (bigger when y->j is often confused)
    - Calibrates logits per-sample: z'_j = z_j - scale * M[y, j], with M[y,y]=0
    - (Optional) Also adds Bayes-consistent logit adjustment by priors (LA)
    """
    def __init__(
        self,
        K,
        scale=0.5,             # strength for confusion-derived margins (0.2–1.0)
        ema_m=0.995,           # EMA momentum for confusion and label counts
        la_alpha=0.0,          # 0 to disable LA; 0.5–1.0 typical if enabled
        warmup_steps=0,        # number of forward() calls before enabling confusion bias
        row_norm="l1",         # how to normalize rows of C -> M ("l1" or "softmax")
        symmetrize=True,       # use A=(C+C^T)/2 so pairwise margins reflect mutual confusion
        min_floor=1e-3,        # tiny additive prior to keep entries > 0
        device=None
    ):
        super().__init__()
        self.K = K
        self.scale = scale
        self.ema_m = ema_m
        self.la_alpha = la_alpha
        self.warmup_steps = warmup_steps
        self.row_norm = row_norm
        self.symmetrize = symmetrize
        self.min_floor = min_floor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # EMA stats
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))   # soft confusion
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))

        # step counter for warmup gating
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def _update_emas(self, logits, targets):
        """
        - Update soft confusion by accumulating predicted probabilities into true-class rows.
        - Update label prior counts (for optional LA).
        """
        P = F.softmax(logits, dim=1)                                 # (B, K)
        Y = F.one_hot(targets, num_classes=self.K).float()           # (B, K)

        # Soft confusion update: (K,K)
        conf_update = Y.T @ P                                        # add prob mass to rows of true class
        self.ema_conf.mul_(self.ema_m).add_((1.0 - self.ema_m) * (conf_update + self.min_floor))

        # Label prior update
        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1.0 - self.ema_m) * (counts + self.min_floor))

        # step
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)

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
    def _build_log_prior(self):
        priors = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)
        return priors.log()  # (K,)

    def forward(self, logits, targets):
        """
        Inputs:
          logits: (B, K) pre-softmax scores
          targets: (B,)
        Output:
          scalar loss tensor (on same device as logits)
        """
        # Update EMA stats (no grad)
        with torch.no_grad():
            self._update_emas(logits.detach(), targets.detach())

        # Optional Bayes-consistent logit adjustment (vector)
        if self.la_alpha > 0.0:
            log_prior = self._build_log_prior()          # (K,)
            logits = logits + self.la_alpha * log_prior.unsqueeze(0)

        # Confusion-aware matrix shift after warmup
        if self.steps.item() >= self.warmup_steps and self.scale > 0.0:
            with torch.no_grad():
                M = self._build_margin_matrix()          # (K, K)

            margin_rows = M[targets]                     # (B, K)
            # Ensure zero on the target class index
            margin_rows[torch.arange(targets.size(0), device=targets.device), targets] = 0.0

            # Apply calibrated subtraction to logits
            logits = logits - self.scale * margin_rows

        # Standard CE on calibrated logits
        loss = F.cross_entropy(logits, targets, reduction='mean')

        # During evaluation, many frameworks expect CPU scalar; if you face numpy issues, uncomment:
        # if not self.training:
        #     return loss.detach().cpu()

        return loss

class CACSLoss(nn.Module):
    """
    Confusion-Aware Cost-Sensitive Multiclass Logistic
    L(z,y) = -z_y + log sum_j exp(z_j + Delta[y,j])
    where Delta[y,j] >= 0 encodes pairwise cost from EMA confusion.
    Optional: add Bayes prior bias b_j = beta * log pi_j to logits.
    """
    def __init__(self, K, m0=0.1, alpha=0.5, ema_m=0.995, warmup_steps=0,
                 prior_beta=0.0, conf_beta=0.5, min_floor=1e-3, device=None):
        super().__init__()
        self.K = K
        self.m0 = m0
        self.alpha = alpha
        self.ema_m = ema_m
        self.warmup_steps = warmup_steps
        self.prior_beta = prior_beta
        self.conf_beta = conf_beta #v2_c_5
        self.min_floor = min_floor
        self.device = device

        # EMA stats
        self.register_buffer("ema_conf", torch.ones(K, K, device=self.device))       # soft confusion
        self.register_buffer("ema_label_counts", torch.ones(K, device=self.device))  # for priors
        self.register_buffer("tiny", torch.tensor(1e-8, device=self.device))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def _update_emas(self, logits, targets):
        P = F.softmax(logits, dim=1)                                # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()          # (B,K)
        conf_update = Y.T @ P                                       # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.min_floor))

        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (counts + self.min_floor))
        
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)
    @torch.no_grad()
    def _update_emas_3(self, logits, targets):
        Pred = torch.argmax(logits, dim=1)                          # (B,1)
        P = F.one_hot(Pred, num_classes=self.K).float()             # (B,K)
        Y = F.one_hot(targets, num_classes=self.K).float()          # (B,K)
        conf_update = Y.T @ P                                       # (K,K)
        self.ema_conf.mul_(self.ema_m).add_((1 - self.ema_m) * (conf_update + self.min_floor))

        counts = torch.bincount(targets, minlength=self.K).float().to(self.device)
        self.ema_label_counts.mul_(self.ema_m).add_((1 - self.ema_m) * (counts + self.min_floor))
        
        if self.steps.item() <= self.warmup_steps:
            self.steps += int(len(logits)/2)
    @torch.no_grad()
    def _update_emas_4(self, logits, targets):
        self.ema_m = 0.95
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

    @torch.no_grad() #v3
    def _build_Delta_3(self):
        # symmetric, row-normalized confusion graph in [0,1], zero diag
        C = self.ema_conf.clone()
        # A = 0.5 * (C + C.T)
        A = C + C.T
        A = A / (A.sum(dim=1, keepdim=True) + self.tiny)
        A = A - torch.diag(torch.diag(A))
        Delta = 1e-6 + A #v3
        Delta.fill_diagonal_(1.0)
        # return Delta.log() #v3
        Delta.exp() #v31
        Delta = torch.clamp(Delta, min=1, max=2)
        return Delta

    @torch.no_grad() # v1,2
    def _log_prior(self):
        pi = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)
        return pi.log()  # (K,)    
    @torch.no_grad() #v4
    def _log_prior_3(self):        
        C = self.ema_conf.clone()
        C = C.sum(dim=0)
        pi = (C / (C.sum() + self.tiny)).clamp_min(1e-8)
        return pi.log()  # (K,) 
    @torch.no_grad() #v5
    def _log_prior_4(self):        
        C = self.ema_conf.clone()
        C = C + 1e-3
        pi = (C / (C.sum())).clamp_min(self.tiny)
        return pi.log() * pi.exp()  # (K,) # v5=pi.log,v51=pi.log * pi.exp      
    @torch.no_grad() # v52=lpi.log * pi.exp       
    def _log_prior_5(self):
        lpi = (self.ema_label_counts / (self.ema_label_counts.sum() + self.tiny)).clamp_min(1e-8)        
        C = self.ema_conf.clone()
        C = C + 1e-3
        pi = (C / (C.sum())).clamp_min(self.tiny)
        return lpi.log() * pi.exp()  # (K,) # v5=pi.log,v51=pi.log * pi.exp

    def forward(self, logits, targets):
        with torch.no_grad():
            # self._update_emas(logits.detach(), targets.detach()) #v1,2
            # self._update_emas_3(logits.clone().detach(), targets.clone().detach()) #v3,4
            self._update_emas_4(logits.clone().detach(), targets.clone().detach()) # v5

        z = logits
        # Optional Bayes logit adjustment
        if self.prior_beta > 0.0:
            # z = z + self.prior_beta * self._log_prior().unsqueeze(0) # v1,2
            # z = z + self.prior_beta * self._log_prior_3().unsqueeze(0) # v4
            z = z + self.prior_beta * self._log_prior_5()[targets] # v4

        # Build Delta after warmup
        if self.steps.item() >= self.warmup_steps and (self.m0 > 0.0 or self.alpha > 0.0):
            with torch.no_grad():
                Delta = self._build_Delta()                   # (K,K) v1,2,5,52
                # Delta = self._build_Delta_3()                   # (K,K) v3
            D_y = Delta[targets]                              # (B,K)
        else:
            # no cost yet
            D_y = torch.zeros_like(z)

        # Cost-augmented softmax log-loss:
        # L = -z_y + logsumexp(z_j + Delta[y,j])
        z_plus = z + D_y # v1        
        # z_plus = z + self.conf_beta * D_y # v2_c_5
        # z_plus = z #+ D_y * self._log_prior_3().unsqueeze(0).expand_as(z) # v4,5
        lse = torch.logsumexp(z_plus, dim=1)                  # (B,)
        zy = z.gather(1, targets.view(-1,1)).squeeze(1)       # (B,)
        loss = (lse - zy).mean()
        # loss = F.cross_entropy(z, targets, reduction="mean")

        return loss

def cross_entropy_loss(outputs, targets):
    """
    Manually computed cross-entropy loss for educational purposes.

    Args:
    - outputs (torch.Tensor): The raw logits output by the neural network.
                              Shape: [batch_size, num_classes]
    - targets (torch.Tensor): The ground truth labels.
                              Shape: [batch_size]
                              Each label is an integer in [0, num_classes-1].

    Returns:
    - loss (torch.Tensor): The mean cross-entropy loss.
    """
    # Step 1: Compute log softmax
    # log_softmax(x_i) = log(exp(x_i) / sum_j(exp(x_j)))
    log_probs = F.log_softmax(outputs, dim=1)

    # Step 2: Gather the log probabilities of the correct classes
    # targets.unsqueeze(1) changes shape from [batch_size] to [batch_size, 1]
    # gather(dim=1, index=targets.unsqueeze(1)) picks out the log_probs for each target class
    log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Step 3: Compute the negative log likelihood loss
    # nll_loss = -1 * (sum of log_probs for the correct classes) / batch_size
    loss = -log_probs.mean()

    return loss

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


def get_loss_fun(loss):
    if loss == "CE":
        return torch.nn.CrossEntropyLoss
    if loss == "MSE":
        return torch.nn.MSELoss
    if loss == "CL":
        return LabelCalibratedCrossEntropyLoss
    if loss == "CS":
        return CostSensitiveCrossEntropyLoss
    if loss == "CSN":
        return CostSensitiveCrossEntropyLossN
    if loss == "PSL":
        return MisclassificationAwarePairwiseLoss
    if loss == "PSL1":
        return MisclassificationAwarePairwiseLoss1
    if loss == "CAPA":
        return CAPALoss
    if loss == "MCAPA":
        return MCAPALoss
    if loss == "FL":
        return FocalLoss
    if loss == "LS":
        return LabelSmoothingCrossEntropy
    if loss == "CB":
        return ClassBalancedCELoss
    if loss == "MSL":
        return MSLCrossEntropyLoss
    if loss == "MCA":
        return MCALoss
    if loss == "DBCC":
        return DBCCLoss
    if loss == "DB":
        return DBLoss
    if loss == "CALB":
        return CALBLoss
    if loss == "CACS":
        return CACSLoss
