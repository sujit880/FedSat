import torch
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple

from flearn.clients.fedavg import FedAvgClient
from flearn.data.dataset import CLASSES


class CCVRClient(FedAvgClient):
    """
    FedAvg client with extra step to compute per-class feature statistics for CCVR:
    - counts[c]
    - sum[c, d]
    - sumsq[c, d]
    Feature extractor is model.get_representation_features(x).
    """

    @torch.no_grad()
    def compute_feature_statistics(self) -> Dict[str, torch.Tensor]:
        device = self.device
        self.model.eval()
        K = CLASSES[self.dataset]

        # Determine feature dim: prefer model.get_feature_dim(), fallback to a tiny pass
        feat_dim = None
        if hasattr(self.model, 'get_feature_dim'):
            try:
                feat_dim = int(self.model.get_feature_dim())
            except Exception:
                feat_dim = None
        if feat_dim is None:
            for xb, yb in self.trainloader:
                xb = xb.to(device, non_blocking=True)
                feats = self.model.get_representation_features(xb)
                feat_dim = feats.shape[1]
                break
        if feat_dim is None:
            # no data (edge case) but keep correct dim if classifier known (rare)
            feat_dim = 1

        counts = torch.zeros(K, dtype=torch.long, device=device)
        sum_feat = torch.zeros(K, feat_dim, dtype=torch.float32, device=device)
        sumsq_feat = torch.zeros(K, feat_dim, dtype=torch.float32, device=device)

        for xb, yb in self.trainloader:
            if yb.numel() == 0:
                continue
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            feats = self.model.get_representation_features(xb)
            feats = feats.detach()
            # accumulate per-class
            for c in yb.unique():
                c = int(c.item())
                idx = (yb == c)
                if not idx.any():
                    continue
                f = feats[idx]
                counts[c] += idx.sum().to(counts.dtype)
                sum_feat[c] += f.sum(dim=0).to(sum_feat.dtype).to(sum_feat.device)
                sumsq_feat[c] += (f * f).sum(dim=0).to(sumsq_feat.dtype).to(sumsq_feat.device)

        return {
            "counts": counts.detach().cpu(),
            "sum": sum_feat.detach().cpu(),
            "sumsq": sumsq_feat.detach().cpu(),
            "feature_dim": torch.tensor(feat_dim, dtype=torch.long),
        }

    def solve_inner_ccvr(self, num_epochs=1, batch_size=10):
        """
        Run normal local training (FedAvg) then compute feature stats for CCVR.
        Returns: (soln, stats, ccvr_stats)
        """
        soln, stats = super().solve_inner(num_epochs=num_epochs, batch_size=batch_size)
        ccvr_stats = self.compute_feature_statistics()
        return soln, stats, ccvr_stats
