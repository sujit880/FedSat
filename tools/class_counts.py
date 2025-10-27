#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

# Ensure project root is on path when running as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from flearn.data.data_utils import get_global_class_counts
from flearn.utils.constants import CLASSES


def human_int(n: int) -> str:
    return f"{n:,}"


def main(
    dataset: str,
    dataset_type: str,
    n_class: Optional[int],
    limit_clients: Optional[int],
):
    K = CLASSES[dataset]
    res = get_global_class_counts(dataset, dataset_type, n_class, limit_clients)
    train_counts: torch.Tensor = res["train_counts"]
    test_counts: torch.Tensor = res["test_counts"]
    total_train = res["total_train"]
    total_test = res["total_test"]
    num_clients = res["num_clients"]
    pickles_dir = res["pickles_dir"]

    print(f"Pickles dir: {pickles_dir}")
    print(f"Processed {num_clients} clients")
    print(f"Train samples total: {human_int(total_train)}")
    print(f"Test samples total:  {human_int(total_test)}")
    print()

    # Per-class table
    header = f"{'Class':>5} | {'Train':>12} | {'Test':>12} | {'Train%':>7} | {'Test%':>7}"
    print(header)
    print("-" * len(header))
    for c in range(K):
        tr = int(train_counts[c].item())
        te = int(test_counts[c].item())
        trp = (100.0 * tr / total_train) if total_train > 0 else 0.0
        tep = (100.0 * te / total_test) if total_test > 0 else 0.0
        print(f"{c:5d} | {human_int(tr):>12} | {human_int(te):>12} | {trp:6.2f}% | {tep:6.2f}%")

    # Biggest and smallest classes by train count
    print()
    topk = min(10, K)
    vals, idxs = torch.topk(train_counts, k=topk)
    print(f"Top-{topk} classes by train count:")
    for rank, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        pct = (100.0 * v / total_train) if total_train > 0 else 0.0
        print(f"  {rank:2d}. class {i:2d}: {human_int(int(v))} ({pct:.2f}%)")

    vals_s, idxs_s = torch.topk(-train_counts, k=topk)
    print(f"Bottom-{topk} classes by train count:")
    for rank, (nv, i) in enumerate(zip(vals_s.tolist(), idxs_s.tolist()), start=1):
        v = -int(nv)
        pct = (100.0 * v / total_train) if total_train > 0 else 0.0
        print(f"  {rank:2d}. class {i:2d}: {human_int(v)} ({pct:.4f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print per-class sample counts across clients and test set")
    p.add_argument("dataset", type=str, help="dataset name, e.g. femnist, emnist, cifar, cifar100")
    p.add_argument("dataset_type", type=str, help="partition folder, e.g. noiid_lbldir_b0_3_k100")
    p.add_argument("--n_class", type=int, default=None, help="optional subfolder for class-specific setups")
    p.add_argument("--limit_clients", type=int, default=None, help="only process first N client pickles for a quick check")

    args = p.parse_args()

    main(args.dataset, args.dataset_type, args.n_class, args.limit_clients)
