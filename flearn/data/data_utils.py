import pickle
import os
from torch.utils.data import random_split, DataLoader, Dataset
from flearn.data.dataset import (
    MNISTDataset,
    CIFARDataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    FEMNISTDataset,
)
from flearn.config.config_paths import DATASET_DIR
from flearn.utils.constants import CLASSES
from path import Path
import numpy as np
import glob
import torch
import random
import re

# from dotenv import load_dotenv
import re

# load_dotenv()

DATASET_DICT = {
    "mnist": MNISTDataset,
    "cifar": CIFARDataset,
    "cifar10": CIFARDataset,
    "cifar100": CIFAR100Dataset,
    "emnist": EMNISTDataset,
    "fashionmnist": FashionMNISTDataset,
    "femnist": FEMNISTDataset,
}
DATASET_TYPES = [
    "iid",
    "niid",
    "dniid",
    "synthetic",
    "mix",
    "qty_lbl_imb",  # non-iid with quantity based label imbalance
    "noiid_lbldir",  # non-iid with dirichilet based label imbalance
    "iid_diff_qty",  # quantity skew
    "iid_diff_qty_n",
    "noiid_lbldir_n",
]


def get_participants_stat(dataset, dataset_type, n_class):
    # print(f'\n->->: Getting list of participants')
    if n_class not in [None, 0]:
        # pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}_fc_{n_class}"
        pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, "*.pkl"))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r"train.*\.pkl")
        test_pattern = re.compile(r"test.*\.pkl")
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list) - training_files - testing_files
        # print(f'\nDataset Dir: {pickles_dir}, Files: {file_list}')
        users = [(u, dataset_type, n_class) for u in range(total_clients)]
        return users
    else:
        pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
        # print(f'Pickle dir: {pickles_dir}')
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, "*.pkl"))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r"train.*\.pkl")
        test_pattern = re.compile(r"test.*\.pkl")
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list) - training_files - testing_files
        users = [(u, dataset_type, n_class) for u in range(total_clients)]
        return users


def custom_collate_fn(batch, lo, hi):
    # Batch is a list of (image, label) tuples
    images, labels = zip(*batch)

    images = [image * (hi - lo) + lo for image in images]

    # Convert back to tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels


def get_dataloader(
    dataset: str,
    client_id: int,
    data_type: str,
    n_class: int,
    batch_size=20,
    valset_ratio=0.1,
    num_workers: int = 0,
    img_float_val_range: tuple[int | float, int | float] = (0, 1),
):
    # if n_class not in [None, 0]:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}/c_{n_class}"
    # else:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}"
    pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(f"{pickles_dir}/{client_id}.pkl", "rb") as f:
        client_dataset: Dataset = pickle.load(f)

    val_num_samples = int(valset_ratio * len(client_dataset))
    train_num_samples = len(client_dataset) - val_num_samples

    lo, hi = img_float_val_range
    collate_fn = lambda batch: custom_collate_fn(batch, lo, hi)

    trainset, valset = random_split(
        client_dataset, [train_num_samples, val_num_samples]
    )

    trainloader = DataLoader(
        trainset,
        batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    valloader = DataLoader(
        valset,
        batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return trainloader, valloader, train_num_samples, val_num_samples


def get_testloader(
    dataset: str, data_type: str, n_class: int, batch_size=20, valset_ratio=0.1
):
    # if n_class not in [None, 0]:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}_fc_{n_class}"
    # else:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}"
    pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(f"{pickles_dir}/test.pkl", "rb") as f:
        test_dataset: Dataset = pickle.load(f)
    testloader = DataLoader(test_dataset, batch_size, drop_last=True)

    return testloader, len(test_dataset)


def get_client_id_indices(dataset):
    # print(f"Dataset Dir: {DATASET_DIR}")
    dataset_pickles_path = DATASET_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])


def get_dataset_stats(dataset, dataset_type: str, n_class: int, client_id: int):
    # calculating datasets stat
    # if n_class not in [None, 0]:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}_fc_{n_class}"
    # else:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
    pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
    dataset_stats = torch.zeros(CLASSES[dataset])
    with open(f"{pickles_dir}/{client_id}.pkl", "rb") as f:
        client_dataset: Dataset = pickle.load(f)
        for x in client_dataset.targets:
            dataset_stats[x.item()] += 1
        dataset_stats[dataset_stats == 0.0] = 1e-8
    return dataset_stats


def get_global_class_counts(
    dataset: str,
    dataset_type: str,
    n_class: int | None,
    limit_clients: int | None = None,
):
    """
    Aggregate per-class sample counts across all client pickles for the given dataset split.

    Args:
        dataset: dataset name (e.g., 'femnist', 'emnist', 'cifar', 'cifar100').
        dataset_type: subfolder name for the partitioning (e.g., 'noiid_lbldir_b0_3_k100').
        n_class: optional extra subfolder for per-class configs; pass None or 0 if unused.
        limit_clients: if provided, only process the first N client files (for quick checks).

    Returns:
        A dict with keys:
            - 'train_counts': Tensor[K]
            - 'test_counts': Tensor[K] (zeros if test.pkl missing)
            - 'total_train': int
            - 'total_test': int
            - 'num_clients': int (processed clients)
            - 'pickles_dir': str
    """
    # Build pickles directory
    # if n_class not in [None, 0]:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}_fc_{n_class}"
    # else:
    #     pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
    pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError(f"Pickles directory not found: {pickles_dir}. Please preprocess and create pickles first.")

    K = CLASSES[dataset]
    train_counts = torch.zeros(K, dtype=torch.long)
    test_counts = torch.zeros(K, dtype=torch.long)

    # Identify client pickle files: numeric filenames like '0.pkl', '1.pkl', ...
    file_list = sorted(glob.glob(os.path.join(pickles_dir, "*.pkl")))
    digit_re = re.compile(r".*/(\d+)\.pkl$")

    client_files = [f for f in file_list if digit_re.match(f)]
    # Optionally limit clients (useful for a quick sanity check)
    if limit_clients is not None and limit_clients > 0:
        client_files = client_files[:limit_clients]

    # Aggregate train (client) counts
    for fpath in client_files:
        with open(fpath, "rb") as f:
            client_dataset: Dataset = pickle.load(f)
        targets = getattr(client_dataset, "targets", None)
        if targets is None:
            raise RuntimeError(f"Client dataset in {fpath} has no 'targets' attribute")
        targets_t = torch.as_tensor(targets, dtype=torch.long)
        train_counts += torch.bincount(targets_t, minlength=K)[:K]

    # Test counts (if test.pkl exists)
    test_pkl = os.path.join(pickles_dir, "test.pkl")
    total_test = 0
    if os.path.exists(test_pkl):
        with open(test_pkl, "rb") as f:
            test_dataset: Dataset = pickle.load(f)
        ttargets = torch.as_tensor(getattr(test_dataset, "targets", []), dtype=torch.long)
        if ttargets.numel() > 0:
            test_counts = torch.bincount(ttargets, minlength=K)[:K]
            total_test = int(ttargets.numel())

    result = {
        "train_counts": train_counts,
        "test_counts": test_counts,
        "total_train": int(train_counts.sum().item()),
        "total_test": total_test,
        "num_clients": len(client_files),
        "pickles_dir": pickles_dir,
    }

    return result
