import pickle
import os
from torch.utils.data import random_split, DataLoader, Dataset
from flearn.data.dataset import (
    MNISTDataset,
    CIFARDataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
)
from flearn.config.config_paths import DATASET_DIR
from flearn.utils.constants import CLASSES
from path import Path
import numpy as np
import glob
import torch
import random

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
        pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}/{n_class}"
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
    if n_class not in [None, 0]:
        pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}/{n_class}"
    else:
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
    if n_class not in [None, 0]:
        pickles_dir = f"{DATASET_DIR}/{dataset}/{data_type}/{n_class}"
    else:
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
    pickles_dir = f"{DATASET_DIR}/{dataset}/{dataset_type}/{n_class}"
    dataset_stats = torch.zeros(CLASSES[dataset])
    with open(f"{pickles_dir}/{client_id}.pkl", "rb") as f:
        client_dataset: Dataset = pickle.load(f)
        for x in client_dataset.targets:
            dataset_stats[x.item()] += 1
        dataset_stats[dataset_stats == 0.0] = 1e-8
    return dataset_stats
