# import sys

# sys.path.append("../")
import copy
import os
import pickle
import numpy as np
import random
import torch
import json
from path import Path
from argparse import ArgumentParser, Namespace
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, EMNIST, FashionMNIST
from torchvision import transforms
from flearn.data.dataset import (
    MNISTDataset,
    CIFARDataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    HAR,
    HARDataset,
    AmazonReview,
    AmazonReviewDataset,
    # DomainNet,
    # DomainNetDataset,
)
from collections import Counter
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from flearn.utils.slicing import (
    noniid_slicing,
    random_slicing,
    noniid_dirichlet,
    quantity_based_label_imbalance,
    distribution_based_label_skew,
    quantity_skew,
)
from flearn.config.config_paths import DATASET_DIR

# from dotenv import load_dotenv

# load_dotenv()

CURRENT_DIR = DATASET_DIR
# /mnt/c/Users/S_G/Documents/GitHub/Dataset

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar": (CIFAR10, CIFARDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFAR100Dataset),
    "emnist": (EMNIST, EMNISTDataset),
    "fmnist": (FashionMNIST, FashionMNISTDataset),
    "har": (HAR, HARDataset),
    "areview": (AmazonReview, AmazonReviewDataset),
    # "domainnet": (DomainNet, DomainNetDataset),
}

DATASETS_TYPES = [
    "iid",
    "niid",
    "dniid",
    "synthetic",
    "mix",
    "qty_lbl_imb",  # non-iid with quantity based label imbalance
    "noiid_lbldir",  # non-iid with dirichilet based label imbalance
    "iid_diff_qty",  # quantity skew
]

SLICING = {
    "iid": random_slicing,
    "niid": noniid_slicing,
    "dniid": noniid_dirichlet,
    "fniid": quantity_based_label_imbalance,
    "qty_lbl_imb": quantity_based_label_imbalance,
    "noiid_lbldir": distribution_based_label_skew,
    "iid_diff_qty": quantity_skew,
}


MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.4914, 0.4822, 0.4465),
    "emnist": (0.1751,),
    "fmnist": (0.2860,),
    "domainnet": (0.485, 0.456, 0.406),
    "har": None,
    "areview": None,
}

STD = {
    "mnist": (0.3081,),
    "cifar": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2023, 0.1994, 0.2010),
    "emnist": (0.3333,),
    "fmnist": (0.3530,),
    "domainnet": (0.229, 0.224, 0.225),
    "har": None,
    "areview": None,
}

SIZE = {
    "mnist": (28, 28),
    "cifar": (32, 32),
    "cifar100": (32, 32),
    "emnist": (28, 28),
    "fmnist": (28, 28),
    "domainnet": (64, 64),
    "har": (128, 9),  # example for sensor data shape
    "areview": None,  # text data, no fixed image size
}



class MNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target


classes_per_client = None


def preprocess(args: Namespace) -> None:
    print(args)
    global classes_per_client
    classes_per_client = args.classes
    # Handle beta in folder name if present and nonzero
    data_settings_str = f"{args.type}"
    if hasattr(args, "beta") and args.beta not in [None, 0.0]:
        data_settings_str += f"_b{str(args.beta).replace('.', '_')}"
    # Handle feature noise in folder name
    noise_val = getattr(args, "feature_noise", 0.0)
    noise_str = "" if noise_val == 0.0 else f"_n{str(noise_val).replace('.', '_')}"
    data_settings_str += noise_str
    # Handle domain in folder name
    domain_val = getattr(args, "domain", None)
    domain_str = "" if domain_val is None else f"_d{domain_val}"
    if domain_str:
        data_settings_str += domain_str
    # Handle num_clients in folder name if present
    if hasattr(args, "client_num_in_total") and args.client_num_in_total not in [None, 0]:
        data_settings_str += f"_k{args.client_num_in_total}"
    # Handle n_class in folder name if present
    if hasattr(args, "classes") and args.classes not in [None, 0]:
        data_settings_str += f"/c_{args.classes}"
    # Compose folder path
    dataset_dir = f"{CURRENT_DIR}/{args.dataset}"
    pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{data_settings_str}"
   

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_train_clients = int(args.client_num_in_total)

    if SIZE[args.dataset] is not None:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(SIZE[args.dataset]),
                transforms.ToTensor(),
            ]
        )
    else: transform = None
    target_transform = None
    trainset_stats = {}
    testset_stats = {}

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(pickles_dir):
        # Create the directory structure
        os.makedirs(pickles_dir, exist_ok=True)
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            dataset_dir,
            split="balanced",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            dataset_dir, split="balanced", train=False, transform=transforms.ToTensor()
        )
    elif args.dataset == "domainnet":
        trainset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            domain=args.domain,
            train=True,
            download=True,
            transform=transform,
        )
        testset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            domain=args.domain,
            train=False,
            transform=transform,
        )
    elif args.dataset == "areview":
        trainset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            split="train",
            download=True,
            max_length=128,       # optional, if you want to customize
        )
        testset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            split="test",
            download=True,
            max_length=128,
        )
    else:
        trainset = ori_dataset(
            dataset_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        testset = ori_dataset(dataset_dir, train=False, transform=transforms.ToTensor())

    noise_factor = args.feature_noise

    num_classes = args.total_classes
    distribution = args.type
    beta = args.beta if hasattr(args, "beta") else 0.0
    all_trainsets, trainset_stats = randomly_alloc_classes(
        ori_dataset=trainset,
        target_dataset=target_dataset,
        num_clients=num_train_clients,
        num_classes=num_classes,
        num_classes_per_client=classes_per_client,
        transform=transform,
        target_transform=target_transform,
        distribution=distribution,
        noise_factor=noise_factor,
        beta=beta,
    )

    client_id = args.start_idx
    for dataset in all_trainsets:
        with open(pickles_dir + f"/{client_id}.pkl", "wb") as f:
            pickle.dump(dataset, f)
        client_id += 1
    if args.dataset == "domainnet":
        if args.train_pkl:
            with open(pickles_dir + f"/train_{args.domain}.pkl", "wb") as f:
                pickle.dump(trainset, f)
        if args.test_pkl:
            with open(pickles_dir + f"/test_{args.domain}.pkl", "wb") as f:
                pickle.dump(testset, f)
    else:
        if args.train_pkl:
            with open(pickles_dir + f"/train.pkl", "wb") as f:
                pickle.dump(trainset, f)
        if args.test_pkl:
            with open(pickles_dir + f"/test.pkl", "wb") as f:
                pickle.dump(testset, f)
    with open(dataset_dir + f"/all_stats.json", "w") as f:
        json.dump({"train": trainset_stats, "test": testset_stats}, f)


def randomly_alloc_classes(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    num_classes_per_client: int,
    transform=None,
    target_transform=None,
    distribution: str = "niid",
    noise_factor: float = 0.0,
    beta: float = 0.5,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    print(
        f"num_clients={num_clients}, num_classes={num_classes}, distribution={distribution}"
    )
    if noise_factor > 0.0:
        ori_dataset = add_feature_noise(
            ori_dataset, noise_factor
        )  # this converts the dataset to a list of tuples
    slicing = SLICING[distribution]
    if distribution == "niid":
        dict_users = slicing(ori_dataset, num_clients, num_clients * num_classes)
    if distribution == "dniid":
        dict_users = slicing(ori_dataset, num_clients, num_clients * num_classes)
    if distribution == "iid":
        dict_users = slicing(ori_dataset, num_clients)
    if distribution == "qty_lbl_imb":
        dict_users = slicing(
            ori_dataset, num_clients, num_classes, num_classes_per_client
        )
    if distribution == "noiid_lbldir":
        dict_users = slicing(ori_dataset, num_clients, num_classes, beta)
    if distribution == "iid_diff_qty":
        dict_users = slicing(ori_dataset, num_clients, beta)
    stats = {}
    for i, indices in dict_users.items():
        targets_numpy = np.array(ori_dataset.targets)
        # print(f'indices={indices}, type: {type(indices[0])}')
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    for indices in dict_users.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats


def add_feature_noise(ori_dataset, noise_factor):
    features = []
    labels = []
    for i in range(len(ori_dataset)):
        print(f"Or_noise_add_sample_check: {ori_dataset[0]}")
        noise_pixel = ori_dataset[i][0] + noise_factor * torch.randn(
            ori_dataset[i][0].shape
        )
        noise_pixel = torch.clamp(noise_pixel, 0, 1)
        label = ori_dataset[i][1]
        features.append(noise_pixel)
        labels.append(label)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return MNISTDataset(features, labels)

def moon_partition_data(ori_dataset, target_dataset, transform, target_transform, partition, n_parties, num_classes, beta=0.4):
    N = len(ori_dataset)
    dict_users = {i: np.array([], dtype="int64") for i in range(n_parties)}

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, n_parties)
        for i in range(n_parties):
            np.random.shuffle(batch_idxs[i])
            dict_users[i] = np.array(batch_idxs[i])

    elif partition == "noniid-labeldir" or partition == "dniid":
        min_size = 0
        min_require_size = 10
        K = num_classes

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(np.array(ori_dataset.targets) == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        dict_users = {i: np.array([], dtype="int64") for i in range(n_parties)}
        for i in range(n_parties):
            np.random.shuffle(idx_batch[i])
            dict_users[i] = np.array(idx_batch[i])

    stats = {}
    for i, indices in dict_users.items():
        targets_numpy = np.array(ori_dataset.targets)
        # print(f'indices={indices}, type: {type(indices[0])}')
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    for indices in dict_users.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats