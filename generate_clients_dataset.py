import os
import time
import dataset_prepare
from argparse import ArgumentParser, Namespace

# CLASSES = {"mnist": 10, "cifar10": 10, "cifar": 10, "cifar100": 100, "domainnet": 344}
CLASSES = {
    "mnist": 10,
    "cifar": 10,
    "cifar100": 100,
    "emnist": 47,
    "fmnist": 10,
    "har": 6,              # Common HAR datasets (e.g., UCI HAR) have 6 activity classes
    "areview": 2,     # Usually binary sentiment classification (positive/negative)
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


def generate(params: Namespace):
    time_start = time.time()
    print(params.type)
    i = params.classes
    if params.type == "iid":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "iid",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)
    elif params.type == "niid":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "niid",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)
    elif params.type == "dniid":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "dniid",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)
        
    elif params.type == "noiid_lbldir":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "noiid_lbldir",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "beta": params.beta,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)

    elif params.type == "qty_lbl_imb":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "qty_lbl_imb",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "beta": params.beta,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)
    elif params.type == "iid_diff_qty":
        args = {
            "dataset": params.dataset,
            "domain": params.domain,
            "type": "iid_diff_qty",
            "client_num_in_total": params.clients,
            "start_idx": params.start_idx,
            "classes": i,
            "total_classes": CLASSES[params.dataset],
            "feature_noise": params.feature_noise,
            "seed": 0,
            "beta": params.beta,
            "train_pkl": params.train_pkl,
            "test_pkl": params.test_pkl,
        }
        args = Namespace(**args)
        dataset_prepare.preprocess(args)

    else:
        for d_type in ["iid", "dniid"]:
            # print(i,d_type)
            args = {
                "dataset": params.dataset,
                "domain": params.domain,
                "type": d_type,
                "client_num_in_total": params.clients,
                "start_idx": params.start_idx,
                "classes": i,
                "total_classes": CLASSES[params.dataset],
                "feature_noise": params.feature_noise,
                "seed": 0,
                "train_pkl": params.train_pkl,
                "test_pkl": params.test_pkl,
            }
            args = Namespace(**args)
            dataset_prepare.preprocess(args)
    
    time_end = time.time()
    print(f"Time cost: {time_end - time_start} seconds")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar", "cifar100", "domainnet", "fmnist", "har", "areview"],
        default="mnist",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "iid",
            "niid",
            "dniid",
            "synthetic",
            "mix",
            "qty_lbl_imb",  # non-iid with quantity based label imbalance
            "noiid_lbldir",  # non-iid with dirichilet based label imbalance
            "iid_diff_qty",  # quantity skew
        ],
        default="mix",
    )
    parser.add_argument("--clients", type=int, default=200)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--classes",
        type=int,
        default=2,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0.0,
        help="Noise factor to add noise to the features.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Imbalance factor to control the imbalance level.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        default="clipart",
        help="domain of domainnet dataset",
    )
    parser.add_argument(
        "--train_pkl",
        type=bool,
        default=True,
        help="Whether to generate a seperate train.pkl file with entire training data.",
    )
    parser.add_argument(
        "--test_pkl",
        type=bool,
        default=True,
        help="Whether to generate a seperate train.pkl file with entire testing data.",
    )

    args = parser.parse_args()
    generate(args)
