"""@sujit880."""

import torch
import numpy as np
import argparse
import importlib
import random
from rich.console import Console
from flearn.utils.model_utils import read_data
from flearn.config.config_main import TRAINERS, PARSER_ARGS
import os
from argparse import Namespace
from generate_clients_dataset import generate

def get_dataset_folder(options):
    """
    Returns the expected dataset folder path based on options.
    """
    # Handle beta in folder name if present and nonzero
    data_settings_str = f"{options['dataset_type']}"
    if "beta" in options and options["beta"] not in [None, 0.0]:
        data_settings_str += f"_b{str(options['beta']).replace('.', '_')}"
    # Handle feature noise in folder name
    noise_str = "" if options.get("feature_noise", 0.0) == 0.0 else "_n"
    data_settings_str += noise_str
    # Handle domain in folder name
    domain_str = "" if options.get("domain") is None else f"_d{options['domain']}"
    if domain_str:
        data_settings_str += domain_str
    # Handle num_clients in folder name if present
    if "num_clients" in options and options["num_clients"] not in [None, 0]:
        data_settings_str += f"_k{options['num_clients']}"
    options["data_settings_name"] = data_settings_str  # Store for future reference
    # Handle n_class in folder name if present
    if "n_class" in options and options["n_class"] not in [None, 0]:
        data_settings_str += f"/c_{options['n_class']}"
    # Compose folder path
    folder = f"{options['dataset']}/{data_settings_str}"
    # If DATASET_DIR is set in config_paths, prepend it
    try:
        from flearn.config.config_paths import DATASET_DIR
        folder = os.path.join(DATASET_DIR, folder)
    except ImportError:
        pass
    return folder

def check_required_generate_args(options):
    """
    Checks if all required arguments for dataset generation are present in options.
    """
    required_args = [
        "dataset",
        "dataset_type",
        "num_clients",
        "n_class",
        "start_idx",
        "feature_noise",
        "beta",
        "domain",
        "train_pkl",
        "test_pkl",
    ]
    missing = [arg for arg in required_args if arg not in options]
    if missing:
        print(f"Missing required options for dataset generation: {missing}")
        return False
    return True

def dataset_ready(options):
    """
    Checks if the dataset folder and required pickle files exist.
    """
    if "data_settings_name" in options and options["data_settings_name"] is not None:
        data_settings_str = options["data_settings_name"]
        if "n_class" in options and options["n_class"] not in [None, 0]:
            data_settings_str += f"/c_{options['n_class']}"
        folder = f"{options['dataset']}/{data_settings_str}"
        try:
            from flearn.config.config_paths import DATASET_DIR
            folder = os.path.join(DATASET_DIR, folder)
        except ImportError:
            pass
    else: folder = get_dataset_folder(options)
    # Check for client pickle files and train/test pickle files
    client_pkl = os.path.join(folder, "0.pkl")
    train_pkl = os.path.join(folder, "train.pkl")
    test_pkl = os.path.join(folder, "test.pkl")
    return os.path.isdir(folder) and os.path.isfile(client_pkl) and os.path.isfile(train_pkl) and os.path.isfile(test_pkl)

def ensure_dataset(options):
    """
    Checks if dataset is ready, otherwise generates it.
    """
    if not dataset_ready(options):
        print("Dataset not found. Generating...")
        params = Namespace(
            dataset=options["dataset"],
            type=options["dataset_type"],
            clients=options["num_clients"],
            start_idx=options.get("start_idx", 0),
            classes=options["n_class"],
            feature_noise=options.get("feature_noise", 0.0),
            beta=options.get("beta", 0.0),
            domain=options.get("domain", "clipart"),
            train_pkl=options.get("train_pkl", True),
            test_pkl=options.get("test_pkl", True),
        )
        generate(params)
    else:
        print("Dataset is ready.")

# In your main function, before reading data:
def main():
    options, trainer = read_options()
    ensure_dataset(options)
    print(options)
    t = trainer(options)
    t.train()

def read_options():
    parser = argparse.ArgumentParser()

    for arg in PARSER_ARGS:
        parser.add_argument(*arg["args"], **arg["kwargs"])

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed["seed"])
    np.random.seed(12 + parsed["seed"])
    torch.manual_seed(123 + parsed["seed"])

    # load selected trainer
    opt_path = f'flearn.trainers.{parsed["trainer"]}'
    mod = importlib.import_module(opt_path)
    trainer = getattr(mod, TRAINERS[parsed["trainer"]]["server"])

    # print and return
    '''maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = "\t%" + str(maxLen) + "s : %s"
    print("Arguments:")
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)#'''

    return parsed, trainer


def main():
    # parse command line arguments
    options, trainer = read_options()
    if not check_required_generate_args(options):
        raise ValueError("Some required options for dataset generation are missing.")
    ensure_dataset(options)
    # read data
    options["dataset_type"] = options["data_settings_name"] # Use the generated dataset settings name
    print(options)

    # call appropriate trainer
    t = trainer(options)
    t.train()


if __name__ == "__main__":
    main()
