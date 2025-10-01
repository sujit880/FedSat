# GLOBAL PARAMETERS
# OPTIMIZERS = [
#     "elastic",
#     "fedavg",
#     "fedavgm",
#     "feddane",
#     "feddyn",
#     "fedlada",
#     "fedlc",
#     "fedproposedAdam",
#     "fedproposed",
#     "proposedCSS",
#     "fedproposedFedprox",
#     "proposedScaffold",
#     "fedproposedSGD",
#     "fedproposedvImp",
#     "fedprox",
#     "fedproposedScaffold",
#     "proposedC",
#     "fedSat",
#     "scaffold",
#     "proposedFedDyn",
#     "proposedFedAvgM",
#     "proposedFedavg",
#     "proposedFedprox",
#     "scaffold_new",
#     "proposedCS",
#     "pefll",
#     "flute",
# ]

TRAINERS = {
    "local": {"server": "LocalServer", "client": "LocalClient"},
    "fedavg": {"server": "FedAvgServer", "client": "FedAvgClient"},
    "elastic": {"server": "FedAvgServer", "client": "FedAvgClient"},
    "fedavgg": {"server": "FedAvgServer", "client": "FedAvgClient"},
    "fedavggs": {"server": "FedAvgServer", "client": "FedAvgClient"},
    "fedprox": {"server": "FedAvgServer", "client": "FedAvgClient"},
    "moon": {"server": "FedMOONServer", "client": "MOONClient"},
    "scaffold": {"server": "SCAFFOLDServer", "client": "SCAFFOLDClient"},
    "fedpvr": {"server": "SCAFFOLDServer", "client": "SCAFFOLDClient"},
    "fedmappvr": {"server": "SCAFFOLDServer", "client": "SCAFFOLDClient"},
    "fedks": {"server": "FedKSeedServer", "client": "FedKSeedClient"},
    "fedsatl": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedsat": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedmap": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedblo": {"server": "FedBLOServer", "client": "FedBLOClient"},
    "fedmapd": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedmapg": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedmapgs": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedmapga": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedmapddpg": {"server": "FedSatLServer", "client": "FedSatLClient"},
    "fedproto": {"server": "ProtoServer", "client": "ProtoClient"},
    "fedprotohn": {"server": "FedPHNServer", "client": "FedPHNClient"},
    "fedppgen": {"server": "FedEPGenServer", "client": "FedEPGenClient"},
    "eproto": {"server": "EprotoServer", "client": "EprotoClient"},
    "fedeprotogen": {"server": "FedEPGenServer", "client": "FedEPGenClient"},
    "fedcvae": {"server": "FedFLAIRServer", "client": "FedFLAIRClient"},
    "fedprotocvae": {"server": "FedFLAIRServer", "client": "FedFLAIRClient"},
    "fedflair": {"server": "FedFLAIRServer", "client": "FedFLAIRClient"},
    "pfedflair": {"server": "FedFLAIRServer", "client": "FedFLAIRClient"},
    "fedtest": {"server": "FedTestServer", "client": "FedTestClient"},
    "fedcgn": {"server": "FedTestServer", "client": "FedTestClient"},
    "pefll": {"server": "PeFLLServer", "client": "PeFLLClient"},
    "flute": {"server": "FLUTEServer", "client": "FLUTEClient"},
    "disthn": {"server": "DistHNServer", "client": "DistHNClient"},
    "ppvae": {"server": "PPVAEServer", "client": "PPVAEClient"},
    "cvae": {"server": "CVAEServer", "client": "CVAEClient"},
    "dcgan": {"server": "DCGANServer", "client": "DCGANClient"},
    "ddpm": {"server": "DDPMServer", "client": "DDPMClient"},
    "ppvae_new": {"server": "PPVAEServer", "client": "PPVAEClient"},
    "cvae_new": {"server": "CVAEServer", "client": "CVAEClient"},
    "ddpm_new": {"server": "DDPMServer", "client": "DDPMClient"},
    "ddpm_moe": {"server": "DDPMServer", "client": "DDPMClient"},
    "ddpm_distil": {"server": "DDPMServer", "client": "DDPMClient"},
    "fed_one_shot": {"server": "OneShotServer", "client": "OneShotClient"},
    "fed_cvae_kd": {"server": "CVAE_KD_Server", "client": "CVAE_KD_Client"},
    "fed_cvae_ens": {"server": "CVAE_ENS_Server", "client": "CVAE_KD_Client"},
    "fed_sd2c": {"server": "FedSD2CServer", "client": "FedSD2CClient"},
    "floco": {"server": "FlocoServer", "client": "FlocoClient"},
    "ditto": {"server": "FedDittoServer", "client": "DittoClient"},
}

DATASETS = ["cifar", "cifar10", "cifar100", "mnist", "fmnist", "areview"]

DATASETS_TYPES = ["iid", "niid", "mix", "dniid", "synthetic", "syntheticM"]

PARSER_ARGS = [
    {
        "args": ["--trainer"],
        "kwargs": {
            "help": "name of trainer",
            "type": str,
            "choices": list(TRAINERS.keys()),
            "default": "fedavg",
        },
    },
    {
        "args": ["--dataset"],
        "kwargs": {
            "help": "name of dataset",
            "type": str,
            "choices": DATASETS,
            "default": "mnist",
        },
    },
    {
        "args": ["--dataset_type"],
        "kwargs": {
            "help": "type of dataset",
            "type": str,
            # "choices": DATASETS_TYPES,
            "default": "iid",
        },
    },
    {
        "args": ["--beta"],
        "kwargs": {
            "help": "Imbalance factor to control the imbalance level",
            "type": float,
            "default": None,
        },
    },
    {
        "args": ["--start_idx"],
        "kwargs": {
            "help": "Starting index for client IDs",
            "type": int,
            "default": 0,
        },
    },
    {
        "args": ["--feature_noise"],
        "kwargs": {
            "help": "Feature noise level for synthetic data",
            "type": float,
            "default": 0.0,
        },
    },
    {
        "args": ["--domain"],
        "kwargs": {
            "help": "Domain name for domain adaptation datasets",
            "type": str,
            "default": None,
            # "choices": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        },
    },
    {
        "args": ["--train_pkl"],
        "kwargs": {
            "help": "Whether to generate train.pkl file",
            "type": bool,
            "choices": [True, False],
            "default": True,
        },
    },
    {
        "args": ["--test_pkl"],
        "kwargs": {
            "help": "Whether to generate test.pkl file",
            "type": bool,
            "choices": [True, False],
            "default": True,
        },
    },
    {
        "args": ["--n_class"],
        "kwargs": {
            "help": "maximum number of class presents in each pickle",
            "type": int,
            "default": None,
        },
    },
    {
        "args": ["--num_clients"],
        "kwargs": {
            "type": int,
            "default": 50,
        },
    },
    {
        "args": ["--model"],
        "kwargs": {
            "help": "name of model",
            "type": str,
            "default": None,
        },
    },
    {
        "args": ["--num_rounds"],
        "kwargs": {
            "help": "number of rounds to simulate",
            "type": int,
            "default": 10,
        },
    },
    {
        "args": ["--eval_every"],
        "kwargs": {
            "help": "evaluate every ____ rounds",
            "type": int,
            "default": -1,
        },
    },
    {
        "args": ["--clients_per_round"],
        "kwargs": {
            "help": "number of clients trained per round",
            "type": int,
            "default": 15,
        },
    },
    {
        "args": ["--batch_size"],
        "kwargs": {
            "help": "batch size when clients train on data",
            "type": int,
            "default": 16,
        },
    },
    {
        "args": ["--num_epochs"],
        "kwargs": {
            "help": "number of epochs when clients train on data",
            "type": int,
            "default": 1,
        },
    },
    {
        "args": ["--num_iters"],
        "kwargs": {
            "help": "number of iterations when clients train on data",
            "type": int,
            "default": 1,
        },
    },
    {
        "args": ["--learning_rate"],
        "kwargs": {
            "help": "learning rate for inner solver",
            "type": float,
            "default": 0.001,
        },
    },
    {
        "args": ["--mu"],
        "kwargs": {
            "help": "constant for prox",
            "type": float,
            "default": 0.01,
        },
    },
    {
        "args": ["--seed"],
        "kwargs": {
            "help": "seed for randomness",
            "type": int,
            "default": 0,
        },
    },
    {
        "args": ["--drop_percent"],
        "kwargs": {
            "help": "percentage of slow devices",
            "type": float,
            "default": 0.1,
        },
    },
    {
        "args": ["--valset_ratio"],
        "kwargs": {
            "help": "Proportion of val set in the entire client local dataset",
            "type": float,
            "default": 0.1,
        },
    },
    {
        "args": ["--gpu"],
        "kwargs": {
            "help": "True value indicates to use gpu",
            "type": bool,
            "default": True,
        },
    },    
    {
        "args": ["--cuda"],
        "kwargs": {
            "help": "+ve integer value indicates to use gpu",
            "type": int,
            "default": -1,
        },
    },
    {
        "args": ["--optm"],
        "kwargs": {
            "type": str,
            "help": "Name of Optmizer",
            "choices": ["SGD", "Adam", "PGD", "PGGD"],
            "default": "SGD",
        },
    },
    {
        "args": ["--loss"],
        "kwargs": {
            "type": str,
            "choices": ["CE", "CL", "MSL", "FL", "LS", "CB", "MSE", "CS", "CSN", "PSL", "PSL1", "CAPA", "MCAPA", "MCA", "DBCC", "DB", "CALB", "CACS"],
            "default": "CE",
        },
    },
    {
        "args": ["--log"],
        "kwargs": {
            "type": int,
            "default": 0,
        },
    },
    {
        "args": ["--all_test"],
        "kwargs": {
            "type": bool,
            "choices": [True, False],
            "default": True,
        },
    },
    {
        "args": ["--weight_decay"],
        "kwargs": {
            "type": float,
            "default": 0.0,
        },
    },
    {
        "args": ["--momentum"],
        "kwargs": {
            "type": float,
            "default": 0.9,
        },
    },
    {
        "args": ["--file_buffer_limit"],
        "kwargs": {
            "help": "Number of rows after which Result csv file Buffers are flushed.",
            "type": int,
            "default": 5,
        },
    },
    {
        "args": ["--num_workers"],
        "kwargs": {
            "help": "Number of workers in dataloader",
            "type": int,
            "default": 0,
        },
    },    
    {
        "args": ["--data_settings_name"],
        "kwargs": {
            "help": "Custom name for dataset settings/folder",
            "type": str,
            "default": None,
        },
    },
]
