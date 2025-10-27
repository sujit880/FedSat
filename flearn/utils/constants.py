from torch import optim

GLR = {
    None: {"mnist": 0.009, "cifar": 0.001, "cifar100": 0.001},
    "lenet5": {"mnist": 0.009, "cifar": 0.001, "cifar100": 0.001},
    "resnet8": {"mnist": 0.09, "cifar": 0.01, "cifar100": 0.01},
    "resnet18": {"mnist": 0.9, "cifar": 0.1, "cifar100": 0.1},
    "tresnet18": {"mnist": 0.9, "cifar": 0.1, "cifar100": 0.1},
    "tresnet20": {"mnist": 0.9, "cifar": 0.1, "cifar100": 0.1},
}

INPUT_CHANNELS = {
    "mnist": 1,
    "medmnistS": 1,
    "medmnistC": 1,
    "medmnistA": 1,
    "covid19": 3,
    "fmnist": 1,
    "emnist": 1,
    "femnist": 1,
    "cifar": 3,
    "cifar10": 3,
    "cinic10": 3,
    "svhn": 3,
    "cifar100": 3,
    "celeba": 3,
    "usps": 1,
    "tinyimagenet": 3,
    "domain": 3,
}

# (C, H, W)
DATA_SHAPE: dict[str, tuple[int, int, int]] = {
    "mnist": (1, 28, 28),
    "medmnistS": (1, 28, 28),
    "medmnistC": (1, 28, 28),
    "medmnistA": (1, 28, 28),
    "fmnist": (1, 28, 28),
    "svhn": (3, 32, 32),
    "emnist": (1, 28, 28),
    "femnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar": (3, 32, 32),
    "cinic10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "covid19": (3, 244, 224),
    "usps": (1, 16, 16),
    "celeba": (3, 218, 178),
    "tinyimagenet": (3, 64, 64),
}

CLASSES: dict[str, int] = {
    "mnist": 10,
    "medmnistS": 11,
    "medmnistC": 11,
    "medmnistA": 11,
    "fmnist": 10,
    "svhn": 10,
    # EMNIST uses the 'balanced' split (47 classes) in this codebase
    "emnist": 47,
    "femnist": 62,
    "cifar": 10,
    "cifar10": 10,
    "cinic10": 10,
    "cifar100": 100,
    "covid19": 4,
    "usps": 10,
    "celeba": 2,
    "tinyimagenet": 200,
}


DATA_MEAN = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    # Use consistent normalization for EMNIST/FEMNIST
    "emnist": [0.1751],
    "fmnist": [0.286],
    "femnist": [0.1751],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "tinyimagenet": [122.5119, 114.2915, 101.388],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}


DATA_STD = {
    "mnist": [0.3015],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3333],
    "fmnist": [0.3205],
    "femnist": [0.3333],
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    "covid19": [56.6888, 56.6933, 56.6979],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "tinyimagenet": [58.7048, 57.7551, 57.6717],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}


LR_SCHEDULERS = {
    "step": optim.lr_scheduler.StepLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "constant": optim.lr_scheduler.ConstantLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}

LAMDA = {
    "mnist": 1,
    "cifar": 0.1,
    "cifar10": 0.1,
    "cifar100": 0.1,
    "fmnist":0.1,
    "tinyimagenet": 0.1,
}

FLAMDA = {
    "mnist": 1.0,
    "cifar": 0.5,
    "cifar10": 0.1,
    "cifar100": 0.1,
    "fmnist":0.5,
    "tinyimagenet": 0.1,
}

CVAE_HD = {
    "lenet5": [512,256], 
    "resnet8":[512,256], 
    "resnet18": [1024,512], 
    "tresnet18": [1024,512],
    "tresnet20": [2048,1024],
}

MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.4914, 0.4822, 0.4465),
    "emnist": (0.1751,),
    "femnist": (0.1751,),
    "fmnist": (0.2860,),
    "tinyimagenet": (),
}

STD = {
    "mnist": (0.3015,),
    "cifar": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2023, 0.1994, 0.2010),
    "emnist": (0.3333,),
    "femnist": (0.3333,),
    "fmnist": (0.3530,),
    "tinyimagenet": (),
}

SIZE = {
    "mnist": (28, 28),
    "cifar": (32, 32),
    "cifar100": (32, 32),
    "emnist": (28, 28),
    "femnist": (28, 28),
    "fmnist": (28, 28),
    "domainnet": (64, 64),
    "tinyimagenet": (64, 64),
}

MOON_MU = {
    "mnist":5.0, 
    "cifar":5.0, 
    "cifar100":1.0, 
    "tinyimagenet":1.0, 
    "emnist": 2.0, 
    "femnist": 2.0,
    "fmnist":2.0,
}

FedProxMU = {
    "mnist":0.01, 
    "cifar":0.01, 
    "cifar100":0.001, 
    "tinyimagenet":0.001, 
    "emnist": 0.01, 
    "fashionmnist":0.01,
    "femnist":0.01,
    "fmnist":0.01,
}

FedSatL_MU = {
    "mnist":3.0, 
    "cifar":1.0, 
    "cifar100":1.0, 
    "tinyimagenet":1.0, 
    "emnist": 2.0, 
    "femnist": 2.0,
    "fmnist":2.0,
}

SAMPLE_PER_CLASS = {
    "mnist":20, 
    "cifar":20, 
    "cifar100":2, 
    "tinyimagenet":1, 
    "emnist": 7, 
    "femnist": 7,
    "fmnist":5
}
