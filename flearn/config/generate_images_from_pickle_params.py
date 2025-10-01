from flearn.models.CVAE import CVAE
from flearn.models.DCGAN import DCGAN
from flearn.models.DDPM import DDPM
from flearn.models.CVAE_new import CVAE as CVAE_new

DATASET = "cifar"

MODEL_DATA = {
    # "CVAE": (
    #     "CVAE",
    #     CVAE,
    #     {
    #         "block_depth": 32,
    #         "noise_type": "gaussian",
    #         "feature_dim": 128,
    #     },
    # ),
    # "DCGAN": (
    #     "DCGAN",
    #     DCGAN,
    #     {
    #         "block_depth": 256,
    #         "feature_dim": 128,
    #         "transformer_model": True,
    #     },
    # ),
    # "DDPM_Encoded": (
    #     "DDPM_Encoded",
    #     DDPM,
    #     {
    #         "block_depth": 256,
    #         "time_dim": 128,
    #         "one_hot_encoding": False,
    #     },
    # ),
    # "DDPM_One-Hot": (
    #     "DDPM_One_Hot",
    #     DDPM,
    #     {
    #         "block_depth": 256,
    #         "time_dim": 128,
    #         "one_hot_encoding": True,
    #     },
    # ),
    "DDPM_Improved": (
        "DDPM_Improved",
        DDPM,
        {
            "block_depth": 256,
            "time_dim": 128,
            # "one_hot_encoding": True,
        },
    ),
    # "CVAE_Adv_Eq_Loss": (
    #     "CVAE_Adv_Eq_Loss",
    #     CVAE_new,
    #     {
    #         "block_depth": 128,
    #         "noise_type": "gaussian",
    #         "feature_dim": 512,
    #     },
    # ),
    # "CVAE_Adv_Log_Reconst_Loss": (
    #     "CVAE_Adv_Log_Reconst_Loss",
    #     CVAE_new,
    #     {
    #         "block_depth": 128,
    #         "noise_type": "gaussian",
    #         "feature_dim": 512,
    #     },
    # ),
}
