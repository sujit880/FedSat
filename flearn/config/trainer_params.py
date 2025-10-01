from typing import Any

PeFLL_ARGS: dict[str, Any] = {
    "embed_dim": -1,
    "embed_y": 1,
    "embed_num_kernels": 16,
    "embed_num_batches": 1,
    "hyper_embed_lr": 2e-4,
    "hyper_hidden_dim": 100,
    "hyper_num_hidden_layers": 3,
    "clip_norm": 50.0,
}

MOON_ARGS: dict[str, Any] = {
}

SCAFFOLD_ARGS: dict[str, Any] = {
}

BLO_ARGS: dict[str, Any] = {
}

PROTO_ARGS: dict[str, Any] = {
}

FLAIR_ARGS: dict[str, Any] = {
}

FLUTE_ARGS: dict[str, Any] = {
    "loss": "CE",
    "rep_round": 1,
    "lambda1": 0.25,
    "lambda2": 0.0025,
    "lambda3": 0.0005,
    "gamma1": 1,
    "gamma2": 1,
    "nc2_lr": 0.5,
    "nc_lr": 0.5,
    "finetune_epochs": 1,
}

FLOCO_ARGS: dict[str, Any] = {
    "method": "floco",
    "endpoints": 20,
    "tau": 50,
    "rho": 0.1,
}

DistHN_ARGS: dict[str, Any] = {
    "hyper_lr": 2e-4,
    "hyper_hidden_dim": 100,
    "hyper_num_hidden_layers": 3,
    "clip_norm": 50.0,
}

PPVAE_ARGS: dict[str, Any] = {
    "VAE_block_depth": 32,
    "VAE_lr": 1e-3,
    "VAE_re": 5.0,
    "VAE_kl": 0.005,
    "VAE_batch_size": 64,
    "VAE_noise_mean": 0,
    "VAE_noise_std1": 0.015,
    "VAE_noise_std2": 0.025,
    "VAE_ce": 2.0,
    "VAE_x_ce": 0.4,
    "VAE_alpha": 2.0,
    "distilling": True,
}

CVAE_ARGS: dict[str, Any] = {
    "VAE_block_depth": 128,
    "VAE_lr": 1e-3,
    "VAE_noise_mean": 0,
    "VAE_noise_std1": 0.015,
    "VAE_noise_std2": 0.025,
}

DCGAN_ARGS: dict[str, Any] = {
    "GAN_block_depth": 256,
    "GAN_lr": 0.0002,
    "GAN_beta1": 0.5,
}

DDPM_ARGS: dict[str, Any] = {
    "block_depth": 256,
    "time_dim": 128,
    "learning_rate": 1e-4,
    "noise_schedule_type": "cosine",
    "NOISE_SCHEDULER_ARGS": {
        "noise_steps": 4000,
    },
}

DDPM_DISTIL_ARGS: dict[str, Any] = {
    "block_depth": 256,
    "time_dim": 128,
    "learning_rate": 1e-4,
    "noise_schedule_type": "cosine",
    "NOISE_SCHEDULER_ARGS": {
        "noise_steps": 4000,
    },
    "num_distil_epochs": 50,
    "distil_timesteps": 100,
    "distil_loss": "kl",
}

DDPM_MOE_ARGS: dict[str, Any] = {
    "block_depth": 64,
    "num_unets": 12,
    "time_dim": 128,
    "learning_rate": 1e-4,
    "noise_schedule_type": "cosine",
    "NOISE_SCHEDULER_ARGS": {
        "noise_steps": 4000,
    },
}

"""
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
"""


FED_ONE_SHOT_ARGS: dict[str, Any] = {}

FED_CVAE_KD_ARGS: dict[str, Any] = {
    "mnist": {
        # sample_ratio=0.5
        # local_epochs=15
        # local_LR=0.001
        "z_dim": 10,
        "beta": 1.0,
        "model_version": 0,
        "classifier_num_train_samples": 5000,
        "classifier_epochs": 10,
        "decoder_LR": 0.01,
        "decoder_num_train_samples": 5000,
        "decoder_epochs": 7,
        "uniform_range": (-1, 1),
        "should_weight": True,
        "should_initialize_same": True,
        "should_avg": False,
        "should_fine_tune": True,
        "heterogeneous_models": False,
        "should_transform": False,
    },
    "cifar": {
        # sample_ratio=1.0
        # local_epochs=50
        # local_LR=0.001
        "z_dim": 15,
        "beta": 1.0,
        "model_version": 2,
        "classifier_num_train_samples": 2500,
        "classifier_epochs": 5,
        "decoder_LR": 0.01,
        "decoder_num_train_samples": 10000,
        "decoder_epochs": 30,
        "uniform_range": (-1, 1),
        "should_weight": True,
        "should_initialize_same": True,
        "should_avg": False,
        "should_fine_tune": True,
        "heterogeneous_models": False,
        "should_transform": False,
    },
}

FED_CVAE_ENS_ARGS: dict[str, Any] = {
    "mnist": {
        # sample_ratio=0.5
        # local_epochs=15
        # local_LR=0.001
        "z_dim": 10,
        "beta": 1.0,
        "model_version": 0,
        "classifier_num_train_samples": 5000,
        "classifier_epochs": 10,
        "uniform_range": (-3, 3),
        "should_initialize_same": True,
        "should_weight": True,
    },
    "cifar": {
        # sample_ratio=1.0
        # local_epochs=100
        # local_LR=0.001
        "z_dim": 25,
        "beta": 1.0,
        "model_version": 2,
        "classifier_num_train_samples": 7500,
        "classifier_epochs": 5,
        "uniform_range": (-3, 3),
        "should_initialize_same": True,
        "should_weight": True,
    },
}

# num_epochs=10
# batch_size=128
# learning_rate=0.01
FED_SD2C_ARGS: dict[str, Any] = {
    "mnist": {},
    "cifar": {
        # client
        "model_name": "ResNet18",
        "client_instance": "coreset+dist_syn",
        # server
        "optim_name": "Adam",
        "server_lr_scheduler": "cos",
        "num_distill_epochs": 100,
        "momentum": 0.9,
        "temperature": 1,
        # Coreset
        "ipc": 50,  # 50
        "mipc": 500,  # 500
        "num_crop": 10,
        # Synthesis
        "iterations_per_layer": 100,
        "jitter": 0,
        "sre2l_lr": 0.1,
        "r_bn": 0,
        "r_c": 0.01,
        "iter_mode": "ipc",
        "first_bn_multiplier": 10.0,
        "inputs_init": "vae+fourier",
        "noise_type": "None",
        "noise_s": 0,
        "noise_p": 0,
        "fourier_lambda": 0.8,
        "sd2c_loss": "mse",
    },
}
