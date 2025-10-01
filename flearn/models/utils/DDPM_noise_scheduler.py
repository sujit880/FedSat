import torch
import torch.nn as nn
import math
import numpy as np
import random
from tqdm import tqdm

from flearn.utils.constants import DATA_SHAPE, CLASSES


class NoiseScheduler:
    def __init__(
        self,
        dataset: str,
        noise_steps: int = 1000,
        device: torch.device | None = None,
    ) -> None:

        self.img_size = DATA_SHAPE[dataset][-1]

        self.noise_steps = noise_steps
        self.device = device

        self.betas = self.prepare_noise_schedule().to(device)

        self.init_alphas()

    def init_alphas(self):

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(
            self.alphas, axis=0
        ).requires_grad_(False)

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1.0, self.alphas_cumprod[:-1].cpu().numpy()),
            dtype=torch.float,
            device=self.device,
        )

        self.alphas_cumprod_next: torch.Tensor = torch.tensor(
            np.append(self.alphas_cumprod[1:].cpu().numpy(), 0.0),
            dtype=torch.float,
            device=self.device,
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).requires_grad_(False)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        ).requires_grad_(False)

        self.log_one_minus_alphas_cumprod = torch.log(
            1.0 - self.alphas_cumprod
        ).requires_grad_(False)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod
        ).requires_grad_(False)

        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1
        ).requires_grad_(False)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).requires_grad_(False)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.tensor(
                np.append(
                    self.posterior_variance[1].cpu().numpy(),
                    self.posterior_variance[1:].cpu().numpy(),
                ),
                dtype=torch.float,
                device=self.device,
            )
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).requires_grad_(False)

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        ).requires_grad_(False)

    def prepare_noise_schedule(self) -> torch.Tensor:
        raise NotImplementedError(
            "prepare_noise_scedule function not implemented in Derived Class"
        )

    def noise_images(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        sqrt_alpha_hat = self.alphas_cumprod[t][:, None, None, None]

        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]
        noise = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,)).to(
            self.device
        )

    def q_posterior_mean(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        return (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start
            + self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )

    def q_posterior_mean_logvar(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start
            + self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )

        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][
            :, None, None, None
        ].expand(-1, -1, self.img_size, self.img_size)

        return posterior_mean, posterior_log_variance_clipped

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:

        sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod[t][
            :, None, None, None
        ]
        sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod[t][
            :, None, None, None
        ]

        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * eps

    def pred_mean_logvar(
        self,
        mean_pred: torch.Tensor,
        var_pred: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        - param mean_pred: Predicted values of mean from model.
        - param var_pred: Predicted values of varience from model.
        - param x: the [N x C x ...] tensor at time t.
        - param t: a 1-D Tensor of timesteps.

        - return: a tuple with the following tensors:
                 - 'mean': the model mean output.
                 - 'log_variance': the log of 'variance'.
        """

        min_log = self.posterior_log_variance_clipped[t][:, None, None, None]
        max_log = torch.log(self.betas)[t][:, None, None, None]

        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (var_pred + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log

        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=mean_pred).clamp(
            -1, 1
        )

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean = self.q_posterior_mean(x_start=pred_xstart, x_t=x, t=t)

        return model_mean, model_log_variance

    def sample(self, model: nn.Module, labels: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            batch_size = labels.size()[0]

            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(
                self.device
            )

            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                model_mean_pred, model_var_pred = model(x, t, labels)
                mean_pred, logvar_pred = self.pred_mean_logvar(
                    model_mean_pred, model_var_pred, x, t
                )

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = mean_pred + torch.exp(0.5 * logvar_pred) * noise

        model.train()
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

    def sample_distil(
        self, model: nn.Module, labels: torch.Tensor, num_timesteps: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs guided sampling from the diffusion model while storing intermediate states
        at specific timesteps for knowledge distillation.

        This function simulates the reverse diffusion process, generating samples step by step.
        At a predefined set of `num_timesteps`, it records the timestep, input image, predicted
        mean, and predicted variance for later use in distillation.

        Args:
            model (nn.Module): The trained diffusion model used for sampling.
            labels (torch.Tensor): Conditional labels for the model.
            num_timesteps (int): Number of timesteps at which intermediate data should be stored.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: A list of tuples where each tuple contains:
                - timestep (torch.Tensor): The diffusion step at which the data was recorded.
                - input_image (torch.Tensor): The noisy input image at that timestep.
                - predicted_mean (torch.Tensor): The model's predicted mean at that timestep.
                - predicted_variance (torch.Tensor): The model's predicted variance at that timestep.
        """

        timesteps = sorted(
            random.sample(population=range(1, self.noise_steps), k=num_timesteps),
            reverse=True,
        )

        t_ind = 0

        model.eval()
        stored_data: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        with torch.no_grad():
            batch_size = labels.size()[0]

            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(
                self.device
            )

            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                model_mean_pred, model_var_pred = model(x, t, labels)
                mean_pred, logvar_pred = self.pred_mean_logvar(
                    model_mean_pred, model_var_pred, x, t
                )

                # If current timestep is in the set, store a tuple of (t, input_image, predicted_mean, predicted_variance)
                if i == timesteps[t_ind]:
                    stored_data.append(
                        (
                            t.clone(),
                            x.clone(),
                            mean_pred.clone(),
                            logvar_pred.clone(),
                        )
                    )
                    t_ind += 1
                    if t_ind == num_timesteps:
                        break

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = mean_pred + torch.exp(0.5 * logvar_pred) * noise

        model.train()
        return stored_data


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(
        self,
        dataset: str,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | None = None,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(dataset, noise_steps, device)

    def prepare_noise_schedule(self) -> torch.Tensor:
        print("Preparing Linear Noise Schedule")
        return torch.linspace(
            self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float
        )


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(
        self,
        dataset: str,
        noise_steps: int = 250,
        s: float = 0.008,
        device: torch.device | None = None,
    ):
        self.s = s  # Small offset to avoid singularities
        super().__init__(dataset, noise_steps, device)

    def prepare_noise_schedule(self) -> torch.Tensor:

        print("Preparing Cosine Noise Schedule")

        timesteps = torch.linspace(
            0,
            self.noise_steps,
            self.noise_steps + 1,
            device=self.device,
            dtype=torch.float,
        )

        alphas_hat = (
            torch.cos(
                ((timesteps / self.noise_steps) + self.s) / (1 + self.s) * math.pi / 2
            )
            ** 2
        )
        alphas_hat = alphas_hat / alphas_hat[0]  # Normalize
        betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])  # Convert to beta
        return betas.clamp(min=1e-8, max=0.999)  # Avoid numerical issues
