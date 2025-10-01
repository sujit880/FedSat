import torch
import math


def mean_flat(x: torch.Tensor):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def kl_loss(
    mean_target: torch.Tensor,
    logvar_target: torch.Tensor,
    mean_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5
        * mean_flat(
            (
                -1.0
                + logvar_pred
                - logvar_target
                + torch.exp(logvar_target - logvar_pred)
                + ((mean_target - mean_pred) ** 2) * torch.exp(-logvar_pred)
            )
        )
        / math.log(2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    - param x: the target images. It is assumed that this was uint8 values,
            rescaled to the range [-1, 1].
    - param means: the Gaussian mean Tensor.
    - param log_scales: the Gaussian log stddev Tensor.
    - return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )

    return log_probs
