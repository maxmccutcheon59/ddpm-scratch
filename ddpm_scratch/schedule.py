"""Noise schedules for DDPM forward / reverse processes.

Implements the linear beta schedule from Ho et al., NeurIPS 2020
("Denoising Diffusion Probabilistic Models"), Section 4 / Appendix.
"""

from __future__ import annotations

import torch
from torch import Tensor


class NoiseSchedule:
    """Discrete-time Gaussian diffusion schedule over T steps.

    Stores betas, alphas, and the closed-form coefficients used by
    q(x_t | x_0) and the reverse posterior q(x_{t-1} | x_t, x_0).
    """

    def __init__(
        self,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str | None = None,
    ) -> None:
        if timesteps < 1:
            raise ValueError(f"timesteps must be >= 1, got {timesteps}")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError(
                f"need 0 < beta_start < beta_end < 1, got "
                f"beta_start={beta_start}, beta_end={beta_end}"
            )

        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        betas = torch.linspace(
            self.beta_start, self.beta_end, self.timesteps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]]
        )

        # q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Posterior q(x_{t-1} | x_t, x_0) variance (Ho et al. Eq. 7)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Clamp t=0 entry (undefined mathematically; unused in sampling)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_log_variance_clipped = torch.log(posterior_variance)

        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            self.device
        )
        self.posterior_variance = posterior_variance.to(self.device)
        self.posterior_log_variance_clipped = posterior_log_variance_clipped.to(
            self.device
        )
        self.posterior_mean_coef1 = posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = posterior_mean_coef2.to(self.device)

    def to(self, device: torch.device | str) -> NoiseSchedule:
        """Move all schedule tensors to ``device`` (in-place) and return self."""
        self.device = torch.device(device)
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    def _extract(self, a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        """Gather schedule values at timesteps ``t`` and reshape for broadcast."""
        out = a.gather(0, t.long())
        return out.reshape(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Forward diffusion: sample ``x_t`` from ``q(x_t | x_0)`` (Ho Eq. 4)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0_from_eps(self, xt: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Recover ``x_0`` from ``x_t`` and predicted noise (rearrangement of Eq. 4)."""
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return (xt - sqrt_omab * eps) / sqrt_ab.clamp(min=1e-8)

    def q_posterior_mean_variance(
        self, x0: Tensor, xt: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """True posterior mean / variance of ``q(x_{t-1} | x_t, x_0)``."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, xt.shape) * x0
            + self._extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        var = self._extract(self.posterior_variance, t, xt.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, xt.shape)
        return mean, var, log_var
