"""Gaussian diffusion training objective and ancestral sampling.

Implements the simplified noise-prediction loss (Ho et al. Eq. 14) and
DDPM ancestral sampling (Algorithm 2).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ddpm_scratch.schedule import NoiseSchedule


class GaussianDiffusion(nn.Module):
    """Wraps a noise-prediction network with the DDPM forward/reverse math."""

    def __init__(self, model: nn.Module, schedule: NoiseSchedule) -> None:
        super().__init__()
        self.model = model
        self.schedule = schedule

    @property
    def timesteps(self) -> int:
        return self.schedule.timesteps

    def p_losses(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Simplified MSE objective: ``E || eps - eps_theta(x_t, t) ||^2``."""
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.schedule.q_sample(x0, t, noise=noise)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_mean_variance(self, xt: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Model mean / variance for ``p_theta(x_{t-1} | x_t)``."""
        eps = self.model(xt, t)
        x0_pred = self.schedule.predict_x0_from_eps(xt, t, eps)
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        return self.schedule.q_posterior_mean_variance(x0_pred, xt, t)

    @torch.no_grad()
    def p_sample(self, xt: Tensor, t: Tensor) -> Tensor:
        """One ancestral reverse step (Ho Algorithm 2, lines 3–5)."""
        mean, _, log_var = self.p_mean_variance(xt, t)
        noise = torch.randn_like(xt)
        # No noise when t == 0
        nonzero = (t > 0).float().reshape(-1, *([1] * (xt.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        """Generate samples from ``p(x_0)`` by iterating t = T-1 ... 0."""
        device = torch.device(device) if device is not None else self.schedule.device
        xt = torch.randn(shape, device=device)
        T = self.timesteps
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t)
            if progress_fn is not None:
                progress_fn(T - i, T)
        return xt.clamp(-1.0, 1.0)

    def forward(self, x0: Tensor) -> Tensor:
        """Training step: sample random ``t`` and return L_simple."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        return self.p_losses(x0, t)
