"""Tests for forward loss and reverse sampling shapes / gradients."""

from __future__ import annotations

import torch

from ddpm_scratch.diffusion import GaussianDiffusion
from ddpm_scratch.schedule import NoiseSchedule
from ddpm_scratch.unet import SmallUNet


def _tiny_diffusion(T: int = 20) -> GaussianDiffusion:
    sch = NoiseSchedule(timesteps=T, beta_start=1e-4, beta_end=2e-2)
    net = SmallUNet(in_channels=1, base_channels=16, channel_mults=(1, 2))
    return GaussianDiffusion(net, sch)


def test_p_losses_scalar_and_grad():
    diff = _tiny_diffusion()
    x0 = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 10])
    loss = diff.p_losses(x0, t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in diff.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_forward_samples_random_t():
    diff = _tiny_diffusion()
    x0 = torch.randn(4, 1, 28, 28)
    loss = diff(x0)
    assert torch.isfinite(loss)


@torch.no_grad()
def test_sample_shape_and_range():
    diff = _tiny_diffusion(T=5)
    out = diff.sample((2, 1, 28, 28), device="cpu")
    assert out.shape == (2, 1, 28, 28)
    assert float(out.min()) >= -1.0 - 1e-5
    assert float(out.max()) <= 1.0 + 1e-5


@torch.no_grad()
def test_predict_x0_roundtrip_with_true_noise():
    """If the model were perfect, x0 reconstruction from true eps is exact."""
    sch = NoiseSchedule(timesteps=25)
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([3, 20])
    eps = torch.randn_like(x0)
    xt = sch.q_sample(x0, t, noise=eps)
    x0_hat = sch.predict_x0_from_eps(xt, t, eps)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
