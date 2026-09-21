"""Unit tests for DDPM noise schedule math (Ho et al. 2020)."""

from __future__ import annotations

import math

import pytest
import torch

from ddpm_scratch.schedule import NoiseSchedule


def test_betas_monotonic_and_in_range():
    sch = NoiseSchedule(timesteps=100, beta_start=1e-4, beta_end=2e-2)
    assert sch.betas.shape == (100,)
    assert torch.all(sch.betas[1:] >= sch.betas[:-1])
    assert float(sch.betas[0]) == pytest.approx(1e-4, rel=1e-5)
    assert float(sch.betas[-1]) == pytest.approx(2e-2, rel=1e-5)
    assert torch.all(sch.betas > 0) and torch.all(sch.betas < 1)


def test_alphas_cumprod_decreasing():
    sch = NoiseSchedule(timesteps=50)
    ab = sch.alphas_cumprod
    assert torch.all(ab[1:] <= ab[:-1])
    assert float(ab[0]) < 1.0
    assert float(ab[-1]) > 0.0
    # alpha_bar_t = prod (1 - beta_s)
    expected = torch.cumprod(1.0 - sch.betas, dim=0)
    assert torch.allclose(ab, expected, atol=1e-6)


def test_q_sample_at_t0_near_x0():
    sch = NoiseSchedule(timesteps=10, beta_start=1e-4, beta_end=1e-3)
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.zeros(4, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = sch.q_sample(x0, t, noise=noise)
    # Small beta => xt ≈ sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise ≈ x0 + tiny noise
    assert xt.shape == x0.shape
    # Reconstruction identity: x0 ≈ (xt - sqrt(1-ab) eps) / sqrt(ab)
    x0_hat = sch.predict_x0_from_eps(xt, t, noise)
    assert torch.allclose(x0_hat, x0, atol=1e-5)


def test_q_sample_deterministic_with_fixed_noise():
    sch = NoiseSchedule(timesteps=20)
    x0 = torch.ones(2, 1, 4, 4)
    t = torch.tensor([5, 15])
    noise = torch.zeros_like(x0)
    xt = sch.q_sample(x0, t, noise=noise)
    # With zero noise: xt = sqrt(alpha_bar_t) * x0
    for i, ti in enumerate(t.tolist()):
        scale = math.sqrt(float(sch.alphas_cumprod[ti]))
        assert torch.allclose(xt[i], x0[i] * scale, atol=1e-5)


def test_posterior_variance_nonnegative():
    sch = NoiseSchedule(timesteps=30)
    assert torch.all(sch.posterior_variance >= 0)
    # t=0 posterior unused; remaining should be finite
    assert torch.all(torch.isfinite(sch.posterior_log_variance_clipped))


def test_invalid_hparams_raise():
    with pytest.raises(ValueError):
        NoiseSchedule(timesteps=0)
    with pytest.raises(ValueError):
        NoiseSchedule(timesteps=10, beta_start=0.5, beta_end=0.1)
    with pytest.raises(ValueError):
        NoiseSchedule(timesteps=10, beta_start=-0.1, beta_end=0.1)


def test_extract_broadcast_shape():
    sch = NoiseSchedule(timesteps=16)
    x = torch.randn(3, 1, 7, 7)
    t = torch.tensor([0, 7, 15])
    out = sch._extract(sch.betas, t, x.shape)
    assert out.shape == (3, 1, 1, 1)
