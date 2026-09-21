"""U-Net shape / parameter smoke tests."""

from __future__ import annotations

import torch

from ddpm_scratch.unet import SmallUNet, SinusoidalPosEmb


def test_sinusoidal_embedding_shape():
    emb = SinusoidalPosEmb(32)
    t = torch.arange(5)
    out = emb(t)
    assert out.shape == (5, 32)


def test_unet_output_matches_input_shape():
    net = SmallUNet(in_channels=1, base_channels=16, channel_mults=(1, 2))
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 7])
    y = net(x, t)
    assert y.shape == x.shape


def test_unet_param_count_cpu_friendly():
    net = SmallUNet(in_channels=1, base_channels=32, channel_mults=(1, 2, 4))
    n = sum(p.numel() for p in net.parameters())
    # Expect well under 5M for this compact config (paper U-Nets are larger).
    assert n < 5_000_000
    assert n > 10_000
