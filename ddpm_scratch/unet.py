"""Small U-Net noise predictor for low-res images (MNIST / Fashion-MNIST).

Original compact architecture inspired by the U-Net used in Ho et al. 2020,
scaled down for CPU-friendly training. Not a copy of any public tutorial repo.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (same functional form as Transformer PE)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and optional time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F_interpolate(x)
        return self.conv(x)


def F_interpolate(x: Tensor) -> Tensor:
    return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class SmallUNet(nn.Module):
    """Compact U-Net: base_ch -> 2x -> 4x channels, two residual blocks / stage.

    Default ``base_channels=32`` keeps parameter count small for CPU training
    on 28x28 grayscale images.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        time_dim = time_dim or base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        channels_list = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, time_dim),
                        ResBlock(out_ch, out_ch, time_dim),
                    ]
                )
            )
            channels_list.append(out_ch)
            ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                channels_list.append(ch)

        # Mid
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # Up path
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # skip concat doubles channels
            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(ch + out_ch, out_ch, time_dim),
                        ResBlock(out_ch, out_ch, time_dim),
                    ]
                )
            )
            ch = out_ch
            if i > 0:
                self.upsamples.append(Upsample(ch))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        temb = self.time_mlp(t)
        h = self.conv_in(x)
        skips: list[Tensor] = []

        ds_idx = 0
        for i, blocks in enumerate(self.downs):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[ds_idx](h)
                ds_idx += 1

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        us_idx = 0
        for i, blocks in enumerate(self.ups):
            skip = skips.pop()
            # Spatial align if off-by-one from odd sizes
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="nearest"
                )
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, temb)
            if i < len(self.upsamples):
                h = self.upsamples[us_idx](h)
                us_idx += 1

        return self.conv_out(self.act(self.norm_out(h)))
