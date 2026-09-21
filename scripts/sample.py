#!/usr/bin/env python3
"""Generate samples from a trained DDPM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddpm_scratch.diffusion import GaussianDiffusion
from ddpm_scratch.schedule import NoiseSchedule
from ddpm_scratch.unet import SmallUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="./samples/generated.png")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    schedule = NoiseSchedule(
        timesteps=int(cfg["timesteps"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        device=device,
    )
    mults = tuple(int(m) for m in cfg["channel_mults"])
    unet = SmallUNet(
        in_channels=1,
        base_channels=int(cfg["base_channels"]),
        channel_mults=mults,
    )
    unet.load_state_dict(ckpt["model"])
    diffusion = GaussianDiffusion(unet, schedule).to(device)
    diffusion.eval()

    samples = diffusion.sample((args.n, 1, 28, 28), device=device)
    images = (samples + 1.0) * 0.5
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, str(out), nrow=int(args.n**0.5) or 1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
