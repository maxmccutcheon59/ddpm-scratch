#!/usr/bin/env python3
"""Train a small DDPM on MNIST or Fashion-MNIST (CPU-friendly defaults)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Allow running without install: repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddpm_scratch.config import load_config
from ddpm_scratch.data import get_dataloader
from ddpm_scratch.diffusion import GaussianDiffusion
from ddpm_scratch.schedule import NoiseSchedule
from ddpm_scratch.unet import SmallUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "cpu_mnist.yaml"),
        help="Path to YAML config",
    )
    p.add_argument("--device", type=str, default=None, help="Override device")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--max-train-steps", type=int, default=None, help="Cap steps (smoke)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> GaussianDiffusion:
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
    ).to(device)
    return GaussianDiffusion(unet, schedule).to(device)


@torch.no_grad()
def save_samples(
    diffusion: GaussianDiffusion,
    path: Path,
    n: int,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = diffusion.sample((n, 1, 28, 28), device=device)
    # map [-1,1] -> [0,1]
    images = (samples + 1.0) * 0.5
    save_image(images, str(path), nrow=int(n**0.5) or 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_train_steps is not None:
        cfg["max_train_steps"] = args.max_train_steps

    device = torch.device(cfg.get("device", "cpu"))
    # Refuse silent GPU cloud spend: default stays CPU unless user overrides.
    set_seed(int(cfg.get("seed", 0)))

    ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
    sample_dir = Path(cfg.get("sample_dir", "./samples"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(
        name=str(cfg["dataset"]),
        root=str(cfg.get("data_root", "./data")),
        batch_size=int(cfg["batch_size"]),
        train=True,
        num_workers=int(cfg.get("num_workers", 0)),
        download=True,
    )

    diffusion = build_model(cfg, device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=float(cfg["lr"]))

    n_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"device={device}  params={n_params:,}  T={cfg['timesteps']}")

    max_steps = cfg.get("max_train_steps")
    global_step = 0
    history: list[dict] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        diffusion.train()
        running = 0.0
        count = 0
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            loss = diffusion(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            count += 1
            global_step += 1

            if global_step % int(cfg.get("log_every", 100)) == 0:
                avg = running / max(count, 1)
                print(f"epoch={epoch} step={global_step} loss={avg:.4f}")

            if max_steps is not None and global_step >= int(max_steps):
                break

        avg_loss = running / max(count, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "steps": global_step})
        print(f"epoch={epoch} done  mean_loss={avg_loss:.4f}")

        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": diffusion.model.state_dict(),
                "config": cfg,
                "loss": avg_loss,
            },
            ckpt_path,
        )

        if epoch % int(cfg.get("sample_every_epochs", 1)) == 0:
            diffusion.eval()
            out = sample_dir / f"epoch_{epoch:03d}.png"
            save_samples(
                diffusion,
                out,
                n=int(cfg.get("num_sample_images", 16)),
                device=device,
            )
            print(f"wrote {out}")

        if max_steps is not None and global_step >= int(max_steps):
            print(f"stopped early at max_train_steps={max_steps}")
            break

    metrics_path = ckpt_dir / "train_history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"history -> {metrics_path}")


if __name__ == "__main__":
    main()
