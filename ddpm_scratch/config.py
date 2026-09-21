"""Tiny YAML config loader (no inventing defaults beyond documented keys)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} must be a mapping")
    return cfg


DEFAULT_CPU = {
    "dataset": "mnist",
    "data_root": "./data",
    "timesteps": 200,
    "beta_start": 1.0e-4,
    "beta_end": 2.0e-2,
    "base_channels": 32,
    "channel_mults": [1, 2, 4],
    "batch_size": 64,
    "lr": 2.0e-4,
    "epochs": 5,
    "num_workers": 0,
    "seed": 0,
    "device": "cpu",
    "checkpoint_dir": "./checkpoints",
    "sample_dir": "./samples",
    "log_every": 100,
    "sample_every_epochs": 1,
    "num_sample_images": 16,
}
