"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

from ddpm_scratch.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_cpu_mnist_config():
    cfg = load_config(ROOT / "configs" / "cpu_mnist.yaml")
    assert cfg["dataset"] == "mnist"
    assert cfg["device"] == "cpu"
    assert cfg["timesteps"] == 200
    assert cfg["base_channels"] == 32


def test_load_smoke_config():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg["max_train_steps"] == 3
    assert cfg["device"] == "cpu"
