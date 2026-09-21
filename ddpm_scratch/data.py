"""Dataset helpers for MNIST / Fashion-MNIST (normalized to [-1, 1])."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
        ]
    )


def get_dataset(
    name: str = "mnist",
    root: str | Path = "./data",
    train: bool = True,
    download: bool = True,
) -> datasets.VisionDataset:
    name = name.lower().replace("-", "").replace("_", "")
    root = Path(root)
    tfm = make_transform()
    if name == "mnist":
        return datasets.MNIST(root=str(root), train=train, download=download, transform=tfm)
    if name in ("fashionmnist", "fashion"):
        return datasets.FashionMNIST(
            root=str(root), train=train, download=download, transform=tfm
        )
    raise ValueError(f"unknown dataset {name!r}; use 'mnist' or 'fashionmnist'")


def get_dataloader(
    name: str = "mnist",
    root: str | Path = "./data",
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    download: bool = True,
) -> DataLoader:
    ds = get_dataset(name=name, root=root, train=train, download=download)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
