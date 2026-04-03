from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    batch_size: int = 32
    num_workers: int = 2
    image_size: int = 224
    pin_memory: bool = True
    download_if_missing: bool = False


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _resolve_food101_dataset(
    root: Path,
    split: str,
    transform: Callable,
    download_if_missing: bool,
) -> Dataset:
    try:
        return datasets.Food101(
            root=str(root),
            split=split,
            transform=transform,
            download=download_if_missing,
        )
    except Exception:
        split_dir = root / "food-101" / split
        if split_dir.exists():
            return ImageFolder(root=str(split_dir), transform=transform)
        raise


class Food101DataModule:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        if self.config.batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if not self.config.data_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.config.data_root}")

        train_ds = _resolve_food101_dataset(
            root=self.config.data_root,
            split="train",
            transform=build_train_transform(self.config.image_size),
            download_if_missing=self.config.download_if_missing,
        )
        val_ds = _resolve_food101_dataset(
            root=self.config.data_root,
            split="test",
            transform=build_eval_transform(self.config.image_size),
            download_if_missing=False,
        )
        test_ds = _resolve_food101_dataset(
            root=self.config.data_root,
            split="test",
            transform=build_eval_transform(self.config.image_size),
            download_if_missing=False,
        )

        pin_memory = self.config.pin_memory and torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    def get_class_names(self) -> list[str]:
        train_ds = _resolve_food101_dataset(
            root=self.config.data_root,
            split="train",
            transform=build_eval_transform(self.config.image_size),
            download_if_missing=self.config.download_if_missing,
        )
        classes = getattr(train_ds, "classes", None)
        if classes is None:
            return []
        return [str(x) for x in classes]
