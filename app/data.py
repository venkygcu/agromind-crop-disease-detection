"""Reproducible ImageFolder data pipeline for PlantVillage-style directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

Split = Literal["Train", "Val", "Test"]
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def train(self) -> Path:
        return self.root / "Train"

    @property
    def validation(self) -> Path:
        return self.root / "Val"

    @property
    def test(self) -> Path:
        return self.root / "Test"


def transforms_for(split: Split) -> transforms.Compose:
    """Return ImageNet-normalized transforms, augmenting only training images."""
    common = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if split == "Train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            *common,
        ])
    return transforms.Compose(common)


def build_dataset(paths: DatasetPaths, split: Split) -> datasets.ImageFolder:
    directories = {
        "Train": paths.train,
        "Val": paths.validation,
        "Test": paths.test,
    }
    directory = directories[split]
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected {split} directory at: {directory}")
    return datasets.ImageFolder(directory, transform=transforms_for(split))


def build_loader(dataset: datasets.ImageFolder, batch_size: int = 32,
                 shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def inspect_dataset(root: Path) -> dict:
    """Validate Train/Val/Test class alignment and return a reproducible manifest."""
    paths = DatasetPaths(root)
    train = build_dataset(paths, "Train")
    validation = build_dataset(paths, "Val")
    test = build_dataset(paths, "Test")
    for split_name, split_dataset in (("Val", validation), ("Test", test)):
        if train.classes != split_dataset.classes:
            only_train = sorted(set(train.classes) - set(split_dataset.classes))
            only_split = sorted(set(split_dataset.classes) - set(train.classes))
            raise ValueError(
                f"Class mismatch for {split_name}. "
                f"Train-only: {only_train}; {split_name}-only: {only_split}"
            )
    counts = {name: sum(label == index for _, label in train.samples)
              for index, name in enumerate(train.classes)}
    return {"dataset_root": str(root.resolve()), "num_classes": len(train.classes),
            "classes": train.classes, "train_images": len(train),
            "validation_images": len(validation), "test_images": len(test),
            "train_class_counts": counts}


def write_manifest(root: Path, output: Path) -> dict:
    manifest = inspect_dataset(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
