"""Dependency-free inspection for the PlantVillage folder layout.

This module deliberately uses only the Python standard library so the raw
dataset can be checked before installing PyTorch for training.
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("Train", "Val", "Test")


def _class_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected split directory at: {directory}")
    classes = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in split: {directory}")
    return classes


def _image_count(directory: Path) -> int:
    return sum(
        1 for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def inspect_raw_dataset(root: Path) -> dict:
    """Check split/class alignment and return a JSON-serializable manifest."""
    classes_by_split = {split: _class_names(root / split) for split in SPLITS}
    reference = classes_by_split["Train"]
    for split in ("Val", "Test"):
        if classes_by_split[split] != reference:
            train_only = sorted(set(reference) - set(classes_by_split[split]))
            split_only = sorted(set(classes_by_split[split]) - set(reference))
            raise ValueError(
                f"Class mismatch for {split}. "
                f"Train-only: {train_only}; {split}-only: {split_only}"
            )

    train_dir = root / "Train"
    class_counts = {
        class_name: _image_count(train_dir / class_name)
        for class_name in reference
    }
    empty_classes = sorted(name for name, count in class_counts.items() if count == 0)
    if empty_classes:
        raise ValueError(f"Train classes without supported image files: {empty_classes}")

    return {
        "dataset_root": str(root.resolve()),
        "num_classes": len(reference),
        "classes": reference,
        "train_images": _image_count(train_dir),
        "validation_images": _image_count(root / "Val"),
        "test_images": _image_count(root / "Test"),
        "train_class_counts": class_counts,
    }


def write_raw_manifest(root: Path, output: Path) -> dict:
    """Inspect the raw folders and write their manifest as JSON."""
    manifest = inspect_raw_dataset(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
