from pathlib import Path

from app.data import DatasetPaths, transforms_for
from app.manifest import inspect_raw_dataset
from app.model import build_model


def test_validation_transform_is_deterministic():
    transform = transforms_for("Val")
    assert "Resize" in repr(transform)
    assert "Normalize" in repr(transform)


def test_dataset_paths_follow_expected_layout(tmp_path: Path):
    paths = DatasetPaths(tmp_path)
    assert paths.train == tmp_path / "Train"
    assert paths.validation == tmp_path / "Val"
    assert paths.test == tmp_path / "Test"


def test_raw_manifest_reports_empty_training_class(tmp_path: Path):
    for split in ("Train", "Val", "Test"):
        (tmp_path / split / "Tomato - healthy").mkdir(parents=True)

    manifest = inspect_raw_dataset(tmp_path)

    assert manifest["empty_train_classes"] == ["Tomato - healthy"]


def test_raw_manifest_reports_counts(tmp_path: Path):
    for split in ("Train", "Val", "Test"):
        class_dir = tmp_path / split / "Tomato - healthy"
        class_dir.mkdir(parents=True)
        (class_dir / "leaf.JPG").write_bytes(b"not decoded by the manifest")

    manifest = inspect_raw_dataset(tmp_path)

    assert manifest["num_classes"] == 1
    assert manifest["train_images"] == 1
    assert manifest["train_class_counts"] == {"Tomato - healthy": 1}
    assert manifest["empty_train_classes"] == []


def test_model_outputs_one_score_per_class():
    model = build_model(num_classes=3, pretrained=False)
    assert model.classifier.out_features == 3
