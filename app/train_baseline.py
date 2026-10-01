"""Train and evaluate the reproducible ResNet-18 baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from app.data import DatasetPaths, build_dataset, build_loader, validate_splits
from app.model import build_model


def score_predictions(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    per_class = {}
    f1_values = []
    for index in range(num_classes):
        precision = matrix[index, index] / predicted_count[index] if predicted_count[index] else 0.0
        recall = matrix[index, index] / support[index] if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[index] = {"precision": precision, "recall": recall,
                            "f1-score": f1, "support": int(support[index])}
        if support[index]:
            f1_values.append(f1)
    return (float(np.trace(matrix) / max(1, matrix.sum())),
            float(np.mean(f1_values)) if f1_values else 0.0,
            per_class, matrix)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, targets = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return {"loss": total_loss / len(loader.dataset),
            "accuracy": score_predictions(targets, predictions, len(loader.dataset.classes))[0],
            "macro_f1": score_predictions(targets, predictions, len(loader.dataset.classes))[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/Crop Disease Detection Dataset/Plant Village Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="Do not load ImageNet weights")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.head_epochs < args.epochs:
        parser.error("epochs must be positive and head-epochs must be in [0, epochs)")

    set_seed(args.seed)
    paths = DatasetPaths(args.data_dir)
    train = build_dataset(paths, "Train")
    validation = build_dataset(paths, "Val")
    test = build_dataset(paths, "Test")
    validate_splits(train, validation, test)
    train_loader = build_loader(train, args.batch_size, shuffle=True)
    val_loader = build_loader(validation, args.batch_size)
    test_loader = build_loader(test, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep torchvision's optional pretrained-weight cache inside the writable project.
    torch.hub.set_dir(str((args.output_dir / "torch_hub").resolve()))
    model = build_model(len(train.classes), pretrained=not args.scratch).to(device)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "resnet18_baseline.pt"
    history, best_loss, stale_epochs = [], float("inf"), 0
    start = time.time()

    for epoch in range(args.epochs):
        if epoch < args.head_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
            learning_rate = 1e-3
        else:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            learning_rate = 1e-4
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics,
                        "learning_rate": learning_rate})
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss, stale_epochs = val_metrics["loss"], 0
            torch.save({"state_dict": model.state_dict(), "classes": train.classes,
                        "class_to_idx": train.class_to_idx, "image_size": 224,
                        "seed": args.seed, "epoch": epoch + 1,
                        "pretrained": not args.scratch}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = run_epoch(model, test_loader, criterion, device)
    model.eval()
    all_predictions, all_targets = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            all_predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            all_targets.extend(labels.tolist())
    _, _, indexed_report, matrix = score_predictions(all_targets, all_predictions, len(train.classes))
    report = {name: indexed_report[index] for index, name in enumerate(train.classes)}
    elapsed = time.time() - start
    results = {"dataset": {"train_images": len(train), "validation_images": len(validation),
                           "test_images": len(test), "num_classes": len(train.classes),
                           "classes": train.classes},
               "config": {"architecture": "ResNet-18", "pretrained": not args.scratch,
                          "seed": args.seed, "batch_size": args.batch_size, "device": str(device),
                          "epochs_requested": args.epochs, "epochs_completed": len(history),
                          "best_epoch": checkpoint["epoch"], "elapsed_seconds": elapsed},
               "history": history,
               "test": {"loss": test_result["loss"], "accuracy": test_result["accuracy"],
                        "macro_f1": test_result["macro_f1"], "per_class": report,
                        "confusion_matrix": matrix.tolist()}}
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.output_dir / "baseline_report.md")
    print(f"Test accuracy={test_result['accuracy']:.4f}; macro-F1={test_result['macro_f1']:.4f}")
    print(f"Saved checkpoint and report under {args.output_dir}")


def write_report(results: dict, path: Path) -> None:
    test, config, dataset = results["test"], results["config"], results["dataset"]
    lines = ["# ResNet-18 baseline report", "",
             "## Benchmark", "",
             "The untouched `Test/` split is used once for final evaluation. `Val/` selects the checkpoint.", "",
             f"- Test images: {dataset['test_images']:,} across {dataset['num_classes']} classes",
             f"- Train / validation images: {dataset['train_images']:,} / {dataset['validation_images']:,}",
             f"- Top-1 accuracy: **{test['accuracy']:.4f}**",
             f"- Macro F1: **{test['macro_f1']:.4f}**",
             f"- Cross-entropy: {test['loss']:.4f}",
             f"- Model: ResNet-18; ImageNet pretrained: {config['pretrained']}",
             f"- Seed: {config['seed']}; best validation epoch: {config['best_epoch']}; device: {config['device']}",
             "", "## Per-class results", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name in dataset["classes"]:
        item = test["per_class"][name]
        lines.append(f"| {name} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1-score']:.3f} | {int(item['support'])} |")
    lines += ["", "## Artifacts", "", "`resnet18_baseline.pt` contains the selected checkpoint and class mapping. `metrics.json` contains epoch history, per-class metrics, and the confusion matrix.", "",
              "## Limitations", "", "This benchmark measures performance on the supplied PlantVillage-style split. It does not establish field performance on photographs with varied lighting, backgrounds, or devices.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
