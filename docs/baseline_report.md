# Baseline report

## Status: benchmark pending

The baseline pipeline and ResNet-18 trainer are implemented in `app/train_baseline.py`, but no completed training run or test score has been captured yet. The refreshed raw-data manifest finds 29 class folders and 44,423 training images overall, but reports **zero accessible training images** for `Corn (Maize) - Cercospora Leaf Spot`. PowerShell lists filenames in that folder, while Python's `Path.is_file()` cannot access them, which is consistent with unhydrated OneDrive placeholders. Resolve that discrepancy and rerun the manifest/trainer before treating the split as benchmark-ready.

The manifest records 10,679 validation and 1,193 test images overall. Class alignment alone is therefore insufficient to certify this as a 29-class benchmark. The PyTorch data pipeline now rejects any split with an empty class rather than silently training/evaluating on incomplete labels.

## Benchmark definition

- Model: ImageNet-pretrained ResNet-18, dropout 0.25, linear 29-class head
- Input: RGB, resized to 224 × 224, ImageNet normalized
- Train augmentation: horizontal flip, ±15° rotation, modest color jitter
- Selection: validation cross-entropy, early stopping; test split held out
- Primary metric: test top-1 accuracy; report macro F1 and per-class precision/recall/F1
- Acceptance target from `docs/architecture.md`: at least 88% test accuracy

## Results

| Metric | Result |
|---|---:|
| Test top-1 accuracy | Not measured: no completed training run |
| Test macro F1 | Not measured: no completed training run |
| Per-class scores | Not measured |

No model checkpoint or score is claimed. Once all split files are accessible, run the command in the README to train and produce a measured report under `artifacts/baseline/`.

## Reproducibility and scope

The trainer defaults to seed 42 and records the seed, class map, epoch history, model configuration, and device in `metrics.json` and the checkpoint. The available runtime has CPU-only PyTorch. Even a successful score measures the supplied PlantVillage-style split and does not establish field performance.
