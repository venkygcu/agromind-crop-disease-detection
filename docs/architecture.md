# Week 1 architecture and evaluation plan

## Objective

Classify one uploaded leaf image into the PlantVillage crop-condition class. The final local FastAPI service will return the predicted class, softmax confidence, and a treatment advisory selected by a deterministic rule table. The required test-set accuracy is at least 88%.

## Selected model

**ResNet-18 transfer learning** is the baseline and planned production model. It accepts a 224 x 224 RGB image and uses ImageNet-pretrained convolutional features, followed by global average pooling and a new `Dropout(0.25) + Linear` head. Its output dimension is derived from the validated dataset manifest (the supplied pack has 29 classes). It is smaller and faster than ResNet-50 while retaining residual connections and strong transfer-learning behaviour.

Training will proceed in two stages:

1. Freeze the backbone and train the head for 3-5 epochs using AdamW and cross-entropy loss.
2. Unfreeze all layers and fine-tune with a lower learning rate, early stopping, and best-validation-checkpoint selection.

The pipeline resizes images to 224 x 224 and applies ImageNet normalization. Training-only augmentation uses horizontal flips, small rotations, and modest colour jitter. Validation and test preprocessing are deterministic.

## Data and split policy

The supplied `Train/`, `Val/`, and `Test/` folders are read with `torchvision.ImageFolder`. `Val/` is reserved for model selection and early stopping; `Test/` remains untouched until one final acceptance evaluation. The manifest step validates class names and class-index order across all three splits, then writes `data/processed/dataset_manifest.json`.

The directory contains augmented variants, so training split construction must group files by the identifier before `___`. This prevents rotations/flips of one original leaf crossing train and validation partitions, an important leakage guard.

## Evaluation criteria

| Metric | Role | Target |
| --- | --- | --- |
| Top-1 accuracy | Acceptance metric on untouched `Test/` | >= 88% |
| Macro F1 | Detect weak minority classes | Reported, maximize |
| Per-class recall | Disease safety check | Report all classes |
| Confusion matrix | Diagnose similar disease classes | Saved as artifact |
| Calibration / confidence | Safe API wording | Flag predictions below 0.60 |

Each experiment records seed, data manifest, class mapping, hyperparameters, epoch metrics and checkpoint. PlantVillage images are captured under controlled conditions; accuracy on this split is not a claim of equal field performance.

## Serving design

`POST /predict` -> validate image -> inference transform -> ResNet-18 -> softmax -> class map -> advisory rule lookup -> JSON response.

Invalid image files receive HTTP 422. Predictions below 0.60 confidence will ask for a clearer photo and recommend local expert confirmation rather than present a label as a diagnosis.
