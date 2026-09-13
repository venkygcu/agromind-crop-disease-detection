# AgroMind AgriTech - Crop Disease Detection System

Week 1 foundation for a CNN-based crop disease classifier using the PlantVillage dataset. The final local service will accept a leaf photo and return a predicted disease label, confidence score, and rule-based treatment advisory.

## Week 1 deliverables

- Research review of three relevant papers/resources: [`docs/research_review.md`](docs/research_review.md)
- Model, data, evaluation, and serving design: [`docs/architecture.md`](docs/architecture.md)
- Reproducible `torchvision.ImageFolder` data pipeline in `app/data.py`, including split/class validation
- Dataset inspection CLI and automated skeleton tests

## Project layout

```text
app/                 data pipeline and model factory
data/                supplied PlantVillage-style Train/, Val/, and Test/ folders
docs/                research review and architecture decision record
models/              generated checkpoints (not committed)
notebooks/           exploratory research
tests/               pipeline checks
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validate the supplied data

The repository expects:

```text
data/Crop Disease Detection Dataset/Plant Village Dataset/
  Train/<crop - condition>/*.jpg
  Val/<crop - condition>/*.jpg
  Test/<crop - condition>/*.jpg
```

Run the manifest step before training:

```powershell
python -m app.prepare_data
```

It verifies that class names and indices agree across Train, Val, and Test, then writes a reproducible manifest to `data/processed/dataset_manifest.json`. This command uses only the Python standard library, so it works before installing PyTorch.

## Architecture at a glance

`leaf image -> RGB/224x224 transform -> ImageNet-pretrained ResNet-18 -> softmax class probabilities -> disease label + confidence -> advisory rule engine`

The acceptance target is **at least 88% top-1 accuracy on the untouched Test split**. `Val` is used for training decisions only. The architecture document details metrics, leakage prevention, and the controlled-lab-versus-field limitation.

## Planned milestones

1. Week 1: research, architecture, and data pipeline complete
2. Weeks 2-4: train, tune, and evaluate the transfer-learning model
3. Weeks 5-7: advisory rule engine and FastAPI endpoint
4. Weeks 8-10: local deployment, testing, and documentation

## Week 1 verification

After installing the requirements, run:

```powershell
python -m pytest -q
python -m app.prepare_data
```
