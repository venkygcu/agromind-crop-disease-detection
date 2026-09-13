# Week 1 research review

## Scope and decision

The Week 1 target is a local classifier for the supplied PlantVillage-style
Train/Val/Test dataset—not a claim of field-ready disease diagnosis. The
recommended baseline is ImageNet-pretrained ResNet-18, fine-tuned for the
validated class count. This is a practical starting point for the required
test accuracy of at least 88%; it still needs to meet that target in a later,
reproducible training run.

## 1. Dataset: Hughes & Salathe (2015)

**Resource:** [An open access repository of images on plant health to enable
the development of mobile disease diagnostics](https://arxiv.org/abs/1511.08060)

Hughes and Salathe released over 50,000 expertly curated images of healthy and
infected crop leaves through PlantVillage. The dataset makes supervised
classification practical, but its controlled backgrounds and image capture
conditions are materially different from typical farmer photographs.

**Decision:** preserve the supplied Test split for one final evaluation and
describe the result as controlled-dataset performance. Do not turn an 88% test
result into a field-performance claim.

## 2. CNN benchmark: Ferentinos (2018)

**Paper:** [Deep learning models for plant disease detection and
diagnosis](https://doi.org/10.1016/j.compag.2018.01.009)

Ferentinos evaluates deep convolutional models for multi-class plant disease
recognition and shows that strong accuracy is attainable on leaf-image
benchmarks. It supports using a CNN baseline instead of handcrafted features,
but does not eliminate the need for split hygiene and per-class evaluation.

**Decision:** start with a pretrained CNN rather than training a network from
scratch. ResNet-18 keeps local training and inference relatively light while
retaining a well-established transfer-learning backbone.

## 3. Domain shift: Tm et al. (2022)

**Paper:** [Transfer learning for versatile plant disease recognition with
limited data](https://pmc.ncbi.nlm.nih.gov/articles/PMC9726777/)

This work compares transfer-learning approaches and highlights the difference
between controlled leaf images and less constrained field imagery. Aggregate
accuracy alone is therefore insufficient for safe user-facing output.

**Decision:** use training-only augmentation, report macro F1, per-class recall
and a confusion matrix, and have the future API return a caution for
low-confidence predictions. Treatment text is informational and must direct
users to local agronomy guidance before applying any product.

## Research-to-implementation traceability

| Finding | Week 1 implementation decision |
| --- | --- |
| PlantVillage is curated and controlled | Keep an untouched test set; document the domain limit. |
| CNNs perform well on leaf-image benchmarks | Use an ImageNet-pretrained ResNet-18 baseline. |
| Field images shift the data distribution | Add modest training augmentation and confidence-aware API behaviour. |
| Similar-looking classes can hide weak performance | Require macro F1, per-class recall and a confusion matrix in Week 2. |
