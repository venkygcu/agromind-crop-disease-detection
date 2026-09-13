# Paper review 2: Deep-learning benchmark

## Citation

Ferentinos, K. P. (2018). *Deep learning models for plant disease detection and diagnosis.* Computers and Electronics in Agriculture, 145, 311-318. [Paper](https://doi.org/10.1016/j.compag.2018.01.009)

## Review

This study evaluates CNNs for multi-class plant disease recognition and demonstrates that convolutional image models can learn visual features useful for crop-condition classification. It supports using a CNN rather than manually engineered image features.

## Project decision

Use ImageNet-pretrained ResNet-18 with a new PlantVillage classification head. Report top-1 accuracy, macro F1, per-class recall, and a confusion matrix rather than accuracy alone.
