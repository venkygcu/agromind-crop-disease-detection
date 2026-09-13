# Paper review 1: PlantVillage dataset

## Citation

Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060. [Paper](https://arxiv.org/abs/1511.08060)

## Review

This paper introduced the PlantVillage repository of expertly curated healthy and diseased leaf photographs. It makes supervised image classification practical for crop disease recognition. The images are captured in relatively controlled conditions, so the dataset is useful for training and benchmarking but cannot alone demonstrate equal accuracy on farm photographs with shadows, cluttered backgrounds, or poor camera quality.

## Project decision

Use the supplied PlantVillage data for development, preserve `Test/` for the final accuracy measurement, and document the controlled-dataset limitation in the API and final report.
