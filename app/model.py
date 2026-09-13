"""Transfer-learning model factory selected in the Week 1 architecture."""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CropDiseaseResNet18(nn.Module):
    """ResNet-18 feature extractor with a dropout classification head."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for a batch of RGB images."""
        features = self.features(images)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(self.dropout(flattened))


def build_model(num_classes: int, pretrained: bool = True) -> CropDiseaseResNet18:
    """Create an ImageNet-initialized ResNet-18 with a PlantVillage head."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    return CropDiseaseResNet18(num_classes=num_classes, pretrained=pretrained)
