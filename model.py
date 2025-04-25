import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str,
                num_classes: int,
                pretrained: bool) -> nn.Module:
    """
    Instantiate a ResNet with a new final layer.

    Args:
      model_name: "resnet18"
      num_classes: number of output classes
      pretrained: whether to load ImageNet‐pretrained weights
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # replace the final fully‐connected layer
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    return model
