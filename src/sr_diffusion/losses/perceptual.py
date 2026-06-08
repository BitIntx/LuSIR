from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class FrozenVGGFeatureLoss(nn.Module):
    """ImageNet VGG16 feature loss for optional perceptual supervision."""

    def __init__(
        self,
        resize: int = 256,
        layer_indices: Sequence[int] = (3, 8, 15),
        layer_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if len(layer_indices) != len(layer_weights):
            raise ValueError("perceptual layer_indices and layer_weights must have equal lengths")
        if not layer_indices:
            raise ValueError("perceptual layer_indices cannot be empty")
        self.resize = int(resize)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.layer_weights = tuple(float(weight) for weight in layer_weights)
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[: max(self.layer_indices) + 1]
        self.features.eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenVGGFeatureLoss":
        super().train(False)
        return self

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        image = image.add(1.0).mul(0.5).clamp(0.0, 1.0)
        if self.resize > 0 and image.shape[-2:] != (self.resize, self.resize):
            image = F.interpolate(image, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return (image - self.mean) / self.std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_features = self._prepare(prediction)
        target_features = self._prepare(target)
        loss = prediction.new_zeros(())
        layer_weights = dict(zip(self.layer_indices, self.layer_weights, strict=True))
        for index, layer in enumerate(self.features):
            prediction_features = layer(prediction_features)
            with torch.no_grad():
                target_features = layer(target_features)
            if index in layer_weights:
                prediction_normalized = F.normalize(prediction_features.float(), p=2, dim=1, eps=1e-8)
                target_normalized = F.normalize(target_features.float(), p=2, dim=1, eps=1e-8)
                loss = loss + layer_weights[index] * F.l1_loss(prediction_normalized, target_normalized)
        return loss
