from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class QualityAssessment:
    score: float
    mean_luminance: float
    underexposed_fraction: float
    overexposed_fraction: float
    glare_fraction: float
    sharpness: float


class QualityGate(nn.Module):
    """Trainable multi-label gate for capture defects and an acceptable class."""

    def __init__(self, output_count: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)
        input_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(input_features, output_count)
        self.register_buffer(
            "normalization_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        normalized = (images - self.normalization_mean) / self.normalization_std
        return self.model(normalized)


def heuristic_quality(image: Tensor) -> QualityAssessment:
    """Provide a deterministic bootstrap quality score before a learned gate exists."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a normalized RGB tensor with shape [3, H, W]")
    image = image.float().clamp(0, 1)
    luminance = 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
    mean_luminance = float(luminance.mean())
    underexposed = float((luminance < 0.08).float().mean())
    overexposed = float((luminance > 0.95).float().mean())
    channel_min = image.min(dim=0).values
    glare = float(((luminance > 0.92) & (channel_min > 0.82)).float().mean())

    gray = luminance.unsqueeze(0).unsqueeze(0)
    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    response = F.conv2d(gray, laplacian_kernel, padding=1)
    sharpness = float(response.var().clamp(0, 0.08) / 0.08)

    exposure_score = 1.0 - min(1.0, underexposed + overexposed)
    glare_score = 1.0 - min(1.0, glare * 3.0)
    brightness_score = max(0.0, 1.0 - abs(mean_luminance - 0.5) / 0.5)
    score = (
        0.35 * sharpness
        + 0.30 * exposure_score
        + 0.20 * glare_score
        + 0.15 * brightness_score
    )
    return QualityAssessment(
        score=max(0.0, min(1.0, score)),
        mean_luminance=mean_luminance,
        underexposed_fraction=underexposed,
        overexposed_fraction=overexposed,
        glare_fraction=glare,
        sharpness=sharpness,
    )


def quality_scores(
    images: list[Tensor],
    gate: QualityGate | None = None,
) -> Tensor:
    if gate is None:
        return torch.tensor(
            [heuristic_quality(image).score for image in images],
            device=images[0].device,
        )
    resized = torch.stack(
        [F.interpolate(image[None], size=(224, 224), mode="bilinear")[0] for image in images]
    )
    logits = gate(resized)
    return logits.softmax(dim=1)[:, 0]
