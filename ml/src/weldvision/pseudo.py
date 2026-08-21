from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PseudoLabelStats:
    accepted: int
    considered: int
    mean_reliability: float


class ScoreTemperature:
    """Post-hoc temperature calibration for detector confidence values."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = float(temperature)

    def transform(self, scores: Tensor) -> Tensor:
        clipped = scores.clamp(1e-6, 1 - 1e-6)
        logits = torch.logit(clipped)
        return torch.sigmoid(logits / self.temperature)

    def fit(self, scores: Tensor, correct: Tensor, max_iterations: int = 100) -> float:
        if scores.numel() == 0 or scores.shape != correct.shape:
            raise ValueError("Scores and correctness labels must have equal non-zero shape")
        log_temperature = torch.tensor(
            self.temperature,
            dtype=scores.dtype,
            device=scores.device,
        ).log().requires_grad_(True)
        optimizer = torch.optim.LBFGS([log_temperature], max_iter=max_iterations)

        def closure() -> Tensor:
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 20.0)
            calibrated = torch.sigmoid(torch.logit(scores.clamp(1e-6, 1 - 1e-6)) / temperature)
            loss = torch.nn.functional.binary_cross_entropy(calibrated, correct.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
        return self.temperature


def select_pseudo_targets(
    predictions: list[dict[str, Tensor]],
    quality: Tensor,
    *,
    base_threshold: float,
    quality_strength: float,
    calibrator: ScoreTemperature | None = None,
    minimum_box_area: float = 16.0,
) -> tuple[list[dict[str, Tensor]], PseudoLabelStats]:
    """Filter teacher predictions using calibrated, quality-conditioned reliability."""
    if len(predictions) != len(quality):
        raise ValueError("One quality score is required for every prediction")
    calibrator = calibrator or ScoreTemperature()
    pseudo_targets: list[dict[str, Tensor]] = []
    reliability_values: list[Tensor] = []
    accepted = 0
    considered = 0

    for prediction, image_quality in zip(predictions, quality, strict=True):
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = calibrator.transform(prediction["scores"])
        considered += int(scores.numel())
        threshold = base_threshold + quality_strength * (1.0 - image_quality.clamp(0, 1))
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        areas = widths * heights
        keep = (scores >= threshold) & (areas >= minimum_box_area)
        reliability = scores[keep] * image_quality
        accepted += int(keep.sum())
        if reliability.numel():
            reliability_values.append(reliability)
        pseudo_targets.append(
            {
                "boxes": boxes[keep].detach(),
                "labels": labels[keep].detach(),
            }
        )

    if reliability_values:
        mean_reliability = float(torch.cat(reliability_values).mean())
    else:
        mean_reliability = 0.0
    return pseudo_targets, PseudoLabelStats(
        accepted=accepted,
        considered=considered,
        mean_reliability=mean_reliability,
    )
