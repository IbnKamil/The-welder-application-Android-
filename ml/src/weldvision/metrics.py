from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ClassMetrics:
    class_id: int
    average_precision: float
    precision: float
    recall: float
    false_positives_per_image: float
    ground_truth_count: int


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(
            (boxes_a.shape[0], boxes_b.shape[0]),
            device=boxes_a.device,
        )
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(dim=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def evaluate_detections(
    predictions: Sequence[dict[str, Tensor]],
    targets: Sequence[dict[str, Tensor]],
    class_count: int,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> list[ClassMetrics]:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal length")
    metrics = []
    for class_id in range(1, class_count + 1):
        scored_matches: list[tuple[float, bool]] = []
        ground_truth_count = 0
        for prediction, target in zip(predictions, targets, strict=True):
            prediction_mask = (prediction["labels"] == class_id) & (
                prediction["scores"] >= confidence_threshold
            )
            target_mask = target["labels"] == class_id
            boxes = prediction["boxes"][prediction_mask]
            scores = prediction["scores"][prediction_mask]
            truth_boxes = target["boxes"][target_mask]
            ground_truth_count += len(truth_boxes)
            matched_truth: set[int] = set()
            order = scores.argsort(descending=True)
            overlaps = box_iou(boxes, truth_boxes)
            for prediction_index in order.tolist():
                best_iou = 0.0
                best_truth = -1
                for truth_index in range(len(truth_boxes)):
                    if truth_index in matched_truth:
                        continue
                    overlap = float(overlaps[prediction_index, truth_index])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_truth = truth_index
                is_match = best_iou >= iou_threshold
                if is_match:
                    matched_truth.add(best_truth)
                scored_matches.append((float(scores[prediction_index]), is_match))

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        true_positives = np.cumsum([match for _, match in scored_matches], dtype=float)
        false_positives = np.cumsum([not match for _, match in scored_matches], dtype=float)
        precision_curve = true_positives / np.maximum(true_positives + false_positives, 1)
        recall_curve = true_positives / max(ground_truth_count, 1)
        average_precision = _interpolated_ap(precision_curve, recall_curve)
        true_positive_count = int(true_positives[-1]) if len(true_positives) else 0
        false_positive_count = int(false_positives[-1]) if len(false_positives) else 0
        metrics.append(
            ClassMetrics(
                class_id=class_id,
                average_precision=average_precision,
                precision=true_positive_count / max(true_positive_count + false_positive_count, 1),
                recall=true_positive_count / max(ground_truth_count, 1),
                false_positives_per_image=false_positive_count / max(len(targets), 1),
                ground_truth_count=ground_truth_count,
            )
        )
    return metrics


def expected_calibration_error(
    scores: Tensor,
    correct: Tensor,
    bin_count: int = 15,
) -> float:
    if scores.numel() == 0 or scores.shape != correct.shape:
        raise ValueError("Scores and correctness must have equal non-zero shape")
    boundaries = torch.linspace(0, 1, bin_count + 1, device=scores.device)
    error = torch.zeros((), device=scores.device)
    for index in range(bin_count):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (scores > lower) & (scores <= upper)
        if mask.any():
            accuracy = correct[mask].float().mean()
            confidence = scores[mask].mean()
            error += mask.float().mean() * (accuracy - confidence).abs()
    return float(error)


def risk_coverage(
    confidence: Tensor,
    errors: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selective prediction coverage and error risk curves."""
    if confidence.numel() == 0 or confidence.shape != errors.shape:
        raise ValueError("Confidence and errors must have equal non-zero shape")
    order = confidence.argsort(descending=True)
    ordered_errors = errors[order].float()
    cumulative_risk = ordered_errors.cumsum(0) / torch.arange(
        1,
        len(ordered_errors) + 1,
        device=errors.device,
    )
    coverage = torch.arange(
        1,
        len(ordered_errors) + 1,
        device=errors.device,
    ) / len(ordered_errors)
    return coverage.cpu().numpy(), cumulative_risk.cpu().numpy()


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("At least one value is required")
    rng = np.random.default_rng(seed)
    estimates = [
        statistic(rng.choice(array, size=len(array), replace=True))
        for _ in range(samples)
    ]
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def _interpolated_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    if not len(precision):
        return 0.0
    recall_levels = np.linspace(0, 1, 101)
    values = []
    for level in recall_levels:
        candidates = precision[recall >= level]
        values.append(float(candidates.max()) if len(candidates) else 0.0)
    return float(np.mean(values))
