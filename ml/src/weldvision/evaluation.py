from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from weldvision.config import ExperimentConfig
from weldvision.data import YoloDetectionDataset, detection_collate
from weldvision.metrics import evaluate_detections
from weldvision.model import create_mobile_detector, move_targets


@torch.no_grad()
def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    device_name: str | None = None,
) -> dict[str, object]:
    manifest = {
        "source_val": config.data.source_val,
        "val": config.data.target_val,
        "test": config.data.target_test,
    }.get(split)
    if manifest is None:
        raise ValueError("Split must be source_val, val, or test")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = YoloDetectionDataset(manifest, config.data.class_names)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    model = create_mobile_detector(
        len(config.data.class_names),
        pretrained_backbone=False,
        image_size=config.data.image_size,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model") or checkpoint.get("student")
    if state is None:
        raise ValueError("Checkpoint contains neither model nor student weights")
    model.load_state_dict(state)
    model.eval()

    predictions = []
    targets = []
    for images, batch_targets in loader:
        predictions.extend(
            {key: value.cpu() for key, value in result.items()}
            for result in model([image.to(device) for image in images])
        )
        targets.extend(
            {key: value.cpu() for key, value in target.items()}
            for target in move_targets(batch_targets, device)
        )

    thresholds = config.raw.get("evaluation", {}).get("iou_thresholds", [0.5])
    confidence = float(config.raw.get("evaluation", {}).get("confidence_threshold", 0.25))
    reports = {}
    for threshold in thresholds:
        metrics = evaluate_detections(
            predictions,
            targets,
            len(config.data.class_names),
            iou_threshold=float(threshold),
            confidence_threshold=confidence,
        )
        reports[f"iou_{float(threshold):.2f}"] = [
            {
                **asdict(metric),
                "class_name": config.data.class_names[metric.class_id - 1],
            }
            for metric in metrics
        ]
    result = {
        "split": split,
        "images": len(dataset),
        "confidence_threshold": confidence,
        "reports": reports,
    }
    output_path = config.output_dir / f"evaluation_{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
