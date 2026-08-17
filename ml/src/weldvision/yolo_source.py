from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from weldvision.config import ExperimentConfig
from weldvision.data import (
    YoloDetectionDataset,
    _find_label_file,
    detection_collate,
    read_image_manifest,
)
from weldvision.metrics import evaluate_detections
from weldvision.model import move_targets


def stage_ultralytics_dataset(
    config: ExperimentConfig,
    destination: str | Path,
) -> Path:
    """Build an Ultralytics dataset that reuses the frozen B0 manifests."""
    root = Path(destination).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    splits = {
        "train": config.data.source_train,
        "val": config.data.source_val,
        "test": config.data.source_test,
    }
    lists: dict[str, list[str]] = {}
    for split, manifest in splits.items():
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        names: list[str] = []
        used: set[str] = set()
        for image_path in read_image_manifest(manifest):
            alias = _alias(image_path)
            if alias in used:
                raise ValueError(f"Duplicate staged name {alias} for {image_path}")
            used.add(alias)
            image_link = image_dir / f"{alias}{image_path.suffix.lower()}"
            _link_or_copy(image_path, image_link)
            label_source = _find_label_file(image_path)
            (label_dir / f"{alias}.txt").write_text(
                label_source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            names.append(str(image_link))
        lists[split] = names
        (root / f"{split}.txt").write_text(
            "\n".join(names) + ("\n" if names else ""),
            encoding="utf-8",
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
                "names": {
                    index: name for index, name in enumerate(config.data.class_names)
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def train_yolo_source(
    config: ExperimentConfig,
    *,
    device_name: str | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Train a research-only Ultralytics YOLO detector on the B0 split.

    Ultralytics is AGPL-3.0. Weights from this command must not be shipped in
    the Play Store app. Use them only for the thesis source-strength comparison.
    """
    from ultralytics import YOLO
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = stage_ultralytics_dataset(config, output_dir / "ultralytics_data")
    device = device_name or ("0" if torch.cuda.is_available() else "cpu")
    variant = str(config.raw.get("model", {}).get("yolo_weights", "yolov8s.pt"))
    if resume:
        model = YOLO(str(resume))
        model.train(resume=True)
    else:
        model = YOLO(variant)
        model.train(
            data=str(data_yaml),
            epochs=config.training.source_epochs,
            imgsz=config.data.image_size,
            batch=config.training.batch_size,
            seed=config.seed,
            workers=config.training.workers,
            project=str(output_dir),
            name="ultralytics",
            exist_ok=True,
            pretrained=True,
            device=device,
            patience=20,
            plots=False,
            verbose=True,
        )
    best = output_dir / "ultralytics" / "weights" / "best.pt"
    last = output_dir / "ultralytics" / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    shutil.copy2(best, output_dir / "student_best.pt")
    if last.is_file():
        shutil.copy2(last, output_dir / "student_final.pt")
    else:
        shutil.copy2(best, output_dir / "student_final.pt")
    return output_dir / "student_best.pt"


def evaluate_yolo_source(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    split: str = "source_test",
    device_name: str | None = None,
) -> dict[str, object]:
    """Score a YOLO checkpoint with the same evaluator as SSDLite B0."""
    manifest = {
        "source_val": config.data.source_val,
        "source_test": config.data.source_test,
        "val": config.data.target_val,
        "test": config.data.target_test,
    }.get(split)
    if manifest is None:
        raise ValueError("Split must be source_val, source_test, val, or test")
    from ultralytics import YOLO
    device = device_name or ("0" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(manifest, config.data.class_names)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    model = YOLO(str(checkpoint_path))
    predictions = []
    targets = []
    image_paths = dataset.images
    for index, (images, batch_targets) in enumerate(loader):
        result = model.predict(
            source=str(image_paths[index]),
            conf=float(config.raw.get("evaluation", {}).get("ap_confidence_floor", 0.001)),
            iou=0.5,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            prediction = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            prediction = {
                "boxes": boxes.xyxy.cpu().float(),
                "scores": boxes.conf.cpu().float(),
                "labels": boxes.cls.cpu().long() + 1,
            }
        predictions.append(prediction)
        targets.extend(
            {key: value.cpu() for key, value in target.items()}
            for target in move_targets(batch_targets, torch.device("cpu"))
        )
        del images

    thresholds = config.raw.get("evaluation", {}).get("iou_thresholds", [0.5])
    confidence = float(config.raw.get("evaluation", {}).get("confidence_threshold", 0.25))
    ap_floor = float(config.raw.get("evaluation", {}).get("ap_confidence_floor", 0.001))
    reports = {}
    for threshold in thresholds:
        metrics = evaluate_detections(
            predictions,
            targets,
            len(config.data.class_names),
            iou_threshold=float(threshold),
            confidence_threshold=confidence,
            ap_confidence_floor=ap_floor,
        )
        reports[f"iou_{float(threshold):.2f}"] = [
            {
                **asdict(metric),
                "class_name": config.data.class_names[metric.class_id - 1],
            }
            for metric in metrics
        ]
    result: dict[str, Any] = {
        "split": split,
        "images": len(dataset),
        "confidence_threshold": confidence,
        "ap_confidence_floor": ap_floor,
        "architecture": "yolov8",
        "license": "AGPL-3.0 research-only",
        "reports": reports,
    }
    output_path = config.output_dir / f"evaluation_{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _alias(image_path: Path) -> str:
    return f"{image_path.parent.name}__{image_path.stem}"


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)
