from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from weldvision.config import ExperimentConfig
from weldvision.data import (
    UnlabeledImageDataset,
    YoloDetectionDataset,
    balanced_sample_weights,
    detection_collate,
    unlabeled_collate,
)
from weldvision.metrics import detection_calibration_pairs, evaluate_detections
from weldvision.model import (
    architecture_from_config,
    create_detector,
    create_ema_teacher,
    detector_spec,
    move_targets,
    update_ema_teacher,
)
from weldvision.pseudo import ScoreTemperature, select_pseudo_targets
from weldvision.quality import QualityGate, quality_scores
from weldvision.quality_training import load_quality_gate


def _model_spec(config: ExperimentConfig) -> dict[str, Any]:
    pretrained = bool(config.raw.get("model", {}).get("pretrained_backbone", True))
    return detector_spec(
        architecture_from_config(config.raw),
        config.data.image_size,
        pretrained,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sum_losses(losses: dict[str, Tensor]) -> Tensor:
    if not losses:
        raise ValueError("Detector returned no training losses")
    return sum(losses.values())


def train_source_epoch(
    model: nn.Module,
    loader: Iterable[tuple[list[Tensor], list[dict[str, Tensor]]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        loss = sum_losses(model(images, targets))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        steps += 1
    return {"source_loss": total_loss / max(steps, 1)}


def adapt_epoch(
    student: nn.Module,
    teacher: nn.Module,
    source_loader: Iterable[tuple[list[Tensor], list[dict[str, Tensor]]]],
    target_loader: Iterable[tuple[list[Tensor], list[Tensor], list[str]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    calibrator: ScoreTemperature,
    base_threshold: float,
    quality_strength: float,
    unsupervised_weight: float,
    ema_decay: float,
    quality_gate: QualityGate | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    source_iterator = _repeat(source_loader)
    total_source = 0.0
    total_target = 0.0
    accepted = 0
    considered = 0
    steps = 0

    for weak_images, strong_images, _ in target_loader:
        source_images, source_targets = next(source_iterator)
        source_images = [image.to(device) for image in source_images]
        source_targets = move_targets(source_targets, device)
        weak_images = [image.to(device) for image in weak_images]
        strong_images = [image.to(device) for image in strong_images]

        with torch.no_grad():
            predictions = teacher(weak_images)
            quality = quality_scores(weak_images, quality_gate)
            pseudo_targets, stats = select_pseudo_targets(
                predictions,
                quality,
                base_threshold=base_threshold,
                quality_strength=quality_strength,
                calibrator=calibrator,
            )

        optimizer.zero_grad(set_to_none=True)
        source_loss = sum_losses(student(source_images, source_targets))
        target_loss = torch.zeros((), device=device)
        if stats.accepted:
            target_loss = sum_losses(student(strong_images, pseudo_targets))
        loss = source_loss + unsupervised_weight * stats.mean_reliability * target_loss
        loss.backward()
        optimizer.step()
        update_ema_teacher(teacher, student, ema_decay)

        total_source += float(source_loss.detach())
        total_target += float(target_loss.detach())
        accepted += stats.accepted
        considered += stats.considered
        steps += 1

    return {
        "source_loss": total_source / max(steps, 1),
        "target_loss": total_target / max(steps, 1),
        "pseudo_acceptance": accepted / max(considered, 1),
        "pseudo_accepted": float(accepted),
    }


def run_training(
    config: ExperimentConfig,
    *,
    device_name: str | None = None,
    resume: str | Path | None = None,
) -> Path:
    seed_everything(config.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = YoloDetectionDataset(
        config.data.source_train,
        config.data.class_names,
        image_size=config.data.image_size if config.data.letterbox else None,
        photometric_augmentation=config.training.photometric_augmentation,
    )
    source_val_dataset = YoloDetectionDataset(
        config.data.source_val,
        config.data.class_names,
        image_size=config.data.image_size if config.data.letterbox else None,
    )
    sampler = (
        WeightedRandomSampler(
            balanced_sample_weights(source_dataset),
            num_samples=len(source_dataset),
            replacement=True,
        )
        if config.training.balanced_sampling
        else None
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    student = create_detector(
        architecture_from_config(config.raw),
        len(config.data.class_names),
        pretrained_backbone=bool(
            config.raw.get("model", {}).get("pretrained_backbone", True)
        ),
        image_size=config.data.image_size,
    ).to(device)
    teacher = create_ema_teacher(student).to(device)
    optimizer = AdamW(
        student.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler: LRScheduler
    if config.training.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(
                config.training.source_epochs + config.training.adaptation_epochs,
                1,
            ),
            eta_min=config.training.scheduler_eta_min,
        )
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    start_phase = "source"
    start_epoch = 0
    best_validation_ap = -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_phase = checkpoint["phase"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_ap = float(checkpoint.get("best_validation_ap", -1.0))

    calibrator = ScoreTemperature(
        float(config.raw.get("calibration", {}).get("initial_temperature", 1.0))
    )
    quality_gate = _optional_quality_gate(config, device)
    log_path = output_dir / "metrics.jsonl"

    if start_phase == "source":
        for epoch in range(start_epoch, config.training.source_epochs):
            metrics = train_source_epoch(student, source_loader, optimizer, device)
            update_ema_teacher(teacher, student, 0.0)
            scheduler.step()
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            if (epoch + 1) % config.training.validation_every == 0:
                validation = validate_detector(
                    student,
                    source_val_loader,
                    device,
                    len(config.data.class_names),
                )
                metrics.update(validation)
                if validation["validation_macro_ap50"] > best_validation_ap:
                    best_validation_ap = validation["validation_macro_ap50"]
                    _save_best_model(
                        output_dir,
                        epoch,
                        student,
                        config,
                        validation,
                    )
            _write_metrics(log_path, "source", epoch, metrics)
            _save_checkpoint(
                output_dir,
                "source",
                epoch,
                student,
                teacher,
                optimizer,
                scheduler,
                best_validation_ap,
                config,
            )
        best_path = output_dir / "student_best.pt"
        if config.training.select_best_checkpoint and best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            student.load_state_dict(best_checkpoint["model"])
            update_ema_teacher(teacher, student, 0.0)
        start_epoch = 0

    _fit_teacher_calibration(config, teacher, calibrator, device)

    if config.training.adaptation_epochs <= 0:
        final_path = output_dir / "student_final.pt"
        torch.save(
            {
                "model": student.state_dict(),
                "class_names": config.data.class_names,
                "config": config.raw,
                "model_spec": _model_spec(config),
            },
            final_path,
        )
        return final_path

    target_dataset = UnlabeledImageDataset(
        config.data.target_unlabeled,
        config.seed,
        image_size=config.data.image_size if config.data.letterbox else None,
    )
    if not len(target_dataset):
        raise ValueError("Target unlabeled manifest is empty; adaptation cannot start")
    target_loader = DataLoader(
        target_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        collate_fn=unlabeled_collate,
    )

    for epoch in range(start_epoch, config.training.adaptation_epochs):
        metrics = adapt_epoch(
            student,
            teacher,
            source_loader,
            target_loader,
            optimizer,
            device,
            calibrator=calibrator,
            base_threshold=config.training.pseudo_threshold,
            quality_strength=config.training.quality_threshold_strength,
            unsupervised_weight=config.training.unsupervised_weight,
            ema_decay=config.training.ema_decay,
            quality_gate=quality_gate,
        )
        scheduler.step()
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        _write_metrics(log_path, "adaptation", epoch, metrics)
        _save_checkpoint(
            output_dir,
            "adaptation",
            epoch,
            student,
            teacher,
            optimizer,
            scheduler,
            best_validation_ap,
            config,
        )

    final_path = output_dir / "student_final.pt"
    torch.save(
        {
            "model": student.state_dict(),
            "class_names": config.data.class_names,
            "config": config.raw,
            "model_spec": _model_spec(config),
        },
        final_path,
    )
    return final_path


def _save_checkpoint(
    output_dir: Path,
    phase: str,
    epoch: int,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    best_validation_ap: float,
    config: ExperimentConfig,
) -> None:
    if (epoch + 1) % config.training.checkpoint_every:
        return
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_ap": best_validation_ap,
            "config": config.raw,
            "model_spec": _model_spec(config),
        },
        output_dir / "last.pt",
    )


@torch.no_grad()
def validate_detector(
    model: nn.Module,
    loader: Iterable[tuple[list[Tensor], list[dict[str, Tensor]]]],
    device: torch.device,
    class_count: int,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in loader:
        predictions.extend(
            {key: value.cpu() for key, value in prediction.items()}
            for prediction in model([image.to(device) for image in images])
        )
        targets.extend(
            {key: value.cpu() for key, value in target.items()}
            for target in batch_targets
        )
    class_metrics = evaluate_detections(
        predictions,
        targets,
        class_count,
        iou_threshold=0.5,
        confidence_threshold=0.05,
    )
    macro_ap = sum(metric.average_precision for metric in class_metrics) / max(
        len(class_metrics),
        1,
    )
    macro_recall = sum(metric.recall for metric in class_metrics) / max(
        len(class_metrics),
        1,
    )
    return {
        "validation_macro_ap50": macro_ap,
        "validation_macro_recall50": macro_recall,
    }


def _save_best_model(
    output_dir: Path,
    epoch: int,
    student: nn.Module,
    config: ExperimentConfig,
    validation: dict[str, float],
) -> None:
    torch.save(
        {
            "model": student.state_dict(),
            "class_names": config.data.class_names,
            "config": config.raw,
            "model_spec": _model_spec(config),
            "epoch": epoch,
            **validation,
        },
        output_dir / "student_best.pt",
    )


def _write_metrics(path: Path, phase: str, epoch: int, metrics: dict[str, Any]) -> None:
    payload = {"phase": phase, "epoch": epoch, **metrics}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def _repeat(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        yield from loader


def _fit_teacher_calibration(
    config: ExperimentConfig,
    teacher: nn.Module,
    calibrator: ScoreTemperature,
    device: torch.device,
) -> None:
    manifest = config.data.target_val
    if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return
    dataset = YoloDetectionDataset(
        manifest,
        config.data.class_names,
        image_size=config.data.image_size if config.data.letterbox else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    teacher.eval()
    score_batches = []
    correct_batches = []
    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        device_targets = move_targets(targets, device)
        with torch.no_grad():
            scores, correct = detection_calibration_pairs(
                teacher(device_images),
                device_targets,
            )
        if scores.numel():
            score_batches.append(scores)
            correct_batches.append(correct)
    if score_batches:
        calibrator.fit(torch.cat(score_batches), torch.cat(correct_batches))
        (config.output_dir / "calibration.json").write_text(
            json.dumps({"temperature": calibrator.temperature}, indent=2),
            encoding="utf-8",
        )


def _optional_quality_gate(
    config: ExperimentConfig,
    device: torch.device,
) -> QualityGate | None:
    raw_path = config.raw.get("model", {}).get("quality_checkpoint")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config.project_root / path
    return load_quality_gate(path, device) if path.is_file() else None
