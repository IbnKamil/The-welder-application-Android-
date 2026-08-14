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
from torch.utils.data import DataLoader

from weldvision.config import ExperimentConfig
from weldvision.data import (
    UnlabeledImageDataset,
    YoloDetectionDataset,
    detection_collate,
    unlabeled_collate,
)
from weldvision.metrics import detection_calibration_pairs
from weldvision.model import (
    create_ema_teacher,
    create_mobile_detector,
    move_targets,
    update_ema_teacher,
)
from weldvision.pseudo import ScoreTemperature, select_pseudo_targets
from weldvision.quality import QualityGate, quality_scores
from weldvision.quality_training import load_quality_gate


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
    )
    target_dataset = UnlabeledImageDataset(config.data.target_unlabeled, config.seed)
    source_loader = DataLoader(
        source_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        collate_fn=detection_collate,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        collate_fn=unlabeled_collate,
    )

    student = create_mobile_detector(
        len(config.data.class_names),
        image_size=config.data.image_size,
    ).to(device)
    teacher = create_ema_teacher(student).to(device)
    optimizer = AdamW(
        student.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_phase = "source"
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_phase = checkpoint["phase"]
        start_epoch = int(checkpoint["epoch"]) + 1

    calibrator = ScoreTemperature(
        float(config.raw.get("calibration", {}).get("initial_temperature", 1.0))
    )
    quality_gate = _optional_quality_gate(config, device)
    log_path = output_dir / "metrics.jsonl"

    if start_phase == "source":
        for epoch in range(start_epoch, config.training.source_epochs):
            metrics = train_source_epoch(student, source_loader, optimizer, device)
            update_ema_teacher(teacher, student, 0.0)
            _write_metrics(log_path, "source", epoch, metrics)
            _save_checkpoint(output_dir, "source", epoch, student, teacher, optimizer, config)
        start_epoch = 0

    _fit_teacher_calibration(config, teacher, calibrator, device)

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
        _write_metrics(log_path, "adaptation", epoch, metrics)
        _save_checkpoint(output_dir, "adaptation", epoch, student, teacher, optimizer, config)

    final_path = output_dir / "student_final.pt"
    torch.save(
        {
            "model": student.state_dict(),
            "class_names": config.data.class_names,
            "config": config.raw,
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
            "config": config.raw,
        },
        output_dir / "last.pt",
    )


def _write_metrics(path: Path, phase: str, epoch: int, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "epoch": epoch, **metrics}) + "\n")


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
    dataset = YoloDetectionDataset(manifest, config.data.class_names)
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
