from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source_train: Path
    source_val: Path
    source_test: Path
    target_unlabeled: Path
    target_val: Path
    target_test: Path
    class_names: tuple[str, ...]
    image_size: int
    group_manifest: Path


@dataclass(frozen=True)
class TrainingConfig:
    source_epochs: int
    adaptation_epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    unsupervised_weight: float
    ema_decay: float
    pseudo_threshold: float
    quality_threshold_strength: float
    checkpoint_every: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    project_root: Path
    output_dir: Path
    data: DataConfig
    training: TrainingConfig
    raw: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration and resolve paths from the config directory."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise TypeError("The configuration root must be a mapping")

    root = config_path.parent.parent
    data = raw["data"]
    training = raw["training"]
    experiment = raw["experiment"]

    data_config = DataConfig(
        source_train=_resolve(root, data["source_train"]),
        source_val=_resolve(root, data["source_val"]),
        source_test=_resolve(root, data["source_test"]),
        target_unlabeled=_resolve(root, data["target_unlabeled"]),
        target_val=_resolve(root, data["target_val"]),
        target_test=_resolve(root, data["target_test"]),
        class_names=tuple(data["class_names"]),
        image_size=int(data["image_size"]),
        group_manifest=_resolve(root, data["group_manifest"]),
    )
    training_config = TrainingConfig(
        source_epochs=int(training["source_epochs"]),
        adaptation_epochs=int(training["adaptation_epochs"]),
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        unsupervised_weight=float(training["unsupervised_weight"]),
        ema_decay=float(training["ema_decay"]),
        pseudo_threshold=float(training["pseudo_threshold"]),
        quality_threshold_strength=float(training["quality_threshold_strength"]),
        checkpoint_every=int(training["checkpoint_every"]),
    )
    return ExperimentConfig(
        name=str(experiment["name"]),
        seed=int(experiment["seed"]),
        project_root=root,
        output_dir=_resolve(root, experiment["output_dir"]),
        data=data_config,
        training=training_config,
        raw=raw,
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
