from __future__ import annotations

import copy
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
    letterbox: bool
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
    balanced_sampling: bool
    photometric_augmentation: bool
    validation_every: int
    scheduler_eta_min: float
    scheduler: str
    select_best_checkpoint: bool


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
        letterbox=bool(data.get("letterbox", True)),
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
        balanced_sampling=bool(training.get("balanced_sampling", False)),
        photometric_augmentation=bool(training.get("photometric_augmentation", False)),
        validation_every=int(training.get("validation_every", 1)),
        scheduler_eta_min=float(training.get("scheduler_eta_min", 1e-6)),
        scheduler=str(training.get("scheduler", "cosine")),
        select_best_checkpoint=bool(training.get("select_best_checkpoint", True)),
    )
    if training_config.validation_every < 1:
        raise ValueError("training.validation_every must be at least 1")
    if training_config.scheduler not in {"constant", "cosine"}:
        raise ValueError("training.scheduler must be constant or cosine")
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


def generate_b0_factor_ablations(
    base_path: str | Path,
    output_directory: str | Path,
    results_root: str | Path,
) -> list[Path]:
    """Generate one-factor-at-a-time configs from the frozen B0 settings."""
    base = Path(base_path).expanduser().resolve()
    with base.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream)
    experiments = {
        "a1_letterbox_only": {("data", "letterbox"): True},
        "a2_best_checkpoint_only": {("training", "select_best_checkpoint"): True},
        "a3_cosine_scheduler_only": {("training", "scheduler"): "cosine"},
        "a4_balanced_sampler_only": {("training", "balanced_sampling"): True},
        "a5_photometric_only": {("training", "photometric_augmentation"): True},
    }
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_root = Path(results_root).expanduser()
    paths = []
    for name, overrides in experiments.items():
        config = copy.deepcopy(source)
        config["experiment"]["name"] = name
        config["experiment"]["output_dir"] = str(result_root / name)
        config["training"]["validation_every"] = 5
        for keys, value in overrides.items():
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
        path = output / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths
