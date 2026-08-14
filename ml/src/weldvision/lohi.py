from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

LOHI_CLASSES = ("pore", "deposit", "discontinuity", "stain")
EXPECTED_IMAGE_COUNTS = {"high": 1022, "low": 2000}
EXPECTED_ANNOTATIONS = {
    "pore": 3950,
    "deposit": 2935,
    "discontinuity": 7220,
    "stain": 8307,
}


def prepare_lohi(
    archive_path: str | Path,
    destination: str | Path,
    manifest_directory: str | Path,
    *,
    fold: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Safely extract, validate, audit, and create research-safe LoHi manifests."""
    archive = Path(archive_path).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    manifests = Path(manifest_directory).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if fold not in range(5):
        raise ValueError("LoHi-WELD provides folds 0 through 4")

    dataset_root = destination_path / "weld-dataset"
    if not dataset_root.is_dir():
        _safe_extract(archive, destination_path)
    manifests.mkdir(parents=True, exist_ok=True)
    report = audit_lohi(dataset_root)

    official = _official_fold_paths(dataset_root, fold)
    for name, paths in official.items():
        _write_manifest(manifests / f"source_official_{name}.txt", paths)

    grouped_rows = _grouped_source_rows(dataset_root)
    grouped_splits = _split_groups(grouped_rows, seed)
    for name, rows in grouped_splits.items():
        _write_manifest(
            manifests / f"source_{name}.txt",
            [row["image_path"] for row in rows],
        )
    _write_group_csv(manifests / "groups.csv", grouped_rows)

    report["archive_sha256"] = _sha256(archive)
    report["fold"] = fold
    report["group_safe_splits"] = {
        name: {
            "images": len(rows),
            "groups": sorted({row["group_id"] for row in rows}),
        }
        for name, rows in grouped_splits.items()
    }
    report["official_fold_sizes"] = {name: len(paths) for name, paths in official.items()}
    report_path = manifests / "lohi_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def audit_lohi(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    image_counts: dict[str, int] = {}
    annotation_counts: Counter[str] = Counter()
    exact_hashes: dict[str, Path] = {}
    duplicates: list[tuple[str, str]] = []
    invalid_files: list[str] = []

    for resolution in ("high", "low"):
        folder = root / f"{resolution}_resolution_welds"
        images = sorted(folder.glob("*.jpg"))
        image_counts[resolution] = len(images)
        for image in images:
            yolo = image.with_suffix(".yolo")
            json_label = image.with_suffix(".json")
            if not yolo.is_file() or not json_label.is_file():
                invalid_files.append(str(image))
                continue
            class_counts = _validate_yolo(yolo)
            annotation_counts.update(
                {LOHI_CLASSES[class_id]: count for class_id, count in class_counts.items()}
            )
            digest = _sha256(image)
            if digest in exact_hashes:
                duplicates.append((str(exact_hashes[digest]), str(image)))
            else:
                exact_hashes[digest] = image

    if image_counts != EXPECTED_IMAGE_COUNTS:
        raise ValueError(f"Unexpected LoHi image counts: {image_counts}")
    if dict(annotation_counts) != EXPECTED_ANNOTATIONS:
        raise ValueError(f"Unexpected LoHi annotation counts: {dict(annotation_counts)}")
    if invalid_files:
        raise ValueError(f"Missing labels for {len(invalid_files)} images")

    return {
        "image_counts": image_counts,
        "annotation_counts": dict(annotation_counts),
        "exact_duplicate_count": len(duplicates),
        "exact_duplicates": duplicates,
        "sequence_groups": {
            resolution: dict(
                Counter(
                    image.stem.split("_", 1)[0]
                    for image in (
                        root / f"{resolution}_resolution_welds"
                    ).glob("*.jpg")
                )
            )
            for resolution in ("high", "low")
        },
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        destination_root = destination.resolve()
        for member in source.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        source.extractall(destination)


def _official_fold_paths(root: Path, fold: int) -> dict[str, list[Path]]:
    fold_root = root / f"kfold_high/fold{fold}"
    return {
        "train": sorted((fold_root / "train/images").glob("*.jpg")),
        "val": sorted((fold_root / "val/images").glob("*.jpg")),
        "test": sorted((root / "kfold_high/test/images").glob("*.jpg")),
    }


def _grouped_source_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for resolution in ("high", "low"):
        folder = root / f"{resolution}_resolution_welds"
        for image in sorted(folder.glob("*.jpg")):
            sequence = image.stem.split("_", 1)[0]
            rows.append(
                {
                    "image_path": image.resolve(),
                    "group_id": f"lohi-{resolution}-{sequence}",
                    "domain": "source",
                    "labeled": True,
                }
            )
    return rows


def _split_groups(
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_resolution: dict[str, list[str]] = {"high": [], "low": []}
    for group in sorted({row["group_id"] for row in rows}):
        resolution = group.split("-", 2)[1]
        by_resolution[resolution].append(group)
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()
    for offset, (resolution, groups) in enumerate(by_resolution.items()):
        rng = random.Random(seed + offset)
        rng.shuffle(groups)
        test_count = max(1, round(len(groups) * 0.20))
        val_count = max(1, round(len(groups) * 0.20))
        test_groups.update(groups[:test_count])
        val_groups.update(groups[test_count : test_count + val_count])
        resolution_train = groups[test_count + val_count :]
        train_groups.update(resolution_train)
        if not resolution_train:
            raise ValueError(f"Not enough {resolution} groups for a train split")
    assignments = {
        **{group: "train" for group in train_groups},
        **{group: "val" for group in val_groups},
        **{group: "test" for group in test_groups},
    }
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        result[assignments[row["group_id"]]].append(row)
    return result


def _validate_yolo(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id not in range(len(LOHI_CLASSES)):
            raise ValueError(f"Invalid class {class_id} at {path}:{line_number}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"Coordinate outside [0,1] at {path}:{line_number}")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"Non-positive box at {path}:{line_number}")
        counts[class_id] += 1
    return counts


def _write_manifest(path: Path, images: list[Path]) -> None:
    path.write_text(
        "\n".join(str(image.resolve()) for image in images) + ("\n" if images else ""),
        encoding="utf-8",
    )


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image_path", "group_id", "domain", "labeled"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_dataset_for_colab(source: str | Path, destination: str | Path) -> Path:
    """Optional helper for staging an extracted dataset on mounted Google Drive."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    return destination_path
