from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

RGB_CHALLENGE_CLASSES = (
    "weld-defect-det",
    "slag inclusion",
    "spatter",
    "undercut",
)


def prepare_rgb_challenge(
    archive_path: str | Path,
    destination: str | Path,
    manifest_directory: str | Path,
) -> dict[str, Any]:
    """Prepare the small CC-BY-4.0 RGB set strictly as an external challenge set."""
    archive = Path(archive_path).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    manifests = Path(manifest_directory).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    root = destination_path / "Final_Dataset_YOLO_Test"
    if not root.is_dir():
        _safe_extract(archive, destination_path)
    images = sorted((root / "images").glob("*.jpg"))
    labels = root / "labels"
    if not images:
        raise ValueError("RGB challenge archive contains no JPEG images")

    class_counts: Counter[str] = Counter()
    capture_groups: Counter[str] = Counter()
    hashes: dict[str, Path] = {}
    exact_duplicates = []
    for image in images:
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}")
        capture_groups[_capture_id(image)] += 1
        digest = _sha256(image)
        if digest in hashes:
            exact_duplicates.append((str(hashes[digest]), str(image)))
        else:
            hashes[digest] = image
        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row at {label}:{line_number}")
            class_id = int(fields[0])
            if class_id not in range(len(RGB_CHALLENGE_CLASSES)):
                raise ValueError(f"Invalid class {class_id} at {label}:{line_number}")
            coordinates = [float(value) for value in fields[1:]]
            if any(value < 0 or value > 1 for value in coordinates):
                raise ValueError(f"Coordinate outside [0,1] at {label}:{line_number}")
            class_counts[RGB_CHALLENGE_CLASSES[class_id]] += 1

    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "target_challenge_unlabeled.txt").write_text(
        "\n".join(str(image.resolve()) for image in images) + "\n",
        encoding="utf-8",
    )
    report = {
        "source": "https://doi.org/10.5281/zenodo.17402020",
        "license": "CC-BY-4.0",
        "archive_sha256": _sha256(archive),
        "images": len(images),
        "unique_capture_groups": len(capture_groups),
        "capture_group_multiplicity": dict(Counter(capture_groups.values())),
        "annotations": sum(class_counts.values()),
        "annotation_counts": dict(class_counts),
        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "allowed_use": (
            "External robustness challenge and unlabeled adaptation smoke test only; "
            "taxonomy is incompatible with LoHi-WELD."
        ),
    }
    (manifests / "rgb_challenge_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _capture_id(path: Path) -> str:
    return path.name.split("_jpg.rf.", 1)[0]


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        destination_root = destination.resolve()
        for member in source.infolist():
            if member.filename.startswith("__MACOSX/") or "/._" in member.filename:
                continue
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            source.extract(member, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
