from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass(frozen=True)
class ManifestRow:
    image_path: Path
    group_id: str
    domain: str
    labeled: bool


def read_image_manifest(path: str | Path) -> list[Path]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Image manifest does not exist: {manifest}")
    paths = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image_path = Path(line).expanduser()
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        paths.append(image_path)
    return paths


def read_group_manifest(path: str | Path) -> list[ManifestRow]:
    manifest = Path(path)
    rows: list[ManifestRow] = []
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "group_id", "domain", "labeled"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Group manifest must contain columns: {sorted(required)}")
        for row in reader:
            image_path = Path(row["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    group_id=row["group_id"].strip(),
                    domain=row["domain"].strip(),
                    labeled=row["labeled"].strip().lower() in {"1", "true", "yes"},
                )
            )
    return rows


def group_split(
    rows: Sequence[ManifestRow],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ManifestRow]]:
    """Split by physical weld/session group to prevent frame-level leakage."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a non-empty test fraction")
    groups = sorted({row.group_id for row in rows})
    assignments: dict[str, str] = {}
    for group in groups:
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < train_fraction:
            split = "train"
        elif value < train_fraction + val_fraction:
            split = "val"
        else:
            split = "test"
        assignments[group] = split

    result = {"train": [], "val": [], "test": []}
    for row in rows:
        result[assignments[row.group_id]].append(row)
    validate_group_isolation(result)
    return result


def validate_group_isolation(splits: dict[str, Sequence[ManifestRow]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            previous = seen.setdefault(row.group_id, split)
            if previous != split:
                raise ValueError(
                    f"Data leakage: group {row.group_id!r} occurs in {previous} and {split}"
                )


def exact_duplicate_report(
    splits: dict[str, Sequence[ManifestRow]],
) -> list[tuple[Path, str, Path, str]]:
    """Report byte-identical images crossing split boundaries."""
    seen: dict[str, tuple[Path, str]] = {}
    duplicates = []
    for split, rows in splits.items():
        for row in rows:
            digest = _sha256(row.image_path)
            if digest in seen and seen[digest][1] != split:
                other_path, other_split = seen[digest]
                duplicates.append((other_path, other_split, row.image_path, split))
            else:
                seen[digest] = (row.image_path, split)
    return duplicates


class YoloDetectionDataset(Dataset):
    """Read YOLO normalized boxes and expose torchvision detection targets."""

    def __init__(
        self,
        manifest: str | Path,
        class_names: Sequence[str],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.images = read_image_manifest(manifest)
        self.class_names = tuple(class_names)
        self.transform = transform or image_to_tensor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_path = self.images[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)

        boxes, labels = _read_yolo_labels(
            _find_label_file(image_path),
            width,
            height,
            len(self.class_names),
        )
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return tensor, target


class UnlabeledImageDataset(Dataset):
    """Return weak/strong photometric views of target-domain smartphone images."""

    def __init__(self, manifest: str | Path, seed: int = 42) -> None:
        self.images = read_image_manifest(manifest)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        with Image.open(self.images[index]) as source:
            image = source.convert("RGB")
            weak = image_to_tensor(image)
            strong = image_to_tensor(strong_photometric_augmentation(image, self.seed + index))
        return weak, strong, str(self.images[index])


def image_to_tensor(image: Image.Image) -> Tensor:
    return pil_to_tensor(image).float().div(255.0)


def strong_photometric_augmentation(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    result = ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 1.45))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.65, 1.45))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.4, 1.4))
    if rng.random() < 0.35:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.6)))
    return result


def detection_collate(
    batch: Iterable[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def unlabeled_collate(
    batch: Iterable[tuple[Tensor, Tensor, str]],
) -> tuple[list[Tensor], list[Tensor], list[str]]:
    weak, strong, paths = zip(*batch, strict=True)
    return list(weak), list(strong), list(paths)


def _find_label_file(image_path: Path) -> Path:
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    if "images" in parts:
        image_index = len(parts) - 1 - parts[::-1].index("images")
        replaced = parts.copy()
        replaced[image_index] = "labels"
        candidates.insert(0, Path(*replaced).with_suffix(".txt"))
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No YOLO label file found for {image_path}")


def _read_yolo_labels(
    path: Path,
    width: int,
    height: int,
    class_count: int,
) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO row in {path}:{line_number}")
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"Class {class_id} is outside [0, {class_count}) in {path}")
        center_x, center_y, box_width, box_height = map(float, values[1:])
        x1 = max(0.0, (center_x - box_width / 2) * width)
        y1 = max(0.0, (center_y - box_height / 2) * height)
        x2 = min(float(width), (center_x + box_width / 2) * width)
        y2 = min(float(height), (center_y + box_height / 2) * height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(class_id + 1)  # zero is reserved for background
    return boxes, labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
