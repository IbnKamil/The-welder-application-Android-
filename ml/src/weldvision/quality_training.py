from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from weldvision.quality import QualityGate

QUALITY_LABELS = (
    "acceptable",
    "blur",
    "underexposed",
    "overexposed",
    "glare",
    "seam_too_small",
)


class QualityLabelDataset(Dataset):
    """Read expert multi-label capture-quality annotations from CSV."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        with self.manifest.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image_path", *QUALITY_LABELS}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Quality CSV must contain: {sorted(required)}")
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.rows[index]
        image_path = Path(row["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = (self.manifest.parent / image_path).resolve()
        with Image.open(image_path) as source:
            image = pil_to_tensor(source.convert("RGB")).float().div(255.0)
        image = F.interpolate(image[None], size=(224, 224), mode="bilinear")[0]
        labels = torch.tensor(
            [float(row[name]) for name in QUALITY_LABELS],
            dtype=torch.float32,
        )
        return image, labels


def train_quality_gate(
    train_manifest: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device_name: str | None = None,
) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = QualityLabelDataset(train_manifest)
    if not len(dataset):
        raise ValueError("Quality manifest is empty")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    gate = QualityGate(len(QUALITY_LABELS)).to(device)
    optimizer = AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        gate.train()
        total_loss = 0.0
        steps = 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                gate(images.to(device)),
                labels.to(device),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        history.append({"epoch": epoch, "loss": total_loss / max(steps, 1)})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": gate.state_dict(),
            "labels": QUALITY_LABELS,
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return output


def load_quality_gate(path: str | Path, device: torch.device) -> QualityGate:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    labels = tuple(checkpoint.get("labels", QUALITY_LABELS))
    if labels != QUALITY_LABELS:
        raise ValueError(f"Unexpected quality labels: {labels}")
    gate = QualityGate(len(labels), pretrained=False).to(device)
    gate.load_state_dict(checkpoint["model"])
    return gate.eval()
