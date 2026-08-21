from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from weldvision.config import ExperimentConfig
from weldvision.model import restore_mobile_detector


class DetectorExportWrapper(nn.Module):
    """Expose stable tensor outputs for one fixed-size mobile input."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = self.detector([image[0]])[0]
        return result["boxes"], result["scores"], result["labels"]


def export_onnx(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 18,
) -> Path:
    """Export FP32 ONNX. INT8 conversion requires target-device calibration afterwards."""
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = restore_mobile_detector(
        checkpoint,
        len(config.data.class_names),
        config.data.image_size,
    )
    model.eval()
    wrapper = DetectorExportWrapper(model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, config.data.image_size, config.data.image_size)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        opset_version=opset,
        do_constant_folding=True,
    )
    metadata = {
        "class_names": config.data.class_names,
        "input_shape": [1, 3, config.data.image_size, config.data.image_size],
        "normalization": "RGB float32 [0, 1]",
        "quantization": "FP32; validate first, then perform representative INT8 calibration",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
