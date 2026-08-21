from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

SSDLITE = "ssdlite320_mobilenet_v3_large"
FASTER_RCNN = "fasterrcnn_resnet50_fpn"


def create_mobile_detector(
    defect_class_count: int,
    *,
    pretrained_backbone: bool = True,
    image_size: int = 320,
) -> nn.Module:
    """Create a mobile student; detector classes include background internally."""
    if defect_class_count < 1:
        raise ValueError("At least one defect class is required")
    if image_size < 320 or image_size % 32:
        raise ValueError("Image size must be at least 320 and divisible by 32")
    backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=defect_class_count + 1,
        trainable_backbone_layers=6 if pretrained_backbone else None,
    )
    model.transform.fixed_size = (image_size, image_size)
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model


def create_faster_rcnn(
    defect_class_count: int,
    *,
    pretrained_backbone: bool = True,
    image_size: int = 640,
) -> nn.Module:
    """FPN detector; stronger on small pores than SSDLite, BSD-3-Clause weights."""
    if defect_class_count < 1:
        raise ValueError("At least one defect class is required")
    if image_size < 320 or image_size % 32:
        raise ValueError("Image size must be at least 320 and divisible by 32")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        defect_class_count + 1,
    )
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model


def create_detector(
    architecture: str,
    defect_class_count: int,
    *,
    pretrained_backbone: bool = True,
    image_size: int = 320,
) -> nn.Module:
    if architecture == FASTER_RCNN:
        return create_faster_rcnn(
            defect_class_count,
            pretrained_backbone=pretrained_backbone,
            image_size=image_size,
        )
    if architecture == SSDLITE:
        return create_mobile_detector(
            defect_class_count,
            pretrained_backbone=pretrained_backbone,
            image_size=image_size,
        )
    raise ValueError(f"Unsupported detector architecture: {architecture}")


def restore_mobile_detector(
    checkpoint: Mapping[str, Any],
    defect_class_count: int,
    image_size: int,
) -> nn.Module:
    """Rebuild SSDLite or Faster R-CNN from a checkpoint."""
    return restore_detector(checkpoint, defect_class_count, image_size)


def restore_detector(
    checkpoint: Mapping[str, Any],
    defect_class_count: int,
    image_size: int,
) -> nn.Module:
    """Rebuild the detector architecture recorded in a checkpoint."""
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("student")
    if state is None:
        raise ValueError("Checkpoint contains neither model nor student weights")

    specification = checkpoint.get("model_spec", {})
    architecture = str(specification.get("architecture", SSDLITE))
    resolved_size = int(specification.get("image_size", image_size))
    if architecture == FASTER_RCNN:
        pretrained_backbone = bool(specification.get("pretrained_backbone", True))
        model = create_faster_rcnn(
            defect_class_count,
            pretrained_backbone=pretrained_backbone,
            image_size=resolved_size,
        )
        model.load_state_dict(state)
        return model

    if "pretrained_backbone" in specification:
        pretrained_backbone = bool(specification["pretrained_backbone"])
    else:
        pretrained_backbone = _infer_full_tail(state)

    model = create_mobile_detector(
        defect_class_count,
        pretrained_backbone=pretrained_backbone,
        image_size=resolved_size,
    )
    model.load_state_dict(state)
    return model


def mobile_model_spec(image_size: int, pretrained_backbone: bool = True) -> dict[str, Any]:
    return detector_spec(SSDLITE, image_size, pretrained_backbone)


def detector_spec(
    architecture: str,
    image_size: int,
    pretrained_backbone: bool = True,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "architecture": architecture,
        "image_size": image_size,
        "pretrained_backbone": pretrained_backbone,
    }
    if architecture == SSDLITE:
        spec["tail"] = "full" if pretrained_backbone else "reduced"
    return spec


def architecture_from_config(raw: Mapping[str, Any]) -> str:
    return str(raw.get("model", {}).get("student", SSDLITE))


def _infer_full_tail(state: Mapping[str, torch.Tensor]) -> bool:
    probe = state.get("backbone.features.1.0.3.0.weight")
    if probe is None:
        raise ValueError("Unable to infer SSDLite backbone variant from legacy checkpoint")
    return int(probe.shape[0]) >= 160


def create_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameters[name].mul_(decay).add_(student_parameter.detach(), alpha=1 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        teacher_buffers[name].copy_(student_buffer)


def move_targets(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]
