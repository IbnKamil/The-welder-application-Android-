from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


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


def restore_mobile_detector(
    checkpoint: Mapping[str, Any],
    defect_class_count: int,
    image_size: int,
) -> nn.Module:
    """Rebuild the exact SSDLite tail variant used by an existing checkpoint."""
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("student")
    if state is None:
        raise ValueError("Checkpoint contains neither model nor student weights")

    specification = checkpoint.get("model_spec", {})
    if "pretrained_backbone" in specification:
        pretrained_backbone = bool(specification["pretrained_backbone"])
    else:
        pretrained_backbone = _infer_full_tail(state)

    model = create_mobile_detector(
        defect_class_count,
        pretrained_backbone=pretrained_backbone,
        image_size=int(specification.get("image_size", image_size)),
    )
    model.load_state_dict(state)
    return model


def mobile_model_spec(image_size: int, pretrained_backbone: bool = True) -> dict[str, Any]:
    return {
        "architecture": "ssdlite320_mobilenet_v3_large",
        "image_size": image_size,
        "pretrained_backbone": pretrained_backbone,
        "tail": "full" if pretrained_backbone else "reduced",
    }


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
