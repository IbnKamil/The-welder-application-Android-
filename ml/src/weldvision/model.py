from __future__ import annotations

import copy

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
