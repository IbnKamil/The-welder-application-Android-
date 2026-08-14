from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from weldvision.config import generate_b0_factor_ablations, load_config
from weldvision.data import (
    ManifestRow,
    YoloDetectionDataset,
    exact_duplicate_report,
    group_split,
    letterbox_image_and_boxes,
)
from weldvision.metrics import (
    box_iou,
    detection_calibration_pairs,
    evaluate_detections,
    expected_calibration_error,
)
from weldvision.model import create_mobile_detector, restore_mobile_detector
from weldvision.pseudo import ScoreTemperature, select_pseudo_targets
from weldvision.quality import heuristic_quality


def test_group_split_keeps_physical_weld_together(tmp_path: Path) -> None:
    rows = []
    for group in ("weld-a", "weld-b", "weld-c", "weld-d", "weld-e"):
        for frame in range(3):
            path = tmp_path / f"{group}-{frame}.jpg"
            path.write_bytes(f"{group}-{frame}".encode())
            rows.append(ManifestRow(path, group, "target", frame == 0))

    splits = group_split(rows, seed=7)
    locations = {}
    for split, split_rows in splits.items():
        for row in split_rows:
            locations.setdefault(row.group_id, set()).add(split)
    assert all(len(split_names) == 1 for split_names in locations.values())
    assert exact_duplicate_report(splits) == []


def test_yolo_boxes_are_converted_to_pixels(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (100, 50), "gray").save(image_path)
    (label_dir / "sample.yolo").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{image_path}\n", encoding="utf-8")

    _, target = YoloDetectionDataset(manifest, ["pore"])[0]
    assert target["labels"].tolist() == [1]
    assert target["boxes"][0].tolist() == pytest.approx([40.0, 15.0, 60.0, 35.0])


def test_letterbox_preserves_aspect_ratio_and_transforms_boxes() -> None:
    image = Image.new("RGB", (100, 50), "gray")
    padded, boxes = letterbox_image_and_boxes(
        image,
        [[40.0, 15.0, 60.0, 35.0]],
        200,
    )
    assert padded.size == (200, 200)
    assert boxes[0] == pytest.approx([80.0, 80.0, 120.0, 120.0])


def test_quality_rejects_flat_dark_frame() -> None:
    good = torch.rand(3, 128, 128) * 0.6 + 0.2
    dark = torch.zeros(3, 128, 128)
    assert heuristic_quality(good).score > heuristic_quality(dark).score


def test_quality_conditioned_pseudo_selection() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.75]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.75]),
        },
    ]
    targets, stats = select_pseudo_targets(
        predictions,
        torch.tensor([1.0, 0.0]),
        base_threshold=0.65,
        quality_strength=0.25,
    )
    assert len(targets[0]["boxes"]) == 1
    assert len(targets[1]["boxes"]) == 0
    assert stats.accepted == 1


def test_temperature_calibration_reduces_overconfidence() -> None:
    scores = torch.tensor([0.99, 0.95, 0.90, 0.80])
    correct = torch.tensor([1, 0, 0, 1])
    before = expected_calibration_error(scores, correct, bin_count=4)
    calibrator = ScoreTemperature()
    calibrator.fit(scores, correct)
    after = expected_calibration_error(calibrator.transform(scores), correct, bin_count=4)
    assert after < before


def test_detection_metrics_match_by_class_and_iou() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.8]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }
    assert float(box_iou(prediction["boxes"][:1], target["boxes"])[0, 0]) == 1.0
    metrics = evaluate_detections([prediction], [target], 1)
    assert metrics[0].recall == 1.0
    assert metrics[0].precision == 0.5
    scores, correct = detection_calibration_pairs([prediction], [target])
    assert scores.tolist() == pytest.approx([0.9, 0.8])
    assert correct.tolist() == [True, False]


def test_ap_uses_low_score_predictions_but_operating_metrics_do_not() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.10]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }
    metric = evaluate_detections(
        [prediction],
        [target],
        1,
        confidence_threshold=0.25,
        ap_confidence_floor=0.001,
    )[0]
    assert metric.average_precision == 1.0
    assert metric.precision == 0.0
    assert metric.recall == 0.0


def test_mobile_detector_training_and_inference_contract() -> None:
    model = create_mobile_detector(4, pretrained_backbone=False)
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [
        {
            "boxes": torch.tensor([[30.0, 40.0, 120.0, 150.0]]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.tensor([[50.0, 60.0, 160.0, 180.0]]),
            "labels": torch.tensor([2]),
        },
    ]
    model.train()
    losses = model(images, targets)
    assert losses
    assert all(torch.isfinite(value) for value in losses.values())
    model.eval()
    with torch.no_grad():
        outputs = model(images[:1])
    assert {"boxes", "labels", "scores"} <= outputs[0].keys()

    restored = restore_mobile_detector(
        {"model": model.state_dict()},
        defect_class_count=4,
        image_size=320,
    )
    assert (
        restored.state_dict()["backbone.features.1.0.3.0.weight"].shape
        == model.state_dict()["backbone.features.1.0.3.0.weight"].shape
    )


def test_ablation_generator_changes_one_named_factor(tmp_path: Path) -> None:
    base = Path(__file__).parents[1] / "configs/lohi_baseline.yaml"
    paths = generate_b0_factor_ablations(base, tmp_path / "configs", tmp_path / "results")
    assert len(paths) == 5
    letterbox = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    balanced = yaml.safe_load(paths[3].read_text(encoding="utf-8"))
    assert letterbox["data"]["letterbox"] is True
    assert letterbox["training"]["balanced_sampling"] is False
    assert balanced["data"]["letterbox"] is False
    assert balanced["training"]["balanced_sampling"] is True
    loaded = load_config(paths[0])
    assert loaded.project_root == Path(__file__).parents[1]
    assert loaded.data.source_train == (
        Path(__file__).parents[1] / "data/manifests/source_train.txt"
    )
