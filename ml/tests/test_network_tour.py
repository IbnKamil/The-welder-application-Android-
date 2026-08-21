from pathlib import Path

import numpy as np
import pytest

from weldvision.network_tour import (
    box_iou_xyxy,
    letterbox,
    make_demo_weld_image,
    map_boxes_to_original,
    tensor_to_grid,
    write_network_tour,
)


def test_demo_weld_is_rgb_and_has_bead() -> None:
    image = make_demo_weld_image()
    assert image.mode == "RGB"
    assert image.size[1] > image.size[0]
    array = np.asarray(image)
    assert array[array.shape[0] // 2, array.shape[1] // 2].sum() > 0


def test_letterbox_and_box_mapping_roundtrip() -> None:
    image = make_demo_weld_image((320, 80))
    boxed, scale, pad = letterbox(image, 640)
    assert boxed.size == (640, 640)
    original = np.array([[40.0, 20.0, 80.0, 60.0]], dtype=np.float32)
    in_square = original.copy()
    in_square[:, [0, 2]] = original[:, [0, 2]] * scale + pad[0]
    in_square[:, [1, 3]] = original[:, [1, 3]] * scale + pad[1]
    restored = map_boxes_to_original(in_square, scale, pad)
    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_iou_of_identical_boxes_is_one() -> None:
    box = (10.0, 10.0, 40.0, 50.0)
    assert box_iou_xyxy(box, box) == 1.0
    assert box_iou_xyxy(box, (100.0, 100.0, 110.0, 110.0)) == 0.0


def test_tensor_grid_has_expected_layout() -> None:
    torch = pytest.importorskip("torch")
    features = torch.linspace(0, 1, 8 * 6 * 6).reshape(1, 8, 6, 6)
    grid = tensor_to_grid(features, max_channels=8)
    assert grid.mode == "RGB"
    assert grid.width > 0 and grid.height > 0


def test_write_network_tour_uses_provided_image(tmp_path: Path) -> None:
    photo = tmp_path / "weld.jpg"
    make_demo_weld_image((200, 320)).save(photo)
    write_network_tour(tmp_path / "tour", image_path=photo)
    saved = tmp_path / "tour" / "figures" / "00_input.png"
    assert saved.is_file()
    from PIL import Image

    assert Image.open(saved).size == (200, 320)


def test_write_network_tour_without_checkpoint(tmp_path: Path) -> None:
    report = write_network_tour(tmp_path / "tour")
    assert report.is_file()
    html = report.read_text(encoding="utf-8")
    assert "Визуальный разбор нейронной сети" in html
    assert "Letterbox" in html
    figures = tmp_path / "tour" / "figures"
    assert (figures / "00_demo_input.png").is_file() or (figures / "00_input.png").is_file()
    for name in (
        "01_letterbox.png",
        "02_rgb_channels.png",
        "03_pixel_numbers.png",
        "04_convolution.png",
        "05_architecture.png",
        "06_pyramid.png",
        "07_detect_grid.png",
        "08_nms_theory.png",
        "09_train_vs_infer.png",
    ):
        assert (figures / name).is_file(), name
    stages = (tmp_path / "tour" / "stages.json").read_text(encoding="utf-8")
    assert "letterbox" in stages
    assert "nms_theory" in stages
