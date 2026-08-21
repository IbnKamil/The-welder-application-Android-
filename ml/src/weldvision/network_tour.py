from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ("pore", "deposit", "discontinuity", "stain")
CLASS_COLORS = {
    "pore": (220, 50, 50),
    "deposit": (40, 120, 220),
    "discontinuity": (40, 170, 80),
    "stain": (230, 160, 30),
}

# Layer indices of ultralytics.nn.tasks.DetectionModel.model for YOLOv8s.
YOLO_HOOKS: tuple[tuple[int, str, str], ...] = (
    (0, "backbone_stem", "Первое свёртывание: изображение сжимается в 2 раза, появляются 32 карты признаков."),
    (2, "backbone_p2", "Мелкий масштаб: сеть видит короткие края и текстуру чешуи шва."),
    (4, "backbone_p3", "Масштаб P3 (1/8). Здесь живут мелкие объекты — поры."),
    (6, "backbone_p4", "Масштаб P4 (1/16). Средние дефекты: пятна, короткие несплошности."),
    (9, "backbone_sppf", "SPPF: один слой смотрит на шов сразу в нескольких размерах окна."),
    (15, "neck_small", "Шея FPN/PAN, выход для мелких объектов (P3). Сюда смешивают детали и контекст."),
    (18, "neck_medium", "Выход шеи для средних объектов (P4)."),
    (21, "neck_large", "Выход шеи для крупных объектов (P5)."),
)


@dataclass(frozen=True)
class TourStage:
    key: str
    title: str
    explanation: str
    image_name: str | None
    extra: dict[str, Any]


def make_demo_weld_image(size: tuple[int, int] = (640, 240)) -> Image.Image:
    """Synthetic weld strip so the tour runs without LoHi files."""
    width, height = size
    image = Image.new("RGB", (width, height), (28, 28, 30))
    draw = ImageDraw.Draw(image)
    bead_top = height // 3
    bead_bottom = (2 * height) // 3
    draw.rectangle((0, bead_top, width, bead_bottom), fill=(96, 96, 102))
    for index in range(8):
        x = 40 + index * (width // 8)
        draw.arc(
            (x, bead_top + 4, x + 70, bead_bottom - 4),
            start=200,
            end=340,
            fill=(130, 130, 136),
            width=3,
        )
    draw.ellipse((90, height // 2 - 6, 102, height // 2 + 6), fill=(20, 20, 20))
    draw.ellipse((108, height // 2 - 4, 116, height // 2 + 4), fill=(18, 18, 18))
    draw.polygon(((300, bead_top + 8), (340, bead_top - 6), (360, bead_bottom - 10)), fill=(160, 160, 168))
    draw.rectangle((470, bead_top + 12, 560, bead_bottom - 12), fill=(58, 58, 62))
    draw.ellipse((200, bead_top + 10, 248, bead_bottom - 8), fill=(150, 140, 90))
    return image


def letterbox(image: Image.Image, size: int = 640, fill: int = 114) -> tuple[Image.Image, float, tuple[int, int]]:
    """YOLO-style resize that keeps aspect ratio and pads to a square."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, (pad_x, pad_y)


def map_boxes_to_original(
    boxes: np.ndarray,
    scale: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """Undo letterbox: 640-space xyxy → original image pixels."""
    pad_x, pad_y = pad
    mapped = np.asarray(boxes, dtype=np.float32).copy()
    if mapped.size == 0:
        return mapped.reshape(0, 4)
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - pad_x) / scale
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - pad_y) / scale
    return mapped


def box_iou_xyxy(box_a: Sequence[float] | np.ndarray, box_b: Sequence[float] | np.ndarray) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def tensor_to_grid(activation: Any, max_channels: int = 16) -> Image.Image:
    """Turn a feature tensor (BCHW or CHW) into a panel of channel heatmaps."""
    tensor = activation.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW features, got shape {tuple(tensor.shape)}")
    channels = min(max_channels, int(tensor.shape[0]))
    columns = 4
    rows = int(np.ceil(channels / columns))
    maps = []
    for index in range(channels):
        plane = tensor[index].numpy()
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            scaled = np.zeros_like(plane, dtype=np.uint8)
        else:
            low, high = float(finite.min()), float(finite.max())
            norm = (plane - low) / (high - low + 1e-6)
            scaled = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        green = np.clip(scaled.astype(np.int32) * 2 // 3, 0, 255).astype(np.uint8)
        colored = np.stack((scaled, green, 255 - scaled), axis=-1)
        maps.append(Image.fromarray(colored, mode="RGB"))
    cell_w, cell_h = maps[0].size
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (10, 10, 12))
    for index, tile in enumerate(maps):
        grid.paste(tile, ((index % columns) * cell_w, (index // columns) * cell_h))
    return grid.resize((min(960, grid.width * 2), min(720, grid.height * 2)), Image.Resampling.NEAREST)


def mean_heatmap(activation: Any, size: tuple[int, int]) -> Image.Image:
    tensor = activation.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    energy = tensor.abs().mean(dim=0).numpy()
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        norm = np.zeros_like(energy)
    else:
        norm = (energy - finite.min()) / (finite.max() - finite.min() + 1e-6)
    scaled = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    heat = np.stack((scaled, (scaled * 0.4).astype(np.uint8), (255 - scaled)), axis=-1)
    return Image.fromarray(heat, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def overlay_heatmap(base: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    heat = heat.resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base.convert("RGB"), heat.convert("RGB"), alpha)


def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    names: tuple[str, ...] = CLASS_NAMES,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for box, score, label in zip(boxes, scores, labels, strict=False):
        class_id = int(label)
        name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        color = CLASS_COLORS.get(name, (255, 255, 255))
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        caption = f"{name} {float(score):.2f}"
        draw.rectangle((x1, max(0, y1 - 14), x1 + 8 * len(caption), y1), fill=color)
        draw.text((x1 + 2, max(0, y1 - 13)), caption, fill=(0, 0, 0), font=font)
    return canvas


def write_network_tour(
    output_directory: str | Path,
    *,
    image_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device_name: str | None = None,
    image_size: int = 640,
    confidence: float = 0.25,
) -> Path:
    """Build an HTML walkthrough of YOLOv8s with optional live activations."""
    output = Path(output_directory).expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    source = Image.open(image_path).convert("RGB") if image_path else make_demo_weld_image()
    source_name = "00_demo_input.png" if image_path is None else "00_input.png"
    source.save(figures / source_name)

    boxed, scale, pad = letterbox(source, image_size)
    boxed.save(figures / "01_letterbox.png")
    _save_rgb_channels(source, figures / "02_rgb_channels.png")
    _save_pixel_patch(source, figures / "03_pixel_numbers.png")
    _save_conv_demo(source, figures / "04_convolution.png")
    _save_architecture(figures / "05_architecture.png")
    _save_pyramid(boxed, figures / "06_pyramid.png")
    _save_detect_grid(boxed, figures / "07_detect_grid.png")
    nms_iou = _save_nms_theory(figures / "08_nms_theory.png")
    _save_train_vs_infer(figures / "09_train_vs_infer.png")

    stages = [
        TourStage(
            "input",
            "1. Входное изображение",
            "Сеть не видит «шов» словами. Для неё это таблица чисел: высота × ширина × 3 цвета (R, G, B), "
            "каждое значение 0…255. Чем больше пикселей занимает пора, тем легче её найти. "
            "Если checkpoint не задан, ниже показан синтетический валик — тот же пайплайн, что и для кадра LoHi.",
            source_name,
            {"shape": f"{source.size[1]}×{source.size[0]}×3", "dtype": "uint8 RGB"},
        ),
        TourStage(
            "rgb",
            "2. Три цветовых канала",
            "Каждый пиксель — тройка (R, G, B). На grayscale LoHi каналы почти одинаковы; на смартфоне RGB "
            "они расходятся (блик, ржавчина, баланс белого). Детектор диссертации учится на промышленных "
            "серых кадрах, поэтому адаптация к телефону — отдельный научный шаг, а не «ещё эпоха YOLO».",
            "02_rgb_channels.png",
            {"channels": "R, G, B independently stacked as tensor 1×3×H×W"},
        ),
        TourStage(
            "pixels",
            "3. Что сеть читает как числа",
            "Вырезан патч 8×8 из середины валика. В каждой клетке — три числа 0…255. Свёртка на следующем "
            "шаге умножает такие окна на веса фильтра и складывает результат: это и есть «нейрон увидел край».",
            "03_pixel_numbers.png",
            {"patch": "8×8 RGB from weld bead center"},
        ),
        TourStage(
            "letterbox",
            "4. Letterbox: квадрат 640×640",
            "YOLOv8s обучен на квадрате фиксированного размера. Картинку масштабируют, сохраняя пропорции, "
            "и дополняют серыми полями (цвет 114). Растягивание без полей — рецепт SSDLite B0. "
            "На узких кропах LoHi letterbox вредил SSDLite (ablation A1, AP50 0.12 вместо 0.28); "
            "YOLO держит поля лучше за счёт шеи FPN/PAN.",
            "01_letterbox.png",
            {"shape": f"{image_size}×{image_size}×3", "scale": round(scale, 4), "pad": list(pad)},
        ),
        TourStage(
            "normalize",
            "5. Нормализация",
            "Пиксели делят на 255 и получают числа 0…1. Так веса, подобранные на ImageNet/COCO, "
            "оказываются в том же диапазоне, что и вход. Батч для одной картинки: тензор "
            f"1×3×{image_size}×{image_size} типа float32. Дальше идут только умножения и сложения.",
            "01_letterbox.png",
            {"formula": "x' = x / 255", "tensor": f"1×3×{image_size}×{image_size} float32"},
        ),
        TourStage(
            "conv",
            "6. Одна свёртка — атом сети",
            "Фильтр 3×3 скользит по яркости. В каждой позиции: 9 умножений + сумма. "
            "Ядро Собеля здесь подчёркивает вертикальные края чешуи. В YOLOv8 таких ядер тысячи, "
            "их числа (веса) не заданы вручную — их подбирает обучение. На рисунке слева патч, "
            "в центре ядро, справа карта отклика: ярко там, где край совпал с шаблоном.",
            "04_convolution.png",
            {"kernel": "Sobel-X 3×3", "op": "out = sum(patch * kernel) + bias"},
        ),
        TourStage(
            "architecture",
            "7. Три блока: backbone → neck → head",
            "Backbone сжимает картинку и копит признаки (C2f + SPPF). Neck (FPN сверху вниз и PAN снизу вверх) "
            "смешивает мелкие детали с крупным контекстом. Head на трёх сетках (P3/P4/P5) предсказывает "
            "рамку, класс и уверенность. Это не классификатор «дефект/норма»: задача — найти где и какой дефект.",
            "05_architecture.png",
            {
                "backbone": "Conv stride 2, C2f, SPPF",
                "neck": "FPN + PAN",
                "head": "Detect, 4 classes, DFL box",
            },
        ),
        TourStage(
            "pyramid",
            "8. Пирамида масштабов P3 / P4 / P5",
            "Одна и та же сцена кодируется картами разного размера. P3 ≈ 80×80 ячеек на кадре 640 "
            "(ячейка покрывает 8×8 пикселей) — поры. P4 ≈ 40×40 (16×16 пикселей) — пятна и короткие "
            "несплошности. P5 ≈ 20×20 (32×32 пикселя) — крупные наплывы. Ниже вход просто уменьшен, "
            "чтобы показать геометрию сетки; настоящие карты — сотни каналов, не RGB.",
            "06_pyramid.png",
            {"P3": "80×80, stride 8", "P4": "40×40, stride 16", "P5": "20×20, stride 32"},
        ),
        TourStage(
            "detect_grid",
            "9. Голова Detect: ячейка предлагает рамку",
            "На сетке P5 каждая клетка предсказывает смещение центра, ширину/высоту (через DFL) и 4 числа "
            "классов. Фонового класса нет: «здесь ничего» = низкая уверенность. На кадре 640 получается "
            "80²+40²+20² = 8400 ячеек, и с нескольких якорей/предсказаний это десятки тысяч сырых гипотез.",
            "07_detect_grid.png",
            {"cells_p3_p4_p5": "6400 + 1600 + 400 = 8400"},
        ),
        TourStage(
            "nms_theory",
            "10. Порог уверенности и NMS",
            "Сначала отбрасывают гипотезы со score ниже рабочей точки (в диссертации 0.25; для кривой AP "
            "считают с пола 0.001). Затем Non-Maximum Suppression: если две рамки одного класса сильно "
            f"перекрываются (IoU на схеме {nms_iou:.2f}), оставляют ту, у которой score выше. "
            "Иначе один дефект превратился бы в пачку почти одинаковых прямоугольников.",
            "08_nms_theory.png",
            {"operating_confidence": confidence, "example_iou": round(nms_iou, 3), "nms_iou_threshold": 0.5},
        ),
        TourStage(
            "train_vs_infer",
            "11. Обучение vs вывод",
            "При обучении сравнивают предсказание с эталоном и считают потери box / cls / dfl. "
            "Оптимизатор AdamW двигает миллионы весов, чтобы потери падали. "
            "При выводе (этот отчёт) веса заморожены: только прямой проход, без градиентов. "
            "Метрики диссертации (AP50, recall) считаются уже после порога и NMS.",
            "09_train_vs_infer.png",
            {},
        ),
    ]

    live: dict[str, Any] = {}
    if checkpoint is not None:
        live = _run_yolo_hooks(
            Path(checkpoint),
            boxed,
            source,
            scale,
            pad,
            figures,
            device_name=device_name,
            confidence=confidence,
        )
        for index, name, text in YOLO_HOOKS:
            image_name = f"hook_{name}.png"
            heat_name = f"heat_{name}.png"
            if not (figures / image_name).is_file():
                continue
            shape = live.get("shapes", {}).get(name, "")
            stages.append(
                TourStage(
                    name,
                    f"12.{index} Живой слой {name}",
                    text + f" Тензор на выходе: {shape}. Яркие плитки — каналы, где фильтр сильно сработал. "
                    "Тепловая карта ниже — средняя энергия всех каналов, наложенная на letterbox.",
                    image_name,
                    {"heatmap": heat_name, "shape": shape},
                )
            )
        if (figures / "10_raw_grid.png").is_file():
            stages.append(
                TourStage(
                    "decode",
                    "13. Сырые кандидаты после декодера",
                    "Ultralytics `predict` уже делает NMS; при conf=0.001 остаётся много слабых рамок. "
                    "Это ближе всего к «сеть ещё не уверена, но гипотез много». Это не до-NMS тензор головы.",
                    "10_raw_grid.png",
                    {"candidates": live.get("raw_count", 0)},
                )
            )
        if (figures / "11_after_threshold.png").is_file():
            stages.append(
                TourStage(
                    "threshold",
                    "14. Рабочая точка conf ≥ 0.25",
                    f"Оставляем рамки со score ≥ {confidence:.2f}. Более низкий порог поднимает recall "
                    "и плодит ложные срабатывания. AP в отчётах считает с пола 0.001, а precision/recall "
                    "в таблице — при этой рабочей точке.",
                    "11_after_threshold.png",
                    {"kept": live.get("threshold_count", 0)},
                )
            )
        if (figures / "12_after_nms.png").is_file():
            stages.append(
                TourStage(
                    "nms",
                    "15. Итог NMS на letterbox 640×640",
                    "Повторный вызов predict с рабочим порогом: дубли уже сняты, рамки в координатах квадрата.",
                    "12_after_nms.png",
                    {"kept": live.get("nms_count", 0)},
                )
            )
        if (figures / "13_final.png").is_file():
            stages.append(
                TourStage(
                    "final",
                    "16. Перенос рамок на исходное фото",
                    "Координаты с квадрата 640×640: вычитают pad и делят на scale. "
                    "Это и есть ответ детектора: класс + рамка + score в пикселях исходного кадра.",
                    "13_final.png",
                    {"detections": live.get("final", [])},
                )
            )

    stages.extend(_static_theory_stages())
    report = output / "index.html"
    report.write_text(_html_page(stages, live, checkpoint), encoding="utf-8")
    (output / "stages.json").write_text(
        json.dumps(
            [
                {
                    "key": stage.key,
                    "title": stage.title,
                    "explanation": stage.explanation,
                    "image": stage.image_name,
                    "extra": _json_ready(stage.extra),
                }
                for stage in stages
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _run_yolo_hooks(
    checkpoint: Path,
    letterboxed: Image.Image,
    original: Image.Image,
    scale: float,
    pad: tuple[int, int],
    figures: Path,
    *,
    device_name: str | None,
    confidence: float,
) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    device = device_name or ("0" if torch.cuda.is_available() else "cpu")
    activations: dict[str, Any] = {}

    def make_hook(name: str):
        def _hook(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if torch.is_tensor(tensor):
                activations[name] = tensor.detach().cpu()

        return _hook

    sequential = model.model.model
    handles = []
    for index, name, _text in YOLO_HOOKS:
        handles.append(sequential[index].register_forward_hook(make_hook(name)))
    result = model.predict(
        source=np.array(letterboxed),
        imgsz=letterboxed.size[0],
        conf=0.001,
        iou=0.5,
        device=device,
        verbose=False,
    )[0]
    for handle in handles:
        handle.remove()

    shapes = {}
    for _index, name, _text in YOLO_HOOKS:
        tensor = activations.get(name)
        if tensor is None:
            continue
        shapes[name] = "×".join(str(int(dim)) for dim in tensor.shape)
        tensor_to_grid(tensor).save(figures / f"hook_{name}.png")
        heat = mean_heatmap(tensor, letterboxed.size)
        overlay_heatmap(letterboxed, heat).save(figures / f"heat_{name}.png")

    boxes = result.boxes
    raw_count = 0 if boxes is None else int(len(boxes))
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
        draw_boxes(letterboxed, xyxy, scores, labels).save(figures / "10_raw_grid.png")
        keep = scores >= confidence
        draw_boxes(letterboxed, xyxy[keep], scores[keep], labels[keep]).save(
            figures / "11_after_threshold.png"
        )
        nms_path = figures / "12_after_nms.png"
        tight = model.predict(
            source=np.array(letterboxed),
            imgsz=letterboxed.size[0],
            conf=confidence,
            iou=0.5,
            device=device,
            verbose=False,
        )[0].boxes
        if tight is not None and len(tight):
            t_xyxy = tight.xyxy.cpu().numpy()
            t_scores = tight.conf.cpu().numpy()
            t_labels = tight.cls.cpu().numpy().astype(int)
            draw_boxes(letterboxed, t_xyxy, t_scores, t_labels).save(nms_path)
            orig_boxes = map_boxes_to_original(t_xyxy, scale, pad)
            draw_boxes(original, orig_boxes, t_scores, t_labels).save(figures / "13_final.png")
            detections = [
                {
                    "class": CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else int(label),
                    "score": float(score),
                    "box": [float(v) for v in box],
                }
                for box, score, label in zip(orig_boxes, t_scores, t_labels, strict=False)
            ]
            nms_count = len(t_xyxy)
        else:
            original.save(figures / "13_final.png")
            detections = []
            nms_count = 0
        threshold_count = int(keep.sum())
    else:
        letterboxed.save(figures / "10_raw_grid.png")
        letterboxed.save(figures / "11_after_threshold.png")
        letterboxed.save(figures / "12_after_nms.png")
        original.save(figures / "13_final.png")
        detections = []
        threshold_count = 0
        nms_count = 0

    return {
        "shapes": shapes,
        "raw_count": raw_count,
        "threshold_count": threshold_count,
        "nms_count": nms_count,
        "final": detections,
    }


def _static_theory_stages() -> list[TourStage]:
    return [
        TourStage(
            "c2f",
            "Блок C2f (остаточная связь)",
            "Часть каналов идёт напрямую, часть — через маленькие свёртки, затем всё склеивается. "
            "Это позволяет обучать глубокую сеть: градиент не затухает, ранние края не забываются. "
            "YOLOv8s состоит из стопки таких блоков разной ширины.",
            None,
            {"idea": "y = concat(skip, conv(x))"},
        ),
        TourStage(
            "sppf",
            "SPPF — одно окно, несколько масштабов",
            "Карта признаков прогоняется через max-pool 5×5 несколько раз подряд и склеивается. "
            "Крупный наплыв и мелкая пора оказываются в одном контексте, без тяжёлого Spatial Pyramid.",
            None,
            {"pool": "MaxPool 5×5 × 3, then concat"},
        ),
        TourStage(
            "ssdlite",
            "Контроль диссертации: SSDLite-MobileNetV3 (B0)",
            "Другая, более лёгкая сеть того же пайплайна. MobileNetV3 — позвоночник из обратных остаточных "
            "блоков и свёрток depthwise. SSDLite ставит якорные рамки сразу на нескольких картах. "
            "На том же split AP50 был 0.2791, recall пор 0.0559: якоря и слабый FPN плохо держат мелкие поры. "
            "YOLOv8s (AP50 0.571, pore recall 0.493) показывает, что ограничение было в архитектуре, "
            "а не в «невозможности задачи». Веса YOLO — AGPL, в Play Store их класть нельзя.",
            None,
            {"macro_ap50": 0.2791, "pore_recall": 0.0559, "yolo_ap50": 0.571, "yolo_pore_recall": 0.493},
        ),
    ]


def _font():
    try:
        return ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def _captioned(image: Image.Image, caption: str, min_width: int = 160) -> Image.Image:
    img = image.convert("RGB")
    width = max(img.width, min_width)
    if img.width < width:
        pad = Image.new("RGB", (width, img.height), (18, 20, 26))
        pad.paste(img, ((width - img.width) // 2, 0))
        img = pad
    top = 22
    canvas = Image.new("RGB", (img.width, img.height + top), (18, 20, 26))
    canvas.paste(img, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), caption, fill=(230, 226, 214), font=_font())
    return canvas


def _hstack(images: list[Image.Image], gap: int = 10) -> Image.Image:
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), (18, 20, 26))
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.height) // 2))
        x += image.width + gap
    return canvas


def _save_rgb_channels(image: Image.Image, path: Path) -> None:
    array = np.asarray(image.convert("RGB"))
    panels = []
    for index, name in enumerate("RGB"):
        plane = np.zeros_like(array)
        plane[:, :, index] = array[:, :, index]
        panels.append(_captioned(Image.fromarray(plane), f"channel {name}"))
    gray = Image.fromarray(array.mean(axis=2).astype(np.uint8), mode="L").convert("RGB")
    panels.append(_captioned(gray, "mean gray"))
    _hstack(panels).save(path)


def _save_pixel_patch(image: Image.Image, path: Path, patch: int = 8) -> None:
    array = np.asarray(image.convert("RGB"))
    center_y, center_x = array.shape[0] // 2, array.shape[1] // 2
    y0 = max(0, center_y - patch // 2)
    x0 = max(0, center_x - patch // 2)
    crop = array[y0 : y0 + patch, x0 : x0 + patch]
    cell = 54
    canvas = Image.new("RGB", (patch * cell + 16, patch * cell + 36), (16, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((8, 8), "8x8 RGB patch (values 0..255)", fill=(230, 226, 214), font=font)
    for row in range(crop.shape[0]):
        for col in range(crop.shape[1]):
            r, g, b = (int(v) for v in crop[row, col])
            x, y = 8 + col * cell, 28 + row * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=(r, g, b), outline=(40, 40, 44))
            ink = (0, 0, 0) if (r + g + b) > 360 else (240, 240, 240)
            draw.text((x + 3, y + 6), f"{r}", fill=ink, font=font)
            draw.text((x + 3, y + 16), f"{g}", fill=ink, font=font)
            draw.text((x + 3, y + 26), f"{b}", fill=ink, font=font)
    preview = Image.fromarray(crop).resize((patch * 12, patch * 12), Image.Resampling.NEAREST)
    canvas.paste(preview, (canvas.width - preview.width - 8, 8))
    canvas.save(path)


def _convolve2d(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    height, width = plane.shape
    out = np.zeros((height - kh + 1, width - kw + 1), dtype=np.float32)
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            window = plane[row : row + kh, col : col + kw]
            out[row, col] = float(np.sum(window * kernel))
    return out


def _save_conv_demo(image: Image.Image, path: Path) -> None:
    array = np.asarray(image.convert("RGB"))
    gray = array.mean(axis=2).astype(np.float32)
    cy, cx = gray.shape[0] // 2, min(gray.shape[1] - 1, 96)
    y0, x0 = max(0, cy - 40), max(0, cx - 48)
    crop = gray[y0 : y0 + 80, x0 : x0 + 96]
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    activation = _convolve2d(crop, kernel)
    crop_img = Image.fromarray(crop.astype(np.uint8)).resize((288, 240), Image.Resampling.NEAREST)
    act_norm = activation - activation.min()
    act_norm = act_norm / (act_norm.max() + 1e-6)
    act_img = Image.fromarray((act_norm * 255).astype(np.uint8)).resize((288, 240), Image.Resampling.NEAREST)
    kernel_cell = 52
    kernel_img = Image.new("RGB", (3 * kernel_cell, 3 * kernel_cell), (18, 20, 26))
    draw = ImageDraw.Draw(kernel_img)
    font = _font()
    for row in range(3):
        for col in range(3):
            value = int(kernel[row, col])
            color = (40, 140, 70) if value > 0 else ((140, 40, 40) if value < 0 else (70, 70, 76))
            x, y = col * kernel_cell, row * kernel_cell
            draw.rectangle((x + 2, y + 2, x + kernel_cell - 2, y + kernel_cell - 2), fill=color)
            draw.text((x + 18, y + 18), str(value), fill=(255, 255, 255), font=font)
    panel = _hstack(
        [
            _captioned(crop_img.convert("RGB"), "input patch (gray)"),
            _captioned(kernel_img, "kernel Sobel-X"),
            _captioned(act_img.convert("RGB"), "activation map"),
        ]
    )
    panel.save(path)


def _save_architecture(path: Path) -> None:
    width, height = 920, 420
    canvas = Image.new("RGB", (width, height), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    boxes = [
        (30, 150, 160, 270, "Image\n640x640x3", (60, 90, 140)),
        (190, 150, 360, 270, "Backbone\nConv + C2f\nSPPF", (70, 120, 90)),
        (390, 70, 560, 160, "P3 80x80\nsmall / pores", (140, 90, 50)),
        (390, 175, 560, 250, "P4 40x40\nmedium", (140, 110, 50)),
        (390, 265, 560, 350, "P5 20x20\nlarge", (140, 70, 50)),
        (590, 150, 740, 270, "Neck\nFPN + PAN", (90, 80, 140)),
        (770, 150, 900, 270, "Head\nbox+cls+DFL\nNMS", (140, 70, 80)),
    ]
    for x1, y1, x2, y2, label, color in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=color)
        draw.multiline_text((x1 + 12, y1 + 18), label, fill=(245, 245, 240), font=font, spacing=4)
    arrows = [
        ((160, 210), (190, 210)),
        ((360, 210), (390, 115)),
        ((360, 210), (390, 210)),
        ((360, 210), (390, 305)),
        ((560, 115), (590, 190)),
        ((560, 210), (590, 210)),
        ((560, 305), (590, 230)),
        ((740, 210), (770, 210)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        draw.line((x1, y1, x2, y2), fill=(210, 210, 200), width=3)
        draw.polygon([(x2, y2), (x2 - 8, y2 - 5), (x2 - 8, y2 + 5)], fill=(210, 210, 200))
    draw.text((30, 20), "YOLOv8s forward pass  (WeldVision classes: pore, deposit, discontinuity, stain)", fill=(230, 226, 214), font=font)
    draw.text((30, 380), "Training updates millions of weights. Inference only runs this arrow chain once per frame.", fill=(180, 180, 176), font=font)
    canvas.save(path)


def _save_pyramid(letterboxed: Image.Image, path: Path) -> None:
    panels = []
    for size, name, stride in ((80, "P3 ~80x80", 8), (40, "P4 ~40x40", 16), (20, "P5 ~20x20", 32)):
        small = letterboxed.resize((size, size), Image.Resampling.BILINEAR)
        zoomed = small.resize((240, 240), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(zoomed)
        step = max(8, 240 // size)
        for coord in range(0, 241, step):
            draw.line((coord, 0, coord, 240), fill=(255, 220, 80), width=1)
            draw.line((0, coord, 240, coord), fill=(255, 220, 80), width=1)
        panels.append(_captioned(zoomed, f"{name}  stride {stride}"))
    _hstack(panels).save(path)


def _save_detect_grid(letterboxed: Image.Image, path: Path) -> None:
    canvas = letterboxed.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    cell = 32
    for coord in range(0, 641, cell):
        draw.line((coord, 0, coord, 640), fill=(240, 200, 80), width=1)
        draw.line((0, coord, 640, coord), fill=(240, 200, 80), width=1)
    example = (10 * cell, 8 * cell, 14 * cell, 12 * cell)
    draw.rectangle(example, outline=(220, 50, 50), width=3)
    draw.ellipse((12 * cell - 4, 10 * cell - 4, 12 * cell + 4, 10 * cell + 4), fill=(220, 50, 50))
    draw.text((example[0] + 6, example[1] - 14), "cell predicts box + 4 class scores", fill=(255, 230, 180), font=_font())
    canvas.save(path)


def _save_nms_theory(path: Path) -> float:
    canvas = Image.new("RGB", (820, 340), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((20, 12), "Before NMS: two overlapping hypotheses of the same class", fill=(230, 226, 214), font=font)
    draw.text((430, 12), "After NMS: keep the higher score", fill=(230, 226, 214), font=font)
    box_hi = (40, 70, 250, 260)
    box_lo = (120, 110, 330, 300)
    iou = box_iou_xyxy(box_hi, box_lo)
    draw.rectangle(box_hi, outline=(40, 170, 80), width=4)
    draw.rectangle(box_lo, outline=(220, 50, 50), width=4)
    draw.text((48, 76), "discontinuity 0.91", fill=(40, 170, 80), font=font)
    draw.text((128, 116), "discontinuity 0.62", fill=(220, 80, 80), font=font)
    draw.text((40, 312), f"IoU = {iou:.2f}  (> 0.50 => suppress weaker box)", fill=(200, 200, 190), font=font)
    kept = (470, 70, 680, 260)
    draw.rectangle(kept, outline=(40, 170, 80), width=4)
    draw.text((478, 76), "discontinuity 0.91", fill=(40, 170, 80), font=font)
    draw.text((470, 280), "score 0.62 removed", fill=(220, 80, 80), font=font)
    canvas.save(path)
    return iou


def _save_train_vs_infer(path: Path) -> None:
    canvas = Image.new("RGB", (900, 280), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.rounded_rectangle((20, 40, 430, 250), radius=12, fill=(40, 70, 90))
    draw.rounded_rectangle((470, 40, 880, 250), radius=12, fill=(70, 50, 40))
    draw.text((36, 56), "TRAINING", fill=(230, 226, 214), font=font)
    draw.text((486, 56), "INFERENCE (this HTML tour)", fill=(230, 226, 214), font=font)
    draw.multiline_text(
        (36, 90),
        "image + ground-truth boxes\n"
        "forward pass\n"
        "loss = box + cls + dfl\n"
        "backward: gradients\n"
        "AdamW updates weights",
        fill=(220, 230, 235),
        font=font,
        spacing=6,
    )
    draw.multiline_text(
        (486, 90),
        "image only, weights frozen\n"
        "forward pass\n"
        "decode boxes\n"
        "confidence threshold\n"
        "NMS -> detections",
        fill=(235, 220, 210),
        font=font,
        spacing=6,
    )
    canvas.save(path)


def _html_page(stages: list[TourStage], live: dict[str, Any], checkpoint: str | Path | None) -> str:
    toc = "".join(
        f'<li><a href="#{html.escape(stage.key)}">{html.escape(stage.title)}</a></li>'
        for stage in stages
    )
    cards = []
    for stage in stages:
        figure = ""
        if stage.image_name:
            figure = f'<img src="figures/{html.escape(stage.image_name)}" alt="{html.escape(stage.title)}">'
            heat = stage.extra.get("heatmap")
            if heat:
                figure += (
                    '<p class="cap">Сетка каналов выше; ниже — средняя энергия слоя на letterbox.</p>'
                    f'<img src="figures/{html.escape(str(heat))}" alt="heatmap {html.escape(stage.key)}">'
                )
        extra = "<ul>" + "".join(
            f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
            for key, value in stage.extra.items()
            if key != "heatmap"
        ) + "</ul>"
        cards.append(
            f'<section class="card" id="{html.escape(stage.key)}">'
            f"<h2>{html.escape(stage.title)}</h2>"
            f"<p>{html.escape(stage.explanation)}</p>"
            f"{figure}{extra}</section>"
        )
    checkpoint_note = (
        f"<p>Живые активации сняты с <code>{html.escape(str(checkpoint))}</code>.</p>"
        if checkpoint
        else "<p>Режим без checkpoint: показаны вход, числа пикселей, свёртка, архитектура, "
        "пирамида, NMS и схемы. Чтобы увидеть реальные карты признаков YOLOv8s, передайте "
        "веса <code>student_best.pt</code> и при желании своё фото шва.</p>"
    )
    detections = live.get("final") or []
    det_rows = "".join(
        f"<tr><td>{html.escape(str(item['class']))}</td>"
        f"<td>{item['score']:.3f}</td><td>{html.escape(str(item['box']))}</td></tr>"
        for item in detections
    )
    mermaid = """
flowchart TD
    A[Foto RGB HxWx3] --> B[Letterbox 640x640]
    B --> C[x / 255]
    C --> D[Backbone Conv + C2f]
    D --> E[P3 pores]
    D --> F[P4 medium]
    D --> G[SPPF then P5 large]
    E --> H[Neck FPN/PAN]
    F --> H
    G --> H
    H --> I[Head Detect]
    I --> J[Thousands of raw boxes]
    J --> K[Confidence threshold]
    K --> L[NMS]
    L --> M[Boxes on original photo]
"""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Как устроен YOLOv8s в Svarshik WeldVision</title>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
  </script>
  <style>
    body {{ font-family: Georgia, serif; background: #12141a; color: #eee8df; margin: 0; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ font-family: "Segoe UI", sans-serif; }}
    .card {{ background: #1c2028; border: 1px solid #333; border-radius: 12px; padding: 20px; margin: 20px 0; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
    code {{ background: #000; padding: 1px 6px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #444; padding: 6px 8px; text-align: left; }}
    a {{ color: #f0a35e; }}
    .cap {{ color: #aaa; font-size: 0.95rem; }}
    nav ol {{ line-height: 1.6; }}
  </style>
</head>
<body>
<main>
  <h1>Визуальный разбор нейронной сети</h1>
  <p>Рабочая модель диссертации — <strong>YOLOv8s</strong>: детектор рамок дефектов
  <em>pore / deposit / discontinuity / stain</em>. Ниже — прямой проход одного кадра
  (то, что происходит при анализе, без обучения) и схемы внутренних блоков.</p>
  {checkpoint_note}
  <nav class="card"><h2>Содержание</h2><ol>{toc}</ol></nav>
  <div class="card"><pre class="mermaid">{mermaid}</pre></div>
  {"".join(cards)}
  <section class="card">
    <h2>Найденные объекты на этом кадре</h2>
    <table><thead><tr><th>Класс</th><th>Score</th><th>xyxy</th></tr></thead>
    <tbody>{det_rows or "<tr><td colspan='3'>Нет детекций или checkpoint не задан</td></tr>"}</tbody></table>
  </section>
  <p class="cap">Файл сгенерирован командой <code>weldvision explain-network</code>.
  Теория: <code>ml/NEURAL_NETWORK_WALKTHROUGH.md</code>.</p>
</main>
</body>
</html>
"""
