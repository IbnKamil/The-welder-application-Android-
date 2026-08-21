# Colab: сильный source-детектор на split B0

Цель: получить модель, которая **реально находит дефекты** на том же
sequence-safe LoHi split, что и замороженный SSDLite B0. Адаптацию к смартфону
не запускаем, пока не выполнен порог ниже.

Порог, после которого имеет смысл идти на RGB:

- macro AP50 ≥ 0.50, и
- recall `pore` ≥ 0.25, и
- recall `discontinuity` не хуже B0 (~0.55).

YOLOv8s — **только для диссертации** (лицензия AGPL-3.0). Эти веса **нельзя**
класть в приложение Google Play. Для Play остаётся SSDLite / Faster R-CNN (BSD).

## 1. Код и зависимость

Новая сессия Colab (репозитория ещё нет):

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content
!git clone --branch cursor/welder-pro-app-33b3 \
  https://github.com/IbnKamil/The-welder-application-Android-.git
%cd The-welder-application-Android-/ml
%pip install -q -e ".[data,research]"
```

Если папка `/content/The-welder-application-Android-` уже есть:

```python
%cd /content/The-welder-application-Android-
!git pull origin cursor/welder-pro-app-33b3
%cd ml
%pip install -q -e ".[data,research]"
```

Проверьте, что `data/manifests/source_train.txt` содержит **1758** строк.

## 2. Конфиг на Drive

```python
from pathlib import Path
import copy
import yaml

project = Path("/content/The-welder-application-Android-/ml")
source = yaml.safe_load((project / "configs/lohi_yolo_s.yaml").read_text())
source["experiment"]["output_dir"] = "/content/drive/MyDrive/SvarshikProAI/lohi_yolo_s"
(project / "configs/lohi_yolo_s_colab.yaml").write_text(
    yaml.safe_dump(source, allow_unicode=True, sort_keys=False)
)
print("config ready")
```

## 3. Обучение (~1–2 ч на T4, 50 эпох)

```python
!python -m weldvision.cli train-yolo \
  --config configs/lohi_yolo_s_colab.yaml \
  --device 0
```

Если сессия оборвалась:

```python
!python -m weldvision.cli train-yolo \
  --config configs/lohi_yolo_s_colab.yaml \
  --device 0 \
  --resume /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/ultralytics/weights/last.pt
```

Дождитесь `student_best.pt`.

## 4. Оценка тем же evaluator, что у B0

```python
!python -m weldvision.cli evaluate-yolo \
  --config configs/lohi_yolo_s_colab.yaml \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt \
  --split source_test \
  --device 0
```

Сравните с B0: AP50 **0.2791**, recall50 **0.3088**, pore recall **0.0559**.

Пришлите JSON. Если порог не взят, следующий шаг — тот же конфиг с
`image_size: 1280` (лучше для мелких пор, дольше).

## Запасной лицензионно чистый путь

Faster R-CNN ResNet50-FPN, тот же split, `weldvision train` (не train-yolo).
На T4 это существенно медленнее YOLO (batch 2). Запускайте, только если YOLO
нельзя использовать или нужен кандидат в приложение.

```python
!python -m weldvision.cli train \
  --config configs/lohi_fasterrcnn_s0.yaml \
  --device cuda
```

## Разбор нейронной сети по шагам

Теория: `NEURAL_NETWORK_WALKTHROUGH.md`. Картинки прямого прохода:

```python
!python -m weldvision.cli explain-network \
  --output /content/drive/MyDrive/SvarshikProAI/network_tour \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt \
  --device 0
```

Скачайте папку `network_tour` и откройте `index.html`. Демонстрационный кадр
загружается ячейкой в [`COLAB_NETWORK_TOUR.md`](COLAB_NETWORK_TOUR.md).
