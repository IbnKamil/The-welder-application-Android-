# Colab: разбор нейронной сети (новая сессия)

Не обучайте YOLO заново. Веса уже лежат на Drive:
`/content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt`.

Нужен Runtime → GPU (T4 достаточно). LoHi-архив и manifests для этой команды
не требуются.

## 1. Диск, код, зависимости

```python
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

repo = Path("/content/The-welder-application-Android-")
if repo.exists():
    %cd /content/The-welder-application-Android-
    !git fetch origin cursor/welder-pro-app-33b3
    !git checkout cursor/welder-pro-app-33b3
    !git pull origin cursor/welder-pro-app-33b3
else:
    %cd /content
    !git clone --branch cursor/welder-pro-app-33b3 \
      https://github.com/IbnKamil/The-welder-application-Android-.git

%cd /content/The-welder-application-Android-/ml
%pip install -q -e ".[research]"
```

Проверка весов:

```python
from pathlib import Path

ckpt = Path("/content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt")
print("GPU:", end=" ")
!nvidia-smi -L
print("checkpoint exists:", ckpt.is_file(), ckpt)
print("size_mb:", round(ckpt.stat().st_size / 1e6, 1) if ckpt.is_file() else None)
```

Если `checkpoint exists: False` — не запускайте обучение. Проверьте, что диск
смонтирован тем же Google-аккаунтом, где лежит папка `SvarshikProAI`.

## 2. Собрать HTML-тур

С живыми картами признаков YOLOv8s:

```python
!python -m weldvision.cli explain-network \
  --output /content/drive/MyDrive/SvarshikProAI/network_tour \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt \
  --device 0
```

Если GPU нет, замените `--device 0` на `--device cpu`.

Без своего фото команда возьмёт синтетический валик. Свой кадр шва:

```python
!python -m weldvision.cli explain-network \
  --output /content/drive/MyDrive/SvarshikProAI/network_tour \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt \
  --image /content/drive/MyDrive/SvarshikProAI/your_weld.jpg \
  --device 0
```

Теория шагов: `NEURAL_NETWORK_WALKTHROUGH.md` в этой же папке `ml/`.

## 3. Посмотреть результат

В Colab картинки:

```python
from pathlib import Path
from IPython.display import Image, display

root = Path("/content/drive/MyDrive/SvarshikProAI/network_tour/figures")
for name in (
    "00_demo_input.png",
    "01_letterbox.png",
    "04_convolution.png",
    "05_architecture.png",
    "06_pyramid.png",
    "08_nms_theory.png",
    "13_final.png",
):
    path = root / name
    if path.is_file():
        print(path.name)
        display(Image(str(path), width=720))
```

Скачать весь отчёт и открыть `index.html` на компьютере:

```python
from google.colab import files

!zip -r /tmp/network_tour.zip /content/drive/MyDrive/SvarshikProAI/network_tour
files.download("/tmp/network_tour.zip")
```

Папка уже на Drive: `Мой диск / SvarshikProAI / network_tour / index.html`.
