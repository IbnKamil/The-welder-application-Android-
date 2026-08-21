# Colab: разбор нейронной сети (новая сессия)

Не обучайте YOLO заново. Веса уже лежат на Drive:
`/content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt`.

Нужен Runtime → GPU (T4 достаточно). LoHi-архив и manifests для этой команды
не требуются.

Демонстрационный кадр — **вертикальный шов с цепочкой пор**, который вы выбрали
для разбора сети. Загрузите его в ячейке 2; дальше он сохранится на Drive.

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

## 2. Загрузить демонстрационное фото шва

В диалоге Colab выберите **то же** вертикальное фото валика с порами.

```python
from pathlib import Path
from google.colab import files
from PIL import Image
from IPython.display import display

demo = Path("/content/drive/MyDrive/SvarshikProAI/demo_weld.jpg")
demo.parent.mkdir(parents=True, exist_ok=True)

print("Выберите фото шва для разбора сети")
uploaded = files.upload()
if uploaded:
    name = next(iter(uploaded))
    demo.write_bytes(uploaded[name])
    print("сохранено:", demo, "байт:", demo.stat().st_size)
else:
    print("файл не выбран; если demo_weld.jpg уже на Drive — будет использован он")

assert demo.is_file(), "Нужно загрузить фото шва (demo_weld.jpg)"
image = Image.open(demo).convert("RGB")
print("размер:", image.size)
display(image.resize((min(420, image.width), min(640, image.height))))
```

## 3. Собрать HTML-тур на этом кадре

```python
!python -m weldvision.cli explain-network \
  --output /content/drive/MyDrive/SvarshikProAI/network_tour \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_yolo_s/student_best.pt \
  --image /content/drive/MyDrive/SvarshikProAI/demo_weld.jpg \
  --device 0
```

Если GPU нет, замените `--device 0` на `--device cpu`.

Теория шагов: `NEURAL_NETWORK_WALKTHROUGH.md` в этой же папке `ml/`.

## 4. Посмотреть результат

В Colab картинки:

```python
from pathlib import Path
from IPython.display import Image, display

root = Path("/content/drive/MyDrive/SvarshikProAI/network_tour/figures")
for name in (
    "00_input.png",
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
