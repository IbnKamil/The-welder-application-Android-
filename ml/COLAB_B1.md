# Colab: улучшенный baseline B1

Предварительно должен быть выполнен `prepare-lohi`, а
`data/manifests/source_train.txt` должен содержать 1758 изображений.

## 1. Получить актуальный код

```python
%cd /content/The-welder-application-Android-
!git pull origin cursor/welder-pro-app-33b3
%cd /content/The-welder-application-Android-/ml
%pip install -e ".[data]"
```

## 2. Создать Colab-конфигурации

```python
from pathlib import Path
import copy
import yaml

project = Path("/content/The-welder-application-Android-/ml")
source = yaml.safe_load(
    (project / "configs/lohi_baseline_b1.yaml").read_text(encoding="utf-8")
)

smoke = copy.deepcopy(source)
smoke["experiment"]["output_dir"] = (
    "/content/drive/MyDrive/SvarshikProAI/lohi_baseline_b1_smoke"
)
smoke["training"]["source_epochs"] = 1
smoke["training"]["validation_every"] = 1
(project / "configs/lohi_baseline_b1_smoke_colab.yaml").write_text(
    yaml.safe_dump(smoke, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

full = copy.deepcopy(source)
full["experiment"]["output_dir"] = (
    "/content/drive/MyDrive/SvarshikProAI/lohi_baseline_b1"
)
(project / "configs/lohi_baseline_b1_colab.yaml").write_text(
    yaml.safe_dump(full, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
```

## 3. Smoke

```python
!python -m weldvision.cli train \
  --config configs/lohi_baseline_b1_smoke_colab.yaml \
  --device cuda
```

## 4. Полное обучение

```python
!python -m weldvision.cli train \
  --config configs/lohi_baseline_b1_colab.yaml \
  --device cuda
```

После отключения среды:

```python
!python -m weldvision.cli train \
  --config configs/lohi_baseline_b1_colab.yaml \
  --device cuda \
  --resume /content/drive/MyDrive/SvarshikProAI/lohi_baseline_b1/last.pt
```

## 5. Оценка лучшей эпохи

```python
!python -m weldvision.cli evaluate \
  --config configs/lohi_baseline_b1_colab.yaml \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_baseline_b1/student_best.pt \
  --split source_test \
  --device cuda
```

Используется `student_best.pt`, а не `student_final.pt`.
