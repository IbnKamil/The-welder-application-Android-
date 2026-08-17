# Svarshik WeldVision

Исследовательская программа для магистерской диссертации:

> «Метод адаптации нейросетевой модели обнаружения визуальных дефектов сварных
> соединений к RGB-изображениям мобильных устройств для задач контроля в
> авиационном производстве».

## Статус

Репозиторий содержит воспроизводимый код эксперимента, но не обученные веса.
Модель нельзя считать валидированной до обучения на экспертно размеченных данных,
проверки на независимых физических швах и испытаний на смартфонах.

Обычная RGB-камера видит только поверхностные признаки. Система не может исключить
внутренний непровар, скрытые включения и внутреннюю пористость и не заменяет
аттестованный визуальный, ультразвуковой или радиографический контроль.

## Реализованная гипотеза

Программа проверяет, уменьшает ли quality-aware pseudo-label selection отрицательный
перенос из промышленного grayscale-домена LoHi-WELD в smartphone RGB:

```text
labeled industrial images ──→ mobile student ──→ EMA teacher
                                      ↑               │
unlabeled smartphone RGB ──→ quality score ──→ calibrated pseudo-labels
```

Порог принятия псевдоразметки повышается при плохом качестве кадра:

```text
threshold_i = base_threshold + quality_strength × (1 - quality_i)
```

Надёжность принятого объекта определяется произведением калиброванной уверенности
teacher и оценки качества снимка. Это проверяемая гипотеза, а не заранее заявленное
научное преимущество.

## Компоненты

- leakage-safe split по `weld_id`/серии, а не по отдельным кадрам;
- аудит точных дубликатов между train/validation/test;
- загрузка YOLO bbox-разметки LoHi-WELD и собственного RGB-набора;
- MobileNetV3 SSDLite student на torchvision (BSD-3-Clause);
- EMA teacher и weak/strong target views;
- bootstrap quality score и интерфейс обучаемого MobileNetV3 quality gate;
- temperature calibration уверенности;
- quality-conditioned фильтрация псевдоразметки;
- source baseline и адаптационная стадия;
- AP, precision, recall, FP/image, ECE, risk-coverage и bootstrap CI;
- checkpoint/resume, JSONL-журнал эксперимента;
- FP32 ONNX export после валидации.

## Установка

В Google Colab сначала включите GPU, затем:

```bash
git clone \
  --branch cursor/welder-pro-app-33b3 \
  https://github.com/IbnKamil/The-welder-application-Android-.git
cd The-welder-application-Android-/ml
pip install -e ".[dev,export,data,research]"
```

Проверка:

```bash
pytest
weldvision --help
```

## Данные

Датасеты не включены в Git из-за размера и условий использования.

Рекомендуемые источники:

1. [LoHi-WELD](https://github.com/SylvioBlock/LoHi-Weld) — размеченный source.
2. [Weld bead images and masks](https://doi.org/10.6084/m9.figshare.25904014)
   — предварительное выделение шва.
3. [INWELD](https://doi.org/10.3390/app152212033) — smartphone RGB/ROI после
   подтверждения прав на данные.
4. Собственный target-набор смартфонов с экспертной разметкой.

Перед коммерческим использованием необходимо зафиксировать версии лицензий и
атрибуцию каждого набора. Рентгеновские изображения нельзя смешивать с RGB как
обычные обучающие примеры: это физически другая модальность.

### Автоматическая подготовка LoHi-WELD

Установите data-зависимость и скачайте официальный архив:

```bash
pip install -e ".[data]"
python -m gdown \
  "https://drive.google.com/uc?id=1pXeEnREfV_MYcL5MY2vkd9njBm_blPUK" \
  -O data/raw/lohi-weld.zip
```

Проверка архива, 3022 пар изображений/аннотаций, диапазонов YOLO-координат,
классов, точных дубликатов и создание manifests:

```bash
weldvision prepare-lohi \
  --archive data/raw/lohi-weld.zip \
  --destination data/raw/lohi \
  --manifests data/manifests \
  --fold 0 \
  --seed 42
```

Отчёт записывается в `data/manifests/lohi_audit.json`. Ожидаемые значения:

```text
high: 1022 изображения
low: 2000 изображений
pore: 3950
deposit: 2935
discontinuity: 7220
stain: 8307
```

Опубликованные folds LoHi воспроизводятся в `source_official_*.txt`, но созданы
случайным разбиением отдельных изображений. Это может помещать кадры одной
производственной серии в train и test. Поэтому основные диссертационные эксперименты
используют `source_train.txt`, `source_val.txt`, `source_test.txt`, где группы
формируются по префиксу серии. Официальный split сохраняется только для честного
сравнения с опубликованным baseline.

### Малый RGB challenge set

Набор Zenodo `10.5281/zenodo.17402020` имеет CC-BY-4.0, однако его классы
`slag inclusion/spatter/undercut` несовместимы с таксономией LoHi-WELD. Он не
добавляется в обычный train/test автоматически.

```bash
curl -L \
  "https://zenodo.org/api/records/17402020/files/Final_Dataset_YOLO_Test.zip/content" \
  -o data/raw/weld-surface-rgb.zip

weldvision prepare-rgb-challenge \
  --archive data/raw/weld-surface-rgb.zip \
  --destination data/raw/rgb-challenge \
  --manifests data/manifests
```

Аудит сохраняется в `rgb_challenge_audit.json`. Набор используется только как
внешний robustness challenge либо как unlabeled smoke-test. Нельзя произвольно
переименовывать `spatter` в `deposit` или `undercut` в `discontinuity`: такое
сопоставление должен утвердить эксперт и отдельный протокол разметки.

### Структура

```text
ml/data/
├── source/
│   ├── images/
│   └── labels/
├── target/
│   ├── images/
│   └── labels/       # только экспертно размеченная часть
└── manifests/
    └── groups.csv
```

YOLO-строка:

```text
class_id center_x center_y width height
```

Координаты нормализованы в диапазоне `[0, 1]`. Классы в конфигурации:

```text
0 pore
1 deposit
2 discontinuity
3 stain
```

Не переименовывайте их в нормативные классы без экспертного сопоставления.

### Групповой манифест

`data/manifests/groups.csv`:

```csv
image_path,group_id,domain,labeled
../source/images/0001.jpg,lohi-weld-0001,source,true
../target/images/a-01.jpg,physical-weld-a,target,true
../target/images/a-02.jpg,physical-weld-a,target,false
../target/images/b-01.jpg,physical-weld-b,target,false
```

Все фотографии одного физического шва, кадры одного видео и одна серия съёмки
должны иметь одинаковый `group_id`.

Создание split:

```bash
weldvision split \
  --manifest data/manifests/groups.csv \
  --output data/manifests \
  --seed 42
```

Аудит:

```bash
weldvision audit --manifest data/manifests/groups.csv --seed 42
```

## Обучение

Сначала обучите воспроизводимый source-only baseline. Он не требует smartphone
target и служит контрольной точкой диссертации:

```bash
weldvision train \
  --config configs/lohi_baseline.yaml \
  --device cuda

weldvision evaluate \
  --config configs/lohi_baseline.yaml \
  --checkpoint outputs/lohi_baseline/student_final.pt \
  --split source_test \
  --device cuda
```

Не подбирайте параметры по `source_test`: используйте `source_val`, а test запускайте
после фиксации конфигурации. Для доверительных интервалов повторите baseline с seeds
из конфигурации.

Результат первой контрольной модели B0 зафиксирован в
[`results/B0_SEQUENCE_SAFE_BASELINE.md`](results/B0_SEQUENCE_SAFE_BASELINE.md).
Однофакторные ablations A1–A5 не дали конфигурации, которая одновременно
улучшает AP50 и recall50. Объединённый B1 хуже B0; letterbox (A1) — основной
отрицательный фактор. Source baseline заморожен как B0, итог в
[`results/ABLATION_SUMMARY.md`](results/ABLATION_SUMMARY.md).

SSDLite B0 не видит класс `pore` (recall ≈ 0.06). Перед адаптацией к смартфону
нужен более сильный source-детектор на **том же** sequence-safe split:

```bash
pip install -e ".[research]"
weldvision train-yolo --config configs/lohi_yolo_s.yaml --device 0
weldvision evaluate-yolo \
  --config configs/lohi_yolo_s.yaml \
  --checkpoint outputs/lohi_yolo_s/student_best.pt \
  --split source_test \
  --device 0
```

YOLOv8s (Ultralytics) лицензирован как AGPL-3.0: только диссертационное
сравнение, не Google Play. Лицензионно чистый запасной путь —
`configs/lohi_fasterrcnn_s0.yaml`. Порог перехода к RGB: macro AP50 ≥ 0.50 и
recall пор ≥ 0.25. Ячейки Colab: [`COLAB_STRONG_SOURCE.md`](COLAB_STRONG_SOURCE.md).

Конфигурация B1 сохраняется только как отрицательный комбинированный контроль,
а не как улучшенный детектор:

```bash
weldvision train \
  --config configs/lohi_baseline_b1.yaml \
  --device cuda
```

Для контроля влияния способа разбиения используйте рецепт **B0** на
опубликованном fold 0. Старый `lohi_official_control.yaml` повторяет B1 и
смешивает split с letterbox/cosine/balancing.

```bash
weldvision train \
  --config configs/lohi_official_b0.yaml \
  --device cuda

weldvision evaluate \
  --config configs/lohi_official_b0.yaml \
  --checkpoint outputs/lohi_official_b0/student_final.pt \
  --split source_test \
  --device cuda
```

Сначала сравниваются B0 и однофакторные ablations на одном sequence-safe test.
Official-control отвечает на отдельный вопрос о влиянии image-level split и не
подменяет основной test.

После появления собственного smartphone target настройте
`configs/lohi_to_mobile.yaml`, затем:

```bash
weldvision train --config configs/lohi_to_mobile.yaml --device cuda
```

Возобновление после отключения Colab:

```bash
weldvision train \
  --config configs/lohi_to_mobile.yaml \
  --device cuda \
  --resume outputs/lohi_to_mobile/last.pt
```

Стадии:

1. supervised baseline на размеченном source;
2. копирование student в EMA teacher;
3. teacher строит псевдоразметку weak RGB view;
4. quality score и calibrator фильтруют объекты;
5. student обучается на source и strong RGB view;
6. teacher обновляется экспоненциальным средним student.

Bootstrap quality score нужен только для запуска первых экспериментов. Для
диссертационного результата необходимо обучить отдельный gate на экспертных метках
`acceptable/blur/underexposed/overexposed/glare/seam_too_small`.

CSV для quality gate:

```csv
image_path,acceptable,blur,underexposed,overexposed,glare,seam_too_small
../target/images/a-01.jpg,1,0,0,0,0,0
../target/images/a-02.jpg,0,1,0,0,0,0
```

Обучение:

```bash
weldvision train-quality \
  --manifest data/manifests/quality_train.csv \
  --output outputs/quality/gate.pt \
  --epochs 20 \
  --device cuda
```

Если файл из `model.quality_checkpoint` существует, адаптационный trainer загрузит
его автоматически; иначе явно используется только эвристический bootstrap. Teacher
автоматически калибруется на `target_val` и сохраняет temperature в
`outputs/lohi_to_mobile/calibration.json`. Test-разметка при калибровке не читается.

## Оценивание

```bash
weldvision evaluate \
  --config configs/lohi_to_mobile.yaml \
  --checkpoint outputs/lohi_to_mobile/student_final.pt \
  --split test \
  --device cuda
```

Отчёт сохраняется в `outputs/lohi_to_mobile/evaluation_test.json`.

Обязательные сравнения:

1. опубликованный LoHi-WELD baseline;
2. source-only student;
3. source-only + photometric/ISP augmentation;
4. target grayscale conversion;
5. Mean Teacher с фиксированным threshold;
6. quality weighting без calibration;
7. quality + calibration;
8. fully supervised target как верхняя граница.

Каждый ключевой эксперимент проводится минимум с несколькими seeds. Разделение
фиксируется до обучения. Метрики публикуются по классам, телефонам и условиям
освещения вместе с доверительными интервалами.

## Экспорт

Экспортировать следует только зафиксированную валидированную модель:

```bash
weldvision export \
  --config configs/lohi_to_mobile.yaml \
  --checkpoint outputs/lohi_to_mobile/student_final.pt \
  --output outputs/mobile/weldvision.onnx
```

Команда создаёт FP32 ONNX и JSON с классами/нормализацией. INT8 не выполняется
«вслепую»: нужна representative calibration выборка смартфонов, сравнение FP32/INT8
на golden set и измерение деградации recall. После этого модель можно подключать к
ONNX Runtime Mobile либо конвертировать в LiteRT при доказанном совпадении выходов.

## Что ещё требуется для диссертации

- экспертная таксономия видимых дефектов;
- собственный smartphone RGB capture protocol;
- обучаемый quality gate;
- device-held-out и lighting-held-out test;
- calibrator, обученный только на validation;
- ablation study всех частей гипотезы;
- несколько seeds и статистический анализ;
- измерение latency, RAM и энергопотребления на реальных Android-устройствах;
- документирование отрицательных результатов и ограничений.
