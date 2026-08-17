# Итог однофакторных ablations A1–A5

Дата фиксации: 17 августа 2026 года  
Все эксперименты: тот же sequence-safe split, seed 42, SSDLite-MobileNetV3,
вход 640, test 867 изображений, AP с confidence floor 0.001, рабочая точка 0.25.

## Macro на source_test

| ID | Единственный фактор vs B0 | AP50 | Precision50 | Recall50 | AP75 | Решение |
|---|---|---:|---:|---:|---:|---|
| B0 | контроль (stretch, last epoch, constant LR) | 0.2791 | 0.3792 | 0.3088 | 0.0304 | **зафиксирован** |
| A1 | letterbox | 0.1156 | 0.2188 | 0.1610 | 0.0070 | отвергнут |
| A2 | best checkpoint | 0.2802 | 0.4108 | 0.2898 | 0.0251 | нет прироста |
| A3 | cosine scheduler | 0.2862 | 0.4067 | 0.2775 | 0.0294 | не принимаем |
| A4 | balanced sampler | 0.2807 | 0.4048 | 0.2803 | 0.0288 | pore ↑, macro flat |
| A5 | photometric augmentation | 0.2904 | 0.4418 | 0.2818 | 0.0366 | AP↑, pore recall↓ |

Объединённый B1 (все факторы сразу) был хуже B0 и **не интерпретируется**
как эффект отдельного фактора. A1 показывает, что letterbox объясняет
существенную часть этой деградации.

## Что принято в диссертационный source baseline

Конфигурация `configs/lohi_baseline.yaml` (B0):

- растягивание узкого weld crop до 640×640 без letterbox;
- без class-balanced sampler;
- без photometric augmentation на source;
- постоянный learning rate 0.0005;
- checkpoint последней эпохи.

Ни один однофакторный вариант не улучшает одновременно AP50 и recall50.
A4 — единственный прирост recall `pore` (0.0559 → 0.0748). A5 — единственный
прирост AP75 (0.0304 → 0.0366). Оба эффекта сопровождаются потерей другого
класса или общего recall, поэтому в baseline не входят.

## Следующие стадии (не source-ablation)

1. Official LoHi fold 0 выполнен. На одном official test рецепт B0 (AP50
   0.1635) лучше B1 (0.0590). Сравнение с sequence-safe B0 (0.2791) смешано
   с high-only vs high+low и не интерпретируется как чистый эффект split.
   Отчёт: `results/OFFICIAL_FOLD_CONTROL.md`.
2. Подготовка unlabeled smartphone RGB и quality gate — **после** того, как
   сильный source-детектор пройдёт порог AP50 ≥ 0.50 и pore recall ≥ 0.25.
   Пока SSDLite B0 не видит поры, адаптация к телефону не запускается.
   Команда: `train-yolo` / `configs/lohi_yolo_s.yaml` (AGPL, только диссертация).
3. Адаптация `configs/lohi_to_mobile.yaml` от замороженного B0 student:
   EMA teacher, калибровка, quality-conditioned pseudo-labels.
4. Для полного кадра смартфона — двухступенчатая схема
   `full frame → weld ROI → tight crop → detector`, а не глобальный letterbox.
