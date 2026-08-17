# Official LoHi fold 0 — текущий прогон (B1-рецепт)

Дата обучения: 17 августа 2026 года  
Checkpoint: `student_final.pt` / `student_best.pt`  
Этот запуск **не изолирует** влияние image-level split.

## Что смешано

Конфиг `lohi_official_control.yaml` повторяет отвергнутый B1, а не замороженный B0:

- letterbox;
- cosine scheduler (LR 0.0005 → 1e-6);
- balanced sampler;
- photometric augmentation;
- best-checkpoint;
- batch size 4;
- только high-resolution official fold 0, без low.

Sequence-safe B0 отличался и split, и рецептом обучения, и составом high+low.
Сравнивать test AP этого прогона с B0 AP50 0.2791 **нельзя**.

## Validation

| Эпоха | val AP50 | val recall50 |
|------:|---------:|-------------:|
| 4 | 0.0559 | 0.2932 |
| 9 | 0.0492 | 0.2829 |
| 14 | **0.0566** | 0.2894 |
| 19 | 0.0490 | 0.2147 |
| 24 | 0.0438 | 0.1763 |
| 49 | 0.0427 | 0.1300 |

Лучшая эпоха ≈ 14. Последняя хуже: модель переобучается. Оценивать нужно
`student_best.pt`.

## Дальше

1. Оценить этот прогон на `source_official_test` — как отрицательный контроль
   «B1-рецепт на official fold».
2. Обучить изолированный контроль `configs/lohi_official_b0.yaml`: рецепт B0,
   manifests `source_official_*.txt`.
