# Official LoHi fold 0 — контроль протокола разбиения

Дата оценки: 17 августа 2026 года  
Test: опубликованный high-resolution fold, **205** изображений
(`source_official_test.txt`). Это не sequence-safe test из 867 кадров.

Оба прогона ниже используют один и тот же official test. Отличается только
рецепт обучения.

## Macro-метрики

| Прогон | Рецепт | Split / данные | AP50 | Precision50 | Recall50 | AP75 |
|---|---|---|---:|---:|---:|---:|
| Sequence-safe B0 | B0 | group-safe, high+low, 867 test | 0.2791 | 0.3792 | 0.3088 | 0.0304 |
| Official B0 | B0 | official fold 0, high only, 205 test | 0.1635 | 0.2986 | 0.1729 | 0.0148 |
| Official B1-recipe | B1 | official fold 0, high only, 205 test | 0.0590 | 0.1443 | 0.1142 | 0.0019 |

## Классы official test, IoU 0.50, порог 0.25

Official test содержит мало пор (86 объектов) и много пятен (856).

| Класс | GT | Official B0 AP | Official B0 Recall | Official B1 AP | Official B1 Recall |
|---|---:|---:|---:|---:|---:|
| pore | 86 | 0.0097 | 0.0116 | 0.0048 | 0.0233 |
| deposit | 390 | 0.1290 | 0.1590 | 0.0115 | 0.0615 |
| discontinuity | 596 | 0.3986 | 0.3926 | 0.2026 | 0.3171 |
| stain | 856 | 0.1165 | 0.1285 | 0.0171 | 0.0549 |

## Что изолировано, а что нет

На одном official test рецепт B0 лучше B1: AP50 0.1635 против 0.0590
(+177% относительно B1-рецепта). Вывод A1/B1 воспроизводится на другом
протоколе разбиения: letterbox и объединённые «улучшения» вредны.

Сравнение official B0 (0.1635) с sequence-safe B0 (0.2791) **нельзя** читать
как «image-level split труднее/легче». Одновременно меняются:

- протокол split (image-level fold vs group-safe);
- состав данных (только high vs high+low);
- размер train/test (205 test-кадров vs 867);
- распределение классов (86 пор vs 1538).

Гипотеза «official fold завышает качество из-за утечки серии» этими двумя
прогонами не подтверждается и не опровергается. Чтобы изолировать только
утечку, нужен image-level split на том же high+low пуле, что и B0. Официальные
folds LoHi этого пула не содержат.

## Official B0: обучение

Learning rate постоянный 0.0005. Loss 7.94 → 0.19. Лучший val AP50 **0.1761**
на эпохе 31, последняя эпоха 0.1471. По протоколу B0 оценена последняя эпоха.

## Решение

1. Source baseline диссертации остаётся sequence-safe B0.
2. Official fold — отдельный контроль воспроизводимости рецепта, не замена
   основного test.
3. B1-рецепт на official fold фиксируется как отрицательный результат.
4. Следующая стадия — smartphone RGB (unlabeled target + quality gate +
   `lohi_to_mobile.yaml` от замороженного B0).
