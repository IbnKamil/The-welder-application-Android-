# B1 — предварительная оценка до исправления AP floor

Дата запуска: 14 августа 2026 года  
Test: тот же sequence-safe набор из 867 изображений, что и B0.  
Выбран checkpoint лучшей validation эпохи: **25**.  
Validation macro truncated AP50: **0.0743**.  
Validation macro recall50: **0.3057**.

## Конфигурация

- aspect-ratio-preserving letterbox;
- class-balanced sampler;
- photometric augmentation;
- cosine scheduler;
- best checkpoint по validation macro AP50.

Эти факторы были включены одновременно, поэтому результат не позволяет определить
причину изменения качества без однофакторных ablations.

## Результат при score floor 0.25

Macro truncated AP50: **0.1113** против B0 **0.2020**.  
Macro precision50: **0.2660** против B0 **0.3792**.  
Macro recall50: **0.2031** против B0 **0.3088**.  
Macro truncated AP75: **0.0101** против B0 **0.0279**.

B1 хуже B0 в рабочей точке confidence 0.25. При этом значения AP обеих моделей
были рассчитаны с ошибочно высоким score floor и должны быть переоценены с floor
0.001. Precision/recall при 0.25 уже показывают реальную деградацию рабочей точки.

## Решение

1. не принимать объединённый B1 как улучшение;
2. переоценить B0/B1 исправленным evaluator без переобучения;
3. выполнить однофакторные ablations: letterbox, best-checkpoint, scheduler,
   balancing и photometric augmentation по отдельности;
4. не начинать smartphone adaptation до выбора сильного source baseline.

## Последующий результат A1

Однофакторный A1 показал снижение macro AP50 с 0.2791 до 0.1156. Таким образом,
letterbox является подтверждённым отрицательным фактором для узких LoHi crops и
объясняет существенную часть деградации объединённого B1. Эффекты остальных факторов
по B1 всё ещё неидентифицируемы и требуют A2–A5.
