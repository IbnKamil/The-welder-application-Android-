# Corrected evaluation and B0-factor ablations

## 1. Re-evaluate existing checkpoints

No retraining is required. Pull the current code and regenerate the Colab configs as
before so that `ap_confidence_floor: 0.001` is present. Then:

```bash
python -m weldvision.cli evaluate \
  --config configs/lohi_baseline_colab.yaml \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_baseline/student_final.pt \
  --split source_test \
  --device cuda

python -m weldvision.cli evaluate \
  --config configs/lohi_baseline_b1_colab.yaml \
  --checkpoint /content/drive/MyDrive/SvarshikProAI/lohi_baseline_b1/student_best.pt \
  --split source_test \
  --device cuda
```

AP is calculated from confidence 0.001. Precision, recall and FP/image remain reported
at the operating threshold 0.25.

## 2. Generate isolated ablations

```bash
python -m weldvision.cli generate-ablations \
  --base configs/lohi_baseline.yaml \
  --output configs/generated_ablations \
  --results-root /content/drive/MyDrive/SvarshikProAI/ablations
```

Generated experiments:

| ID | The only changed factor relative to B0 | Status |
|---|---|---|
| A1 | letterbox | rejected (AP50 0.1156) |
| A2 | best checkpoint selection | no test AP gain (AP50 0.2802) |
| A3 | cosine scheduler | no clear gain (AP50 0.2862, recall down) |
| A4 | class-balanced sampler | pore recall up, macro flat (AP50 0.2807) |
| A5 | photometric augmentation | next |

Run one experiment at a time. All use the same sequence-safe train/validation/test
and seed. After screening, promising factors must be repeated with multiple seeds.

Do not infer the effect of an individual factor from combined B1.
