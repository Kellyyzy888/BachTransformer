# M3-metric ablation — run book

This ablation tests whether **metric-weighted rule masking** reduces
HarmonicScore beyond the uniform M3 mask that produced the 2.24 result.

The claim being tested: species counterpoint says parallel motion on a
downbeat is more egregious than on an off-beat. The current M3 mask
penalizes both equally. If the weighting matches Bach's tolerance, we
expect HS to drop further, or at minimum for the spacing-violation
tradeoff to relax (since we are masking less aggressively off-beat).

## Commands

All runs use the same M4 checkpoint (`checkpoints/m4/best.pt`, step 4000).

### 1. Re-baseline (M4 + uniform M3) — reproduce the 2.24 number
```bash
python -m sample.decode_m4 \
  --ckpt checkpoints/m4/best.pt \
  --constrained \
  --n 50 \
  --out_dir samples/m4_m3_uniform
python3 -m eval.run_eval --samples_dir samples/m4_m3_uniform --out eval/m4_m3_uniform.csv
```

### 2. New — M4 + metric-weighted M3
```bash
python -m sample.decode_m4 \
  --ckpt checkpoints/m4/best.pt \
  --constrained --metric \
  --n 50 \
  --out_dir samples/m4_m3_metric
python3 -m eval.run_eval --samples_dir samples/m4_m3_metric --out eval/m4_m3_metric.csv

# side-by-side comparison
python3 -m eval.run_eval \
  --samples_dir samples/m4_m3_metric \
  --compare_to samples/m4_m3_uniform \
  --out eval/m4_m3_metric_vs_uniform.csv
```

### 3. Listenable renders for the A/B study (same tokens, different MIDI)
```bash
python -m sample.decode_m4 \
  --ckpt checkpoints/m4/best.pt \
  --constrained --metric --listen \
  --n 20 --out_dir samples/ab_study/m4_m3_metric_listen

python -m sample.decode_m4 \
  --ckpt checkpoints/m4/best.pt \
  --constrained --listen \
  --n 20 --out_dir samples/ab_study/m4_m3_uniform_listen
```

The `--listen` flag renders sustained notes (merges HOLD runs) and uses
100 BPM. These stimuli are for the A/B study only; they cannot be fed
to `rule_checker` because the renderer does not emit one note per 16th.

## Table to populate (5 pieces × 50 samples each)

| Arm | HS mean | HS std | ∥5ths | ∥8ves | Voice cross. | Hidden 5ths | Hidden 8ves | Spacing | Large leap | Aug. leap |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 (baseline) | 4.20 | 3.16 | 37 | 57 |  |  |  |  |  |  |
| M4 (chord only) | 4.56 | 3.84 | 51 | 61 |  |  |  |  |  |  |
| M4 + M3 uniform | 2.24 | 2.80 | 11 | 1 |  |  |  |  |  |  |
| **M4 + M3 metric** |  |  |  |  |  |  |  |  |  |  |

## What to report on the poster

- Headline unchanged: M4+M3 cuts HS by 47% vs M1.
- New poster bullet ONLY if M4+M3-metric does *something* interesting:
  - Lower HS → "metric-weighted masking further reduces HS by X%"
  - Same HS, lower spacing → "metric weighting relaxes the spacing
    tradeoff at no HS cost"
  - Higher HS → negative result: "uniform masking is already near the
    ceiling at this model scale; metric weighting helps only at larger
    scale" (honest, consistent with the small-scale thesis)

All three outcomes are publishable. A negative result that supports
the "explicit > implicit at small scale" thesis is the second-best case.

## Why this isn't a copy of DeepBach / TonicNet / REMI

Those systems inject metric information as a **model-side conditioning
feature**. This ablation treats metric position as a **decode-time
constraint modulator** — the model is unchanged, the same step-4000
checkpoint is used for every arm. This is a direct extension of the
project's existing M3 contribution ("explicit structural enforcement at
decode time beats implicit train-time conditioning at small scale"),
generalized from harmonic structure to metric structure.

## Sanity checks before reporting numbers

1. Confirm the M4+M3 uniform rerun matches the filed 2.24 within noise
   (±0.2). If not, something else drifted — debug before trusting the
   metric row.
2. Spot-check 3 metric-weighted MIDI files: are parallel motions that
   occur off-beat still present, while downbeat parallels are gone?
   That's the signature of the ablation actually working, independent
   of the aggregate HS.
3. Tempo: the rule-checker renderer uses 0.25 s/step (60 BPM). The
   listen renderer uses 0.15 s/step (~100 BPM). Do not compare audio
   between the two — use listen-render audio only for the A/B study.
