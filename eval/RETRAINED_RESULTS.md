# Eval results after retraining M2 + M4 against the merged code

After the teammate's M2 / M3-M4 / M4-improvement commits were merged into main
and M2 + M4 were retrained on OSCAR, n=20 generated chorales per condition were
scored with `eval/rule_checker.py`. All numbers are means over the 20 samples.

## Mean HarmonicScore by condition

| Condition | Mean HS | par8 | par5 | cross | hid8 | hid5 | space | leap |
|-----------|--------:|-----:|-----:|------:|-----:|-----:|------:|-----:|
| M1        |   3.80  |  1.4 |  1.1 |   0.2 |  0.0 |  0.2 |   0.0 |  0.9 |
| M2        |   3.15  |  1.2 |  0.7 |   0.1 |  0.1 |  0.1 |   0.1 |  0.8 |
| M3        |   1.65  |  0.0 |  0.1 |   0.0 |  0.1 |  0.1 |   0.0 |  1.5 |
| M4        |   3.25  |  1.4 |  0.7 |   0.1 |  0.1 |  0.1 |   0.2 |  0.7 |
| **M4+M3** | **1.35**|  0.0 |  0.3 |   0.1 |  0.1 |  0.1 |   0.1 |  0.8 |
| M4+M5     |   3.05  |  0.0 |  0.3 |   0.1 |  0.1 |  0.4 |   1.1 |  1.1 |

Headline: **M1 (3.80) → M4+M3 (1.35) is a −64.5% reduction in mean HS.**

## How this differs from the writeup's original result

| | Writeup (pre-retrain, n=50) | Retrained (n=20) |
|---|--:|--:|
| M1 mean HS | 4.20 | 3.80 |
| M4 mean HS | 4.55 (+8% vs M1) | 3.25 (−14% vs M1) |
| M4+M3 mean HS | 2.24 (−47% vs M1) | **1.35 (−64.5% vs M1)** |
| M2 mean HS | ~4.62 (+10% vs M1) | **3.15 (−17% vs M1)** |

Three claims in the writeup need updating:

1. **"Chord conditioning alone (M4) does not help"** — no longer true. With the
   teammate's `chord_attn_bias` parameter actually trained-in (the previous
   M4 ckpt loaded with `missing=['pos.rn_bias']`, randomly-initializing it),
   M4 alone now beats M1 by 14%. The combo with M3 still wins by a much
   larger margin, so the "rule-masked decoding does the heavy lifting" point
   stands, but the framing should be "M4 helps a little, M4+M3 helps a lot."

2. **"M2 (PPO with −HS reward) is +10% worse than M1"** — flipped. The
   teammate's PPO modifications (commit `3d472a7`) brought M2 from a negative
   result to a positive result: 17% better than M1. Worth a callout in the
   Reflection or Challenges section.

3. **"−47% HS reduction"** is now **"−64.5% HS reduction"** for the M4+M3
   arm. The qualitative win (parallel octaves go to zero in M4+M3, voice
   crossings vanish) is preserved.

## Sample-size caveat

The original writeup used n=50; this re-eval is n=20 because we needed to
chunk generation through 45-second shell calls. With the bigger M4
architecture each sample takes ~17s. The direction of the changes is robust
(means are too far apart for n=20→50 to flip a sign), but the exact percent
reductions could shift by a few points. **Re-run with n=50 before citing exact
numbers in the final paper.**

To regenerate at n=50, rerun each condition's sampler against its
checkpoint with the project's existing `sample/decode_*.py` scripts at
n=50, write the MIDIs into `samples_eval/<arm>/`, and feed the per-arm
folders into the rule checker:

```python
from eval.rule_checker import diagnose_folder, harmonic_score
for c in ['m1','m2','m3','m4','m4_m3','m4_m5']:
    reps = diagnose_folder(f'samples_eval/{c}')
    print(c, sum(harmonic_score(r) for r in reps) / len(reps))
```

## Per-arm CSVs

Each `eval/retrained_<arm>.csv` contains the per-piece rule counts and
HarmonicScore so you can compute std-dev / median / Wilcoxon stats if needed
for the paper.
