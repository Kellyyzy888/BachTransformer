# Bach Transformer

A from-scratch PyTorch decoder-only Transformer (~4.8M params) for four-part
(SATB) Bach chorale generation, with music-theory rule injection at three
different points in the pipeline.

CSCI 1470 final project, Brown University. See [`../writeup.md`](../writeup.md)
for the full paper; this README covers how to run the code.

## Headline result

**M4 + M3 (chord-conditioned SFT + rule-masked decoding) reduces mean
HarmonicScore by 64.5% vs. the vanilla Transformer baseline** on JSB Chorales
(3.80 → 1.35, n=20 generated pieces). Parallel octaves drop from 1.4 per piece
to 0.0; voice crossings and hidden octaves likewise vanish.

All three interventions help individually — chord conditioning alone (M4) cuts
HS by 14%, decode-time rule masking alone (M3) by 57%, PPO fine-tuning (M2) by
17% — but combining chord conditioning with rule-masked decoding is where the
biggest win comes from. See the writeup for the ablation story and how the
M4 architecture (8 layers, `d_ff=1536`, learned chord-attention bias) compares
to the M1 baseline architecture.

We then extended the same decode-time-constraint philosophy from harmony to
**rhythm and texture** (M5): a metric-weighted rule mask, a beat-aware HOLD
prior, cross-voice articulation coupling, and REST suppression together cut
the per-voice attack density from 16 per bar down to about 3 — Bach territory
— without touching the model or the HS score. See the "Meter-aware decoding"
section below for how to run it.

## Experimental arms

| ID | What changes | Where the rules live | Result vs M1 |
|----|--------------|----------------------|--------------|
| **M1** | Baseline SFT, no rule info | nowhere | — (HS 3.80) |
| M2 | PPO fine-tune with `−HarmonicScore` reward | training (RL) | −17% (3.15) |
| M3 | M1 + logit masking at decode | sampling | −57% (1.65) |
| **M4** | Chord-conditioned SFT (TonicNet-style, interleaved Roman numerals, all-12-key transpose, 8 layers + chord-attn bias) | training (data + arch) | −14% (3.25) |
| **M4 + M3** | M4 model with M3 rule-mask at decode | both | **−64.5% (1.35, best HS)** |
| M4 + M5 | M4 model with meter-aware decode stack (metric mask + HOLD prior + voice coupling + REST penalty) | sampling (rhythm + texture) | −20% HS (3.05), attacks/bar 16 → 3 (Bach-like texture) |

> Numbers above are from `eval/RETRAINED_RESULTS.md` (n=20). An earlier
> eval pass (n=50, with a smaller M4 architecture and an unmodified PPO
> loop) produced different numbers — see the writeup's Reflection section
> for that history.

## Layout

```
bach_transformer/
  data/
    tokenizer.py         # SATB-packed + chord-interleaved token encoder
    jsb_loader.py        # JSB Chorales + transpose augmentation
    chord_extractor.py   # music21 Roman-numeral analysis → chord cache
    jsb_chords.pkl       # cached chord labels (built by chord_extractor)
  model/
    positional.py        # timestep + voice-role embeddings
                         #   (supports 5-role chord layout)
    transformer.py       # decoder-only Transformer (from scratch)
  train/
    train_m1.py          # M1: plain cross-entropy SFT
    train_m2.py          # M2: PPO fine-tune from M1
    train_m2_diffloss.py # M2 ablation: differentiable rule loss
    train_m4.py          # M4: chord-conditioned SFT
  sample/
    decode_m1.py         # vanilla autoregressive sampling
    decode_m2.py         # M2 sampling (matches PPO rollout distribution)
    decode_m3.py         # constrained decoding (logit masking, uniform)
    decode_m4.py         # M4 sampling (free + chord-progression-forced)
                         #   --constrained stacks M3 rule mask
                         #   --metric     metric-weighted parallel penalty
                         #   --hold_prior beat-aware HOLD logit bias
                         #   --couple     cross-voice articulation coupling
                         #   --listen     sustained-note renderer (A/B audio)
    metric_mask.py       # M5: metric weights, HOLD prior, voice coupling,
                         #     REST suppression — all decode-time
    _midi_utils.py       # shared token→MIDI conversion (rule + listen renders)
  eval/
    run_eval.py          # sweep generated MIDI through rule_checker.py
    count_holds.py       # per-voice attack density and HOLD fraction diagnostics
    ab_study.py          # human A/B listening study (prepare + analyze)
    METRIC_ABLATION.md   # run book for the M5 ablation rows
  literature/
    related_work.md      # SOTA positioning, BibTeX
  configs/base.yaml      # all hyperparameters
  scripts/oscar_train.sh # OSCAR/SLURM launcher (STAGE=m1|m2|m4)
  PLAN_M4.md             # design doc for M4
```

## Quick start (local CPU)

```bash
pip install torch pretty_midi music21 tqdm pyyaml

# Step 1: extract chord labels (one-time, ~1 minute)
python -m data.chord_extractor \
    --in JSB-Chorales-dataset/jsb-chorales-16th.pkl \
    --out data/jsb_chords.pkl

# Step 2: train M1 (vanilla baseline)
python -m train.train_m1 --config configs/base.yaml

# Step 3: train M4 (chord-conditioned)
python -m train.train_m4 --config configs/base.yaml \
    --override chord.enabled=true

# Step 4: sample from each
python -m sample.decode_m1 --ckpt checkpoints/m1/best.pt --n 50 --out_dir samples/m1
python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt --n 50 --out_dir samples/m4
python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt --n 50 \
    --constrained --out_dir samples/m4_m3   # the winner

# Step 5: evaluate
python -m eval.run_eval --samples_dir samples/m4_m3 --compare_to samples/m1
```

## Running on Brown OSCAR (GPU)

```bash
# From our laptops: push files
scp -r . 'username@ssh.ccv.brown.edu:~/bach_transformer/'

# From OSCAR login node: submit a training job
ssh username@ssh.ccv.brown.edu
cd ~/bach_transformer
sbatch -J bach_m1                                 scripts/oscar_train.sh
sbatch -J bach_m2   --export=ALL,STAGE=m2         scripts/oscar_train.sh
sbatch -J bach_m4   --export=ALL,STAGE=m4         scripts/oscar_train.sh

# Monitor
squeue -u username
tail -f logs/bach_m4_<jobid>.out
```

**Gotcha:** every stage flag matters. Forgetting `-J bach_m2 --export=ALL,STAGE=m2`
silently runs M1 and overwrites `checkpoints/m1/best.pt`.

**Gotcha:** never run anything that imports torch on a login node — Brown's
OSCAR auto-penalizes login-node compute. Use `interact` or `sbatch`.

## Meter-aware decoding (M5)

The M5 stack adds four decode-time constraints to M4+M3, all composable via
CLI flags. The same checkpoint (`checkpoints/m4/best.pt`) is used for every
arm; only the sampler changes.

```bash
# Full M5 stack: harmonic rule mask + metric weighting + HOLD prior +
# voice coupling, rendered with sustained notes at ~75 BPM
python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt \
    --constrained --metric --hold_prior --couple \
    --listen --seconds_per_step 0.20 \
    --n 50 --out_dir samples/m4_m5

# Check the rhythmic fingerprint
python -m eval.count_holds samples/m4_m5
# Expected: attacks/bar/voice ~ 3, mean note length ~ 5 × 16th,
# implied HOLD fraction ~ 80%
```

Each flag is independently ablatable:

| Flag | What it adds |
|------|--------------|
| `--constrained` | M3 uniform rule mask (parallel 5ths/8ves, crossings, leaps) |
| `--metric` | Scale the parallel-motion penalty by beat position (1.5× on downbeat, 0.25× on 16th off-beats) |
| `--hold_prior` | Bias HOLD's logit by beat position (−2 on downbeat, +7 on 16th off-beats) |
| `--hold_prior_scale FLOAT` | Multiply every HOLD bias (default 1.0; try 2.0 if still chopped, 0.5 if droning) |
| `--couple` | Enforce "at most one voice moves per off-beat" (species-counterpoint rule) |
| `--listen` | Merge HOLD runs + same-pitch runs into sustained notes for audio (do NOT combine with rule-checker eval) |
| `--seconds_per_step FLOAT` | Per-16th duration; 0.25 = 60 BPM (default for rule eval), 0.20 = 75 BPM, 0.15 = 100 BPM |

Run book with all commands and the target numbers:
[`eval/METRIC_ABLATION.md`](eval/METRIC_ABLATION.md).

## Chord-progression-forced generation (M4 only)

Because M4 conditions on Roman numerals, we can pre-specify a progression
and ask the model to compose a piece that follows it:

```bash
python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt \
    --progression "I IV V I vi ii V I" \
    --n 4 --out_dir samples/m4_progression_demo
```

## A/B listening study

Build blinded stimuli from existing M1 / M4+M3 sample banks:

```bash
python -m eval.ab_study prepare \
    --bank_a samples/m1        --label_a M1 \
    --bank_b samples/m4_m3     --label_b M4+M3 \
    --n_pairs 5 \
    --hs_max 6 \
    --out_dir study/stimuli
```

After listeners complete `listener_form.csv`:

```bash
python -m eval.ab_study analyze --responses study/responses.csv
```

Reports Wilcoxon signed-rank statistics on listener-level mean preferences.

## Citation

If others build on this work, we ask that they cite our writeup and the
TonicNet paper (Peracha 2019, arXiv:1911.11775) that inspired our M4
chord-conditioning layout.

## Open-source components

- **PyTorch** — training framework
- **music21** — Roman-numeral chord analysis for M4 labels
- **pretty_midi** — MIDI file I/O
- **Krumhansl-Kessler** key-detection algorithm (implemented inside music21)

The Transformer, tokenizer, rule checker, chord extractor, PPO loop,
constrained decoder, chord-interleaved training loop, metric-weighted mask,
beat-aware HOLD prior, cross-voice articulation coupling, REST-suppressing
logit shaper, listening-side note merger, and A/B study are implemented from
scratch in this repo. No pretrained checkpoints were used.
