# Bach Transformer

A from-scratch PyTorch decoder-only Transformer (~4.8M params) for four-part
(SATB) Bach chorale generation, with music-theory rule injection at three
different points in the pipeline.

CSCI 1470 capstone project, Brown University. See [`../writeup.md`](../writeup.md)
for the full paper; this README covers how to run the code.

## Headline result

**M4 + M3 (chord-conditioned SFT + rule-masked decoding) reduces mean
HarmonicScore by 47% vs. the vanilla Transformer baseline** on JSB Chorales
(4.20 → 2.24, n=50 generated pieces). Parallel octaves drop from 57 to 1.

Chord conditioning alone (M4) does not help; rule-masked decoding alone (M3)
helps; the combination is where the win comes from. See the writeup for the
ablation story.

## Experimental arms

| ID | What changes | Where the rules live | Result vs M1 |
|----|--------------|----------------------|--------------|
| **M1** | Baseline SFT, no rule info | nowhere | — |
| M2 | PPO fine-tune with `−HarmonicScore` reward | training (RL) | +10% (worse) |
| M3 | M1 + logit masking at decode | sampling | 1.35 HS (much better, earlier eval) |
| **M4** | Chord-conditioned SFT (TonicNet-style, interleaved Roman numerals, all-12-key transpose) | training (data) | +8% (worse) |
| **M4 + M3** | M4 model with M3 rule-mask at decode | both | **−47% (best)** |

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
    decode_m3.py         # constrained decoding (logit masking)
    decode_m4.py         # M4 sampling (free + chord-progression-forced)
                         #   pass --constrained to stack M3 rule mask
    _midi_utils.py       # shared token→MIDI conversion
  eval/
    run_eval.py          # sweep generated MIDI through rule_checker.py
    ab_study.py          # human A/B listening study (prepare + analyze)
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
scp -r . 'zyang188@ssh.ccv.brown.edu:~/bach_transformer/'

# From OSCAR login node: submit a training job
ssh zyang188@ssh.ccv.brown.edu
cd ~/bach_transformer
sbatch -J bach_m1                                 scripts/oscar_train.sh
sbatch -J bach_m2   --export=ALL,STAGE=m2         scripts/oscar_train.sh
sbatch -J bach_m4   --export=ALL,STAGE=m4         scripts/oscar_train.sh

# Monitor
squeue -u zyang188
tail -f logs/bach_m4_<jobid>.out
```

**Gotcha:** every stage flag matters. Forgetting `-J bach_m2 --export=ALL,STAGE=m2`
silently runs M1 and overwrites `checkpoints/m1/best.pt`.

**Gotcha:** never run anything that imports torch on a login node — Brown's
OSCAR auto-penalizes login-node compute. Use `interact` or `sbatch`.

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
constrained decoder, chord-interleaved training loop, and A/B study are
implemented from scratch in this repo. No pretrained checkpoints were used.
