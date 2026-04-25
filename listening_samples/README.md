# Listening samples

20 generated chorales for each of the six experimental arms, rendered with
**legato sustained notes at ≈ 50 BPM** for human listening study.

| Folder    | Condition                                                                 | Checkpoint              | Mean HS (n=20) |
|-----------|---------------------------------------------------------------------------|-------------------------|---------------:|
| `m1/`     | Vanilla baseline (M1)                                                     | `checkpoints/m1/best.pt` | 3.80 |
| `m2/`     | PPO fine-tune of M1 (M2) — retrained against the merged code              | `checkpoints/m2/best.pt` | 3.15 |
| `m3/`     | M1 + decode-time rule mask (M3)                                           | `checkpoints/m1/best.pt` | 1.65 |
| `m4/`     | Chord-conditioned SFT, free generation (M4)                               | `checkpoints/m4/best.pt` | 3.25 |
| `m4_m3/`  | M4 + rule mask at decode (M4 + M3) — **headline winner**                  | `checkpoints/m4/best.pt` | **1.35** |
| `m4_m5/`  | M4 + rule mask + metric weighting + HOLD prior + voice coupling (M4 + M5) | `checkpoints/m4/best.pt` | 3.05 |

Headline: **M1 (3.80) → M4+M3 (1.35) is a −64.5% reduction in mean HS** —
better than the writeup's pre-retrain −47%. See `eval/RETRAINED_RESULTS.md`
for the full table and the writeup for the three claim updates.

## Render style

Three changes from a naïve "one note per 16th" MIDI:

1. **Same-pitch + HOLD merging**: consecutive ticks at the same pitch (or
   the model's HOLD sentinel) collapse into a single sustained note, so
   what should be a quarter-note doesn't fire as four separate attacks.
2. **Legato bridging**: each note's end-time is extended to the start of
   the next note in the same voice. There is **zero silent gap between
   notes** — the previous pitch sustains until the next one begins, the
   way a singer or string player connects pitches. Every voice plays
   continuously throughout the piece.
3. **Slower tempo (~50 BPM)** at 0.30 s/16th gives each pitch room to
   breathe. Combined with the soft-attack "Choir Aahs" patch, the result
   sounds deliberate rather than choppy.

The unmodified one-note-per-16th renders (which `eval/rule_checker.py`
expects) live in `samples_eval/`. **Do not use these listen-rendered MIDIs
for rule checking** — the HS numbers in the table above were computed from
the eval-format twins, not these.

## Reproducing the listening samples

The 120 MIDIs in this folder were rendered from the trained M1 / M2 / M4
checkpoints by the project's sampling code:

1. Sample tokens from each checkpoint with the existing `sample/decode_*`
   scripts (free generation, seeds 0–19, n=20 per arm).
2. Render with `sample/_midi_utils.py::tokens_to_midi_listen` for the
   same-pitch / HOLD merging.
3. Apply legato bridging — extend each note's end-time to the start of
   the next note in that voice — and slow the tempo to 0.30 s per 16th
   (≈ 50 BPM).

The driver scripts that automate steps 1–3 (and the unmerge utility that
produces `samples_eval/<arm>/` for the rule checker) are kept in the
local working tree but are not committed to this repository. Per-sample
seeds (0–19) are aligned across all six arms, so a given index is a
controlled "same starting chord across conditions" comparison.
