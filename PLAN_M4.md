# M4 Plan — Chord-conditioned tokenizer + transpose augmentation + A/B study

**Created:** 2026-04-20
**Goal:** Replace the dropped Gibbs-refinement M4 with a chord-conditioned tokenizer (TonicNet-style, Roman-numeral granularity). Add transpose-to-12-keys augmentation. Run a human A/B listening study for the poster's qualitative row.

The PPO arm (M2) stays in the writeup as a negative result. The constrained-decoder arm (M3) still runs on M1 for comparison; we don't re-run its eval (keeping the old HS ≈ 1.35 number per Kelly 2026-04-20).

## Final experiment matrix

| Arm    | Training data      | Training objective                  | Decoding            |
|--------|--------------------|-------------------------------------|---------------------|
| M1     | JSB (1× keys)      | CE, SATB-packed                      | vanilla             |
| M1-aug | JSB (12× keys)     | CE, SATB-packed                      | vanilla             |
| M2     | JSB (1× keys)      | PPO on −HS (failed, keep as ablation) | vanilla             |
| M3     | M1 weights         | —                                   | constrained (M1 + logit shaping) |
| **M4** | JSB (12× keys)     | CE, chord + SATB-packed (interleaved)| vanilla             |
| M4 + M3| M4 weights         | —                                   | constrained         |

The row that matters for the headline is **M4 + M3 vs M1 + M3**: does chord conditioning add value on top of the best-known arm?

## Key design decisions (decided)

- **Chord representation:** Roman numerals with key context (vocab ~100+, final size determined after extraction from JSB).
- **Key normalization:** We keep each chorale in its detected key for training. Roman numerals are key-relative and stay invariant under transposition, so one RN vocab covers all 12 keys.
- **Sequence layout (interleaved):** `[KEY_token, RN_0, S_0, A_0, T_0, B_0, RN_1, S_1, A_1, T_1, B_1, ...]`. Period-1 timestep = 5 tokens (1 chord + 4 voices). A 64-timestep piece becomes `1 + 64*5 = 321` tokens (fits in `max_seq_len`).
- **Voice-aware positional embedding:** extend to 5 roles `{CHORD, S, A, T, B}` instead of 4.
- **Transpose augmentation:** at dataset load time, yield all 12 transpositions per chorale (or a random 1 per epoch — configurable).
- **Sampling modes:**
  1. *Free generation:* model emits its own chord sequence interleaved with notes.
  2. *Chord-progression-conditioned:* user supplies an RN progression; model fills voices.

## File changes

| File | Change |
|---|---|
| `data/chord_extractor.py` | **NEW.** Uses music21 to convert each JSB chorale → (key, [rn_0, rn_1, ...]) at 16th-note resolution. Caches result to `data/jsb_chords.pkl`. |
| `data/tokenizer.py` | Extend `ChoraleTokenizer`: add chord-RN token range, key tokens; interleaved encode/decode. |
| `data/jsb_loader.py` | Use new tokenizer; add `transpose_all_keys` flag; merge chord labels from `jsb_chords.pkl`. |
| `model/positional.py` | Extend voice embedding to 5 roles (add CHORD). |
| `model/transformer.py` | No changes (vocab size auto-propagates). |
| `train/train_m4.py` | **NEW.** Copy of `train_m1.py` with the new tokenizer + augmentation. Class weights updated for new tokens. |
| `sample/decode_m4.py` | **REWRITE** (replacing old rerank). Chord-conditioned autoregressive sampler. Supports `--chord_progression` flag. |
| `sample/_midi_utils.py` | Strip chord tokens before writing MIDI. |
| `eval/ab_study.py` | **NEW.** Prepares survey stimuli: samples N from each arm, randomizes order, exports audio + form CSV. |
| `configs/base.yaml` | Add `chord` block: `enabled`, `vocab_size` (filled after extraction), `granularity: roman_numeral`, `transpose_augment: true`. |

## Step-by-step order (with rough time estimates on OSCAR)

### Step 1 — Chord extraction (2 hr dev + 10 min run)
- [ ] Write `data/chord_extractor.py`:
  - Load JSB pickle.
  - For each chorale: build a `music21.stream.Score` from the 4-voice arrays.
  - Detect key with `music21.analysis.discrete.KrumhanslSchmuckler` (or `.analyze('key')`).
  - At each 16th-note timestep, `chord.Chord(pitches_at_t)` → `roman.romanNumeralFromChord(c, key)`.
  - Strip inversion to start; decide whether to keep inversions after seeing vocab size.
  - Save `(key, [rn_str, ...])` lists to `data/jsb_chords.pkl`.
- [ ] Print vocab statistics (unique RN symbols, coverage of top 30, etc.). If vocab > 200 consider collapsing.

### Step 2 — Tokenizer (2 hr dev)
- [ ] Extend `ChoraleTokenizer` to know about chord tokens.
  - New ranges in vocab: `KEY_MAJOR`, `KEY_MINOR`, then `RN_0 .. RN_{K-1}` where K is the extracted vocab size.
  - `encode_with_chords(chorale, rn_list) -> Tensor` produces the interleaved sequence.
  - `decode_with_chords(tokens) -> (chorale, rn_list)`.
- [ ] Update `voice_index(position)` helper to return one of `{0:CHORD, 1:S, 2:A, 3:T, 4:B}` for the 5-wide layout.

### Step 3 — Positional (30 min dev)
- [ ] `VoiceAwarePositional`: `n_voices=5`.
- [ ] Verify checkpoint-loading shapes for M1 don't regress (old checkpoints stay loadable as M1 baseline).

### Step 4 — Data loader + transpose aug (2 hr dev)
- [ ] `JSBChorales.__init__` takes `jsb_chords.pkl` path; joins pitch arrays with RN sequences.
- [ ] Add `transpose_all_keys: bool` option. If true, iterate over `shift in [-6, +5]` (or similar), yielding 12 copies per chorale with pitches shifted and RN unchanged.
- [ ] Sanity check: a transposed chorale + original RN should produce the same RN sequence when re-extracted. (One-off test, gate the run.)

### Step 5 — Train M4 (4 hr dev + 4-6 hr train)
- [ ] `train/train_m4.py`:
  - New class-weighted CE: down-weight HOLD (same as M1) + down-weight KEY token (appears once) so it doesn't dominate.
  - Rest of the training loop is identical to `train_m1.py`.
- [ ] Run on OSCAR: `sbatch scripts/oscar_train.sh train.train_m4`.
- [ ] Expect: lower pitch-only val CE than M1 (chord conditioning should reduce perplexity on pitch predictions).

### Step 6 — Decode M4 (3 hr dev)
- [ ] `sample/decode_m4.py`:
  - Free generation: model emits chord then voices, repeat.
  - Chord-conditioned generation: take a `--chord_progression` CSV (one RN per timestep), force the RN positions to those tokens.
  - Reuse `_midi_utils.tokens_to_midi` after filtering out chord tokens.
- [ ] Generate 50 samples for eval, 5 demo samples for the poster.

### Step 7 — Eval (1 hr)
- [ ] Run `rule_checker` on samples/m4 and samples/m4_m3.
- [ ] Produce the new headline table:
  - M1 unconstrained, M1+M3, M4 unconstrained, M4+M3.
  - HS + per-rule breakdown.
- [ ] Expected direction: M4 < M1 unconstrained (chord conditioning regularizes voice leading); M4+M3 ≤ M1+M3.

### Step 8 — A/B study (3 hr dev + 1-2 day recruit/run)
- [ ] `eval/ab_study.py`: pick 6 pairs (A=M1, B=M4 or A=M1+M3, B=M4+M3), randomize within each pair, export as `pair_{k}_option_A.mid` / `..._B.mid` + a ground-truth Bach sample.
- [ ] Build Google Form with embedded audio (or render to .mp3 and upload to public bucket). Questions: "Which is more Bach-like?" (5-pt Likert), "Any obvious voice-leading errors?" (yes/no).
- [ ] Recruit n=10 (1470 classmates, roommates). Aim for a mix of music-trained and untrained listeners.
- [ ] Analyze with Wilcoxon signed-rank (matched-pair) per-pair, aggregate to poster.

## Poster-ready deliverables

1. **Four-arm comparison table:** M1, M2 (negative result), M3, M4.
2. **New table:** chord-conditioning ablation — does it help *with* and *without* M3?
3. **Qualitative row:** A/B study results (human preference %, musician vs non-musician split).
4. **Demo:** chord-progression-conditioned generation (M4 filling voices for a user-supplied RN progression).

## Risk register

- **Risk 1:** music21 RN extraction is noisy on fermata endings / Picardy thirds / non-functional passages. *Mitigation:* seed a `RN_OTHER` token for unparseable moments; inspect the top-N most common RN_OTHER contexts and decide whether to special-case.
- **Risk 2:** 12× data blows up training time. *Mitigation:* keep `piece_length=64` / `stride=32` so the number of *chunks* grows ~12×, but training steps per epoch scale linearly. Budget 4-6 hr OSCAR instead of 1-2.
- **Risk 3:** the model learns to ignore chord tokens. *Mitigation:* validate by scoring held-out chorales with *wrong* chord labels and confirming val CE rises (sanity check that chord conditioning is load-bearing).
- **Risk 4:** A/B sample size too small for statistical significance. *Mitigation:* pre-register the comparison, report effect sizes, don't over-claim. n=10 is a poster number, not a paper number.

## Out of scope

- No Gibbs refinement (Kelly 2026-04-20).
- No re-run of M3 eval with the fixed sampler (Kelly 2026-04-20).
- No Tier 2 Option B (TCN branch) or Option C (SCG).

## Status checkpoints

| Checkpoint | Expected date | Gate |
|---|---|---|
| Chord extraction done + vocab printed | +1 day | Vocab size < 200 |
| M4 trainer runs 1 epoch locally | +2 days | Loss decreases |
| M4 full training on OSCAR | +3 days | Val CE < M1 |
| M4 samples + eval table | +4 days | HS < M1 (directional) |
| A/B stimuli ready | +5 days | 6 matched pairs |
| A/B results collected | +7 days | n ≥ 10 listeners |
| Final writeup + poster draft | +10 days | Submittable |
