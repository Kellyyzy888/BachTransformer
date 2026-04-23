"""Shared helpers for the three decoding scripts.

Main job: convert a (4*T,) token sequence into a MIDI file that
rule_checker.py can ingest unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from data.tokenizer import ChoraleTokenizer


VOICE_NAMES = ["Soprano", "Alto", "Tenor", "Bass"]
VOICE_PROGRAMS = [52, 53, 49, 42]       # GM: choir aahs, voice oohs, strings, cello


def _strip_chord_tokens(
    tokens: torch.Tensor, tokenizer: ChoraleTokenizer
) -> torch.Tensor:
    """Drop KEY / PC / RN tokens from a chord-interleaved sequence.

    A no-op if the tokenizer isn't in chord mode or the sequence doesn't
    look chord-interleaved — makes `tokens_to_midi` safe to call with
    either M1 (4-wide) or M4 (5-wide + prefix) sequences without the
    caller having to know which.
    """
    if tokenizer.cfg.chord_vocab is None:
        return tokens
    ids = tokens.detach().cpu().tolist()
    # The chord layout is [MODE, PC, RN, S, A, T, B, RN, S, A, T, B, ...].
    # We detect it heuristically: if the first token is a KEY_MAJOR/MINOR,
    # assume the whole sequence is interleaved.
    if not (len(ids) >= 2
            and (ids[0] == tokenizer.KEY_MAJOR or ids[0] == tokenizer.KEY_MINOR)):
        return tokens
    body = ids[2:]
    kept: list[int] = []
    for i, t in enumerate(body):
        if i % 5 == 0:
            continue      # RN slot — drop
        kept.append(t)
    return torch.tensor(kept, dtype=tokens.dtype, device=tokens.device)


def tokens_to_midi(
    tokens: torch.Tensor,
    tokenizer: ChoraleTokenizer,
    out_path: str | Path,
    seconds_per_step: float = 0.25,
) -> None:
    """tokens: (L,) LongTensor. Writes a 4-track MIDI to `out_path`.

    Accepts either an M1 layout (4*T tokens, pitch-only) or an M4 layout
    (2 + 5*T tokens with a key prefix and interleaved RNs). Chord tokens
    are stripped before decoding, so downstream evaluators (rule_checker,
    pretty_midi) see a pitch-only stream in both cases.

    Emits **one discrete note per 16th-note tick per voice**, even when a
    pitch repeats. This matches what rule_checker.load_satb() expects (it
    recovers the pitch sequence by reading one note per step). REST ticks
    produce no note for that step.
    """
    pitch_tokens = _strip_chord_tokens(tokens, tokenizer)
    grid = tokenizer.decode(pitch_tokens)                  # (4, T)
    grid = tokenizer.resolve_holds(grid)                   # drop HOLD sentinels
    pm = pretty_midi.PrettyMIDI()
    T = grid.shape[1]
    for v in range(4):
        instr = pretty_midi.Instrument(program=VOICE_PROGRAMS[v], name=VOICE_NAMES[v])
        for t in range(T):
            p = int(grid[v, t])
            if p < 0:            # REST — skip this tick for this voice
                continue
            instr.notes.append(pretty_midi.Note(
                velocity=80, pitch=p,
                start=t * seconds_per_step,
                end=(t + 1) * seconds_per_step,
            ))
        pm.instruments.append(instr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def tokens_to_midi_listen(
    tokens: torch.Tensor,
    tokenizer: ChoraleTokenizer,
    out_path: str | Path,
    seconds_per_step: float = 0.15,
) -> None:
    """Listening-friendly MIDI renderer.

    Differs from `tokens_to_midi` in ONE way: consecutive (pitch, HOLD,
    HOLD, ...) runs in the same voice are merged into a single sustained
    `pretty_midi.Note` whose duration equals run_length * seconds_per_step.
    This is what you want for A/B listening stimuli — you hear quarter
    notes as quarters, halves as halves, and phrase ends as long tones.

    DO NOT use this for `rule_checker.load_satb()` — the rule checker
    recovers the pitch sequence by reading one note per 16th-note tick
    and depends on `tokens_to_midi`'s one-note-per-step output. Only use
    this renderer for the audio demo + A/B study.

    `seconds_per_step` defaults to 0.15 s — at 16 ticks per bar that is
    ~2.4 s/bar ≈ 100 BPM quarter, a typical chorale tempo. The rule-
    checker renderer uses 0.25 s (60 BPM) because timing doesn't matter
    there.
    """
    # Inline the layout-agnostic prep: strip chord tokens if present, then
    # convert the voice-only stream to a (4, T) pitch grid *without*
    # resolve_holds — we want to keep the HOLD sentinels so we can merge
    # runs below.
    pitch_tokens = _strip_chord_tokens(tokens, tokenizer)
    grid = tokenizer.decode(pitch_tokens)                  # (4, T), has HOLD_RAW
    T = grid.shape[1]

    pm = pretty_midi.PrettyMIDI()
    for v in range(4):
        instr = pretty_midi.Instrument(program=VOICE_PROGRAMS[v], name=VOICE_NAMES[v])
        t = 0
        while t < T:
            p = int(grid[v, t])
            if p == tokenizer.REST_RAW:
                t += 1
                continue
            if p == tokenizer.HOLD_RAW:
                # Dangling HOLD at the start of a voice (no pitch yet) —
                # just skip. resolve_holds would have made it a REST too.
                t += 1
                continue
            # p is a real pitch. Walk forward through HOLDs AND
            # same-pitch repeats — treat a repeated pitch as a sustained
            # note. The tokenizer's training-time encoding collapses
            # "sustain" and "re-articulate same pitch" into the same
            # representation (pitch + HOLDs), so at render time we
            # cannot distinguish them and should render the less-ugly
            # version (one sustained note, not a chopped re-attack).
            start = t
            end = t + 1
            while end < T and int(grid[v, end]) in (tokenizer.HOLD_RAW, int(p)):
                end += 1
            instr.notes.append(pretty_midi.Note(
                velocity=80, pitch=int(p),
                start=start * seconds_per_step,
                end=end * seconds_per_step,
            ))
            t = end
        pm.instruments.append(instr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


# Common Bach-chorale opening chords (SATB, MIDI pitches). All are in keys
# Bach actually uses, in the voice ranges the model was trained on. We prime
# with a real first chord instead of BOS because BOS never appears in training
# sequences (see data/jsb_loader.py::__getitem__), so its embedding stays at
# random init and sampling from it produces garbage that consistently attracts
# to MIDI 82-84. Using a real opener keeps sampling on-distribution.
_COMMON_OPENERS = (
    (72, 67, 64, 48),   # C major:  C5 G4 E4 C3
    (71, 67, 62, 55),   # G major:  B4 G4 D4 G3
    (69, 65, 60, 53),   # F major:  A4 F4 C4 F3
    (74, 69, 65, 57),   # D major:  D5 A4 F4 A3
    (76, 72, 67, 60),   # C major:  E5 C5 G4 C4
    (70, 65, 62, 53),   # F major:  A#4 F4 D4 F3
)


def make_prompt(
    tokenizer: ChoraleTokenizer,
    batch_size: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Real SATB chord per row — NOT a BOS token.

    Returns (batch_size, 4) LongTensor. The model was trained on sequences
    that begin directly with a Soprano pitch at t=0 (no BOS prepended), so
    priming with BOS was off-distribution and caused a collapse attractor
    at MIDI 82-84 across all voices. A real first chord is in-distribution.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch_size):
        chord = _COMMON_OPENERS[int(rng.integers(len(_COMMON_OPENERS)))]
        rows.append([tokenizer.pitch_to_token(int(p)) for p in chord])
    return torch.tensor(rows, dtype=torch.long)
