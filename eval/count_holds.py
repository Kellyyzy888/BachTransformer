"""Diagnostic: count per-voice note attacks and durations on a MIDI folder.

Counts everything in units of 16th-notes, independent of the tempo used
to render the MIDI. Works correctly for BOTH the rule-checker render
(seconds_per_step=0.25) and the listen render (seconds_per_step=0.15).

How: we assume every piece in the folder has `ticks_per_piece` 16th-note
timesteps (default 64 — matches `configs/base.yaml:sample.piece_length`).
We detect the per-piece tempo by dividing pm.get_end_time() by ticks_per_piece
and report attacks/bar and mean note length in the true 16th-note grid.

Usage:
    python3 -m eval.count_holds samples/m4_m3_uniform
    python3 -m eval.count_holds samples/listen_hold --ticks 64
"""
import argparse
from pathlib import Path

import pretty_midi

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--ticks", type=int, default=64,
                help="expected 16th-note ticks per piece (default: 64)")
args = ap.parse_args()

folder = Path(args.folder)
ticks_per_piece = args.ticks
ticks_per_bar = 16

total_pieces = 0
total_attacks = 0
attacks_per_voice = [0, 0, 0, 0]
note_lengths_ticks = []     # in real 16th-note units

for mid_path in sorted(folder.glob("*.mid")):
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    end_time = pm.get_end_time()
    if end_time <= 0:
        continue
    # infer this piece's seconds-per-16th from its own end time
    step_s = end_time / ticks_per_piece
    total_pieces += 1
    for v, instr in enumerate(pm.instruments[:4]):
        for note in instr.notes:
            attacks_per_voice[v] += 1
            total_attacks += 1
            dur_ticks = (note.end - note.start) / step_s
            note_lengths_ticks.append(dur_ticks)

total_bars = total_pieces * ticks_per_piece / ticks_per_bar
voice_bars = total_bars * 4

print(f"files: {total_pieces}")
print(f"total 16th-note ticks: {total_pieces * ticks_per_piece}")
print(f"total note attacks: {total_attacks}")
print(f"attacks per voice (S/A/T/B): {attacks_per_voice}")
if voice_bars > 0:
    print(f"mean attacks/bar/voice: {total_attacks / voice_bars:.2f}   "
          f"(Bach typical: ~2.5 — quarters+occasional 8ths; 16 = every 16th)")
if note_lengths_ticks:
    avg = sum(note_lengths_ticks) / len(note_lengths_ticks)
    print(f"mean note length: {avg:.2f} × 16th   "
          f"(1.0 = every note is a 16th; 4.0 = quarter notes)")
    # HOLD emission fraction: if mean note length is L ticks, there are
    # (L-1) HOLD ticks per attack, so HOLD fraction = (L-1)/L.
    hold_frac = max(0.0, (avg - 1.0) / avg)
    print(f"implied HOLD fraction: {hold_frac:.2%}   "
          f"(Bach training data: ~50%; 0% = never held)")
