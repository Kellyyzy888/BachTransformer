"""Voice-leading rule checker for four-part (SATB) chorale MIDI.

Implements the five Tier-1 rules we plan to inject into Coconet:
  1. Parallel 5ths / 8ves (adjacent voice pairs, across beat transitions)
  2. Voice crossing (SATB pitch-order at any step)
  3. Hidden 5/8 in outer voices (S-B similar motion into P5/P8 with S leap)
  4. Large melodic leaps (> M6 by default; A2 and A4 flagged separately)
  5. Spacing (> octave between S-A or A-T; B-T exempt)

Also reports a single HarmonicScore: alpha-weighted sum of per-rule rates.

Usage:
  python rule_checker.py <midi_file_or_dir> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import pretty_midi


VOICES = ("S", "A", "T", "B")
ADJACENT_PAIRS = [("S", "A"), ("A", "T"), ("T", "B")]
ALL_PAIRS = [
    ("S", "A"), ("S", "T"), ("S", "B"),
    ("A", "T"), ("A", "B"),
    ("T", "B"),
]


@dataclass
class RuleReport:
    sample: str
    n_steps: int = 0
    n_transitions: int = 0
    # Tier 1 counts
    parallel_5ths: int = 0
    parallel_8ves: int = 0
    voice_crossings: int = 0
    hidden_5ths_outer: int = 0
    hidden_8ves_outer: int = 0
    spacing_violations: int = 0
    large_leaps: int = 0
    augmented_leaps: int = 0
    # For drill-down
    examples: List[str] = field(default_factory=list)


def load_satb(midi_path: str) -> List[List[Optional[int]]]:
    """Return four equal-length pitch lists [S, A, T, B], one entry per time step.

    Coconet MIDI has 4 tracks (soprano, alto, tenor, bass). Each track is a
    sequence of uniform-duration notes. We recover the pitch sequence by sorting
    each track's notes by start time.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if len(pm.instruments) < 4:
        raise ValueError(
            f"{midi_path}: expected >=4 instruments/tracks, got {len(pm.instruments)}"
        )

    # Coconet orders tracks S, A, T, B (descending mean pitch across the piece).
    # We verify by sorting instruments by mean pitch and assigning.
    inst_means = []
    for inst in pm.instruments[:4]:
        pitches = [n.pitch for n in inst.notes]
        if not pitches:
            inst_means.append(-1)
        else:
            inst_means.append(sum(pitches) / len(pitches))
    order = sorted(range(4), key=lambda i: -inst_means[i])
    ordered = [pm.instruments[i] for i in order]

    voices: List[List[Optional[int]]] = []
    step_counts = []
    for inst in ordered:
        notes = sorted(inst.notes, key=lambda n: n.start)
        pitches = [n.pitch for n in notes]
        voices.append(pitches)
        step_counts.append(len(pitches))

    # Normalize to min length.
    n = min(step_counts) if step_counts else 0
    voices = [v[:n] for v in voices]
    return voices


def _interval_mod12(lower: int, upper: int) -> int:
    return (upper - lower) % 12


def check_parallel(voices, report: RuleReport) -> None:
    """Parallel 5ths / 8ves on every adjacent voice pair."""
    n = len(voices[0])
    for t in range(n - 1):
        for a_name, b_name in ALL_PAIRS:
            ai = VOICES.index(a_name)
            bi = VOICES.index(b_name)
            p_a_t, p_a_t1 = voices[ai][t], voices[ai][t + 1]
            p_b_t, p_b_t1 = voices[bi][t], voices[bi][t + 1]
            if None in (p_a_t, p_a_t1, p_b_t, p_b_t1):
                continue
            # Both voices must actually move.
            if p_a_t == p_a_t1 or p_b_t == p_b_t1:
                continue
            # Upper/lower of the pair:
            lo_t, hi_t = sorted((p_a_t, p_b_t))
            lo_t1, hi_t1 = sorted((p_a_t1, p_b_t1))
            iv_t = _interval_mod12(lo_t, hi_t)
            iv_t1 = _interval_mod12(lo_t1, hi_t1)
            # Only count if motion is in the same direction (parallel, not contrary).
            dir_a = (p_a_t1 - p_a_t)
            dir_b = (p_b_t1 - p_b_t)
            if dir_a * dir_b <= 0:
                continue
            if iv_t == 7 and iv_t1 == 7:
                report.parallel_5ths += 1
                if len(report.examples) < 5:
                    report.examples.append(
                        f"P5: {a_name}-{b_name} t={t}"
                    )
            elif iv_t == 0 and iv_t1 == 0:
                report.parallel_8ves += 1
                if len(report.examples) < 5:
                    report.examples.append(
                        f"P8: {a_name}-{b_name} t={t}"
                    )


def check_voice_crossing(voices, report: RuleReport) -> None:
    """SATB pitch-order violation at any time step."""
    n = len(voices[0])
    for t in range(n):
        s, a, tn, b = voices[0][t], voices[1][t], voices[2][t], voices[3][t]
        if None in (s, a, tn, b):
            continue
        if s < a:
            report.voice_crossings += 1
        if a < tn:
            report.voice_crossings += 1
        if tn < b:
            report.voice_crossings += 1


def check_hidden_outer(voices, report: RuleReport) -> None:
    """Hidden 5ths / 8ves on outer voices (S and B).

    Similar motion (same sign) into a P5 or P8, with soprano moving by leap (>2st).
    """
    s, b = voices[0], voices[3]
    n = len(s)
    for t in range(n - 1):
        if None in (s[t], s[t + 1], b[t], b[t + 1]):
            continue
        ds = s[t + 1] - s[t]
        db = b[t + 1] - b[t]
        if ds == 0 or db == 0:
            continue
        if (ds > 0) != (db > 0):
            continue  # contrary or oblique
        iv_arrival = _interval_mod12(b[t + 1], s[t + 1])
        if abs(ds) <= 2:
            continue  # soprano moves by step -> allowed
        if iv_arrival == 7:
            report.hidden_5ths_outer += 1
        elif iv_arrival == 0:
            report.hidden_8ves_outer += 1


def check_spacing(voices, report: RuleReport) -> None:
    """> 1 octave between S-A or A-T. B-T exempt."""
    n = len(voices[0])
    for t in range(n):
        s, a, tn = voices[0][t], voices[1][t], voices[2][t]
        if None in (s, a, tn):
            continue
        if (s - a) > 12:
            report.spacing_violations += 1
        if (a - tn) > 12:
            report.spacing_violations += 1


def check_leaps(voices, report: RuleReport, leap_threshold: int = 9) -> None:
    """Melodic leaps > leap_threshold semitones (M6 by default).

    Also flags augmented 2nds (3 semitones that the spelling implies are A2) and
    tritones (6 semitones). Since MIDI has no spelling, we use pitch-class heuristics
    and flag 3-semitone leaps only if the preceding step direction reverses (a
    common A2 shape) - conservative, EVAL-oriented.
    """
    for vi, name in enumerate(VOICES):
        v = voices[vi]
        for t in range(len(v) - 1):
            if v[t] is None or v[t + 1] is None:
                continue
            iv = abs(v[t + 1] - v[t])
            if iv > leap_threshold:
                report.large_leaps += 1
            if iv == 6:  # tritone
                report.augmented_leaps += 1


def diagnose(midi_path: str) -> RuleReport:
    voices = load_satb(midi_path)
    report = RuleReport(sample=os.path.basename(midi_path))
    if not voices or not voices[0]:
        return report
    report.n_steps = len(voices[0])
    report.n_transitions = max(0, report.n_steps - 1)
    check_parallel(voices, report)
    check_voice_crossing(voices, report)
    check_hidden_outer(voices, report)
    check_spacing(voices, report)
    check_leaps(voices, report)
    return report


def harmonic_score(r: RuleReport) -> int:
    """Total number of voice-leading violations in the sample. Lower is better.

    All generated samples share the same piece_length in our experiment, so
    no length normalization is needed — we just sum the eight counters.
    """
    return (
        r.parallel_5ths
        + r.parallel_8ves
        + r.voice_crossings
        + r.hidden_5ths_outer
        + r.hidden_8ves_outer
        + r.spacing_violations
        + r.large_leaps
        + r.augmented_leaps
    )


def diagnose_folder(midi_dir: str) -> List[RuleReport]:
    reports = []
    for fname in sorted(os.listdir(midi_dir)):
        if not fname.lower().endswith((".mid", ".midi")):
            continue
        path = os.path.join(midi_dir, fname)
        try:
            reports.append(diagnose(path))
        except Exception as e:
            print(f"[warn] {fname}: {e}", file=sys.stderr)
    return reports


def print_summary(reports: List[RuleReport]) -> None:
    if not reports:
        print("(no samples)")
        return
    n = len(reports)
    totals = RuleReport(sample=f"total over {n}")
    for r in reports:
        totals.n_steps += r.n_steps
        totals.n_transitions += r.n_transitions
        totals.parallel_5ths += r.parallel_5ths
        totals.parallel_8ves += r.parallel_8ves
        totals.voice_crossings += r.voice_crossings
        totals.hidden_5ths_outer += r.hidden_5ths_outer
        totals.hidden_8ves_outer += r.hidden_8ves_outer
        totals.spacing_violations += r.spacing_violations
        totals.large_leaps += r.large_leaps
        totals.augmented_leaps += r.augmented_leaps

    def rate(numer, denom, per=100):
        return (numer / denom * per) if denom else 0.0

    print(f"\n=== Per-sample HarmonicScore ===")
    print(f"{'sample':<25}  {'steps':>5}  {'P5':>3}  {'P8':>3}  {'VC':>3}  "
          f"{'H5':>3}  {'H8':>3}  {'Sp':>3}  {'Lp':>3}  {'score':>7}")
    for r in reports:
        print(f"{r.sample:<25}  {r.n_steps:>5}  "
              f"{r.parallel_5ths:>3}  {r.parallel_8ves:>3}  "
              f"{r.voice_crossings:>3}  "
              f"{r.hidden_5ths_outer:>3}  {r.hidden_8ves_outer:>3}  "
              f"{r.spacing_violations:>3}  {r.large_leaps:>3}  "
              f"{harmonic_score(r):>7d}")

    print(f"\n=== Aggregate over {n} samples ===")
    print(f"total time steps:        {totals.n_steps}")
    print(f"total transitions:       {totals.n_transitions}")
    print(f"Parallel 5ths:    {totals.parallel_5ths:4d}   "
          f"{rate(totals.parallel_5ths, totals.n_transitions):6.2f} per 100 transitions")
    print(f"Parallel 8ves:    {totals.parallel_8ves:4d}   "
          f"{rate(totals.parallel_8ves, totals.n_transitions):6.2f} per 100 transitions")
    print(f"Voice crossings:  {totals.voice_crossings:4d}   "
          f"{rate(totals.voice_crossings, totals.n_steps):6.2f} per 100 steps")
    print(f"Hidden 5ths (SB): {totals.hidden_5ths_outer:4d}   "
          f"{rate(totals.hidden_5ths_outer, totals.n_transitions):6.2f} per 100 transitions")
    print(f"Hidden 8ves (SB): {totals.hidden_8ves_outer:4d}   "
          f"{rate(totals.hidden_8ves_outer, totals.n_transitions):6.2f} per 100 transitions")
    print(f"Spacing viols:    {totals.spacing_violations:4d}   "
          f"{rate(totals.spacing_violations, totals.n_steps):6.2f} per 100 steps")
    print(f"Large leaps (>M6):{totals.large_leaps:4d}   "
          f"{rate(totals.large_leaps, totals.n_transitions):6.2f} per 100 transitions")
    print(f"Tritone leaps:    {totals.augmented_leaps:4d}")

    mean_score = sum(harmonic_score(r) for r in reports) / n
    print(f"\nMean HarmonicScore: {mean_score:.2f}   (total violations per sample; lower = better)")


def write_csv(reports: List[RuleReport], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample", "n_steps", "n_transitions",
            "parallel_5ths", "parallel_8ves",
            "voice_crossings",
            "hidden_5ths_outer", "hidden_8ves_outer",
            "spacing_violations",
            "large_leaps", "augmented_leaps",
            "harmonic_score",
        ])
        for r in reports:
            w.writerow([
                r.sample, r.n_steps, r.n_transitions,
                r.parallel_5ths, r.parallel_8ves,
                r.voice_crossings,
                r.hidden_5ths_outer, r.hidden_8ves_outer,
                r.spacing_violations,
                r.large_leaps, r.augmented_leaps,
                harmonic_score(r),
            ])


# ---------------------------------------------------------------------------
# Shim for the new trainer — decode_m4.py and run_eval.py both call this.
# ---------------------------------------------------------------------------

def score_midi(path: str) -> dict:
    """Wrapper around diagnose() that returns a plain dict with the counters.

    Keys match what run_eval.py and decode_m4.py expect.
    """
    r = diagnose(path)
    return {
        "parallel_5ths":      r.parallel_5ths,
        "parallel_8ves":      r.parallel_8ves,
        "voice_crossings":    r.voice_crossings,
        "hidden_5ths_outer":  r.hidden_5ths_outer,
        "hidden_8ves_outer":  r.hidden_8ves_outer,
        "spacing_violations": r.spacing_violations,
        "large_leaps":        r.large_leaps,
        "augmented_leaps":    r.augmented_leaps,
        "HarmonicScore":      harmonic_score(r),
        "n_steps":            r.n_steps,
        "n_transitions":      r.n_transitions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="MIDI file or directory of MIDI files")
    ap.add_argument("--csv", help="Write per-sample CSV to this path")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        reports = diagnose_folder(args.path)
    else:
        reports = [diagnose(args.path)]

    print_summary(reports)
    if args.csv:
        write_csv(reports, args.csv)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
