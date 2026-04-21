"""Roman-numeral chord extraction for JSB chorales (M4 tokenizer input).

Pipeline
--------
1. Load the Boulanger-Lewandowski JSB pickle. Each split is a list of
   chorales; each chorale is a list of timesteps; each timestep is a
   tuple of MIDI pitch ints (variable arity — typically 4 voices, but
   sometimes fewer on fermatas/rests).

2. For each chorale:
     a. Pool every pitch in the chorale into a music21 stream and call
        `.analyze('key')` (Krumhansl-Schmuckler by default) to get a
        global tonic + mode. This is cheap (pitch-class histogram) and
        robust enough for Bach chorales, which almost never modulate
        far enough to fool the correlation.
     b. Walk the chorale timestep-by-timestep. For each timestep, build
        `chord.Chord(pitches)` and call `roman.romanNumeralFromChord`
        relative to the detected key. Strip inversion to keep vocab
        small — we re-enable inversions later only if the ablation
        asks for it.
     c. Emit the scale-degree figure (e.g. "I", "V", "vii°", "iv",
        "Cad64") as the RN string for that timestep.

3. Edge cases fall back to sentinel symbols:
     - Empty / all-rest timestep    -> "RN_REST"
     - music21 error / unparseable  -> "RN_OTHER"
   These two become dedicated tokens in the extended tokenizer, so the
   model can treat "we don't have a chord here" explicitly instead of
   picking an arbitrary fallback RN.

4. Serialize `{split: [(tonic, mode, [rn_str, ...]), ...]}` to pickle.
   Print vocab statistics (unique RN symbols, coverage of top-30,
   counts of sentinel tokens) so we can decide whether to collapse
   further before wiring into the tokenizer.

Usage
-----
    python -m data.chord_extractor \
        --in  JSB-Chorales-dataset/jsb-chorales-16th.pkl \
        --out data/jsb_chords.pkl

Requires music21 (`pip install music21`).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

# music21 is noisy on import. Suppress the deprecation warnings from its
# internal corpus loading so the vocab-stats output is readable.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# RN normalization
# ---------------------------------------------------------------------------

# Sentinel RN strings we emit for non-extractable moments.
RN_REST = "RN_REST"       # every voice rests this timestep
RN_OTHER = "RN_OTHER"     # unparseable (music21 raised, or non-tertian)


def _normalize_rn_figure(rn) -> str:
    """Strip inversion / extensions from a music21 RomanNumeral.

    Keep: the scale-degree letters + quality (I, i, V, v, vii°, III+).
    Drop: inversion digits (65, 43, 42), added tones (V7, V9),
          cadential 6/4 marker — collapse to the underlying triad.

    music21's `.romanNumeral` property is the scale-degree string
    alone (no figure extensions). That's what we want as a starting
    vocab. We wrap it in quality symbols so major/minor V are distinct:

        rn.romanNumeral       -> "V", "vii"
        rn.quality            -> "major", "minor", "diminished", "augmented"

    We collapse by using `rn.romanNumeral` (which already carries case-
    based quality for most cases) and append unicode markers for the
    two cases case can't express:
        diminished  -> append "°"
        augmented   -> append "+"
    """
    figure = rn.romanNumeral
    q = rn.quality
    if q == "diminished" and "°" not in figure:
        figure = figure + "°"
    elif q == "augmented" and "+" not in figure:
        figure = figure + "+"
    return figure


# ---------------------------------------------------------------------------
# Per-chorale extraction
# ---------------------------------------------------------------------------

def _detect_key(chorale_pitches: Iterable[int]):
    """Return a music21 Key object for this chorale.

    We flatten every pitch in the chorale into a single Stream and let
    music21 run Krumhansl-Schmuckler on it. Bach chorales rarely modulate
    far enough that a global key is misleading, and for our RN vocabulary
    what matters is consistency — the same chord shape maps to the same
    RN label across the corpus.
    """
    from music21 import note, stream

    s = stream.Stream()
    for p in chorale_pitches:
        if p is None or p < 0:
            continue
        s.append(note.Note(midi=int(p), quarterLength=0.25))
    if len(s.notes) == 0:
        # Fallback to C major — shouldn't happen on real chorales but
        # guards against degenerate inputs.
        from music21 import key as m21key
        return m21key.Key("C")
    return s.analyze("key")


def _rn_for_timestep(pitches: tuple[int, ...], key) -> str:
    """Return the Roman-numeral string for one 16th-note timestep.

    `pitches` is a tuple of MIDI ints (possibly empty, possibly with
    fewer than 4 voices).
    """
    from music21 import chord, roman

    clean = [int(p) for p in pitches if p is not None and int(p) >= 0]
    if len(clean) == 0:
        return RN_REST

    try:
        c = chord.Chord(clean)
        rn = roman.romanNumeralFromChord(c, key)
    except Exception:
        return RN_OTHER

    try:
        return _normalize_rn_figure(rn)
    except Exception:
        return RN_OTHER


def extract_chorale(chorale: list[tuple[int, ...]]) -> tuple[str, str, list[str]]:
    """Extract (tonic_name, mode, [rn_str, ...]) for a single chorale.

    `tonic_name` is like "C", "F#", "Bb"; `mode` is "major" or "minor".
    The RN list has one entry per 16th-note timestep (same length as
    the input chorale).
    """
    # Flatten all pitches for key detection.
    all_pitches = []
    for step in chorale:
        for p in step:
            if p is None:
                continue
            pi = int(p)
            if pi >= 0:
                all_pitches.append(pi)

    key = _detect_key(all_pitches)
    tonic_name = key.tonic.name            # "C", "F#", etc.
    mode = key.mode                         # "major" / "minor"

    rns = [_rn_for_timestep(step, key) for step in chorale]
    return tonic_name, mode, rns


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _print_progress(i: int, n: int, every: int = 10) -> None:
    if i == 0 or (i + 1) % every == 0 or i == n - 1:
        print(f"    {i+1:4d}/{n}", file=sys.stderr)


def extract_all(
    jsb_pickle: str | Path,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict:
    """Run RN extraction on every chorale in every split."""
    blob = _load_pickle(jsb_pickle)
    out: dict[str, list[tuple[str, str, list[str]]]] = {}

    # the pickle may use 'valid' or 'val'
    alias = {"val": "valid", "valid": "val"}

    for split in splits:
        if split not in blob and alias.get(split) in blob:
            src = alias[split]
        elif split in blob:
            src = split
        else:
            print(f"  [warn] split {split!r} not found, skipping", file=sys.stderr)
            continue
        print(f"  extracting split={src} ({len(blob[src])} chorales)", file=sys.stderr)
        extracted: list[tuple[str, str, list[str]]] = []
        for i, chorale in enumerate(blob[src]):
            extracted.append(extract_chorale(chorale))
            _print_progress(i, len(blob[src]))
        out[split] = extracted
    return out


def print_vocab_stats(extracted: dict) -> list[str]:
    """Print vocab statistics across all splits; return sorted vocab list."""
    counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    total_steps = 0
    for split, chorales in extracted.items():
        for tonic, mode, rns in chorales:
            counts.update(rns)
            key_counts[f"{tonic} {mode}"] += 1
            total_steps += len(rns)

    vocab = sorted(counts.keys())
    print()
    print(f"RN vocab size: {len(vocab)} unique symbols across {total_steps} timesteps")
    print(f"RN_REST:  {counts[RN_REST]:>6} ({100*counts[RN_REST]/total_steps:.2f}%)")
    print(f"RN_OTHER: {counts[RN_OTHER]:>6} ({100*counts[RN_OTHER]/total_steps:.2f}%)")
    print()
    print("Top 30 RN symbols:")
    top30 = counts.most_common(30)
    covered = sum(c for _, c in top30)
    for rn, c in top30:
        print(f"  {rn:<10} {c:>6}  ({100*c/total_steps:.2f}%)")
    print(f"  top-30 coverage: {100*covered/total_steps:.2f}%")
    print()
    print(f"Unique detected keys: {len(key_counts)}")
    print("Top 10 keys:")
    for k, c in key_counts.most_common(10):
        print(f"  {k:<12} {c:>4}")
    return vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="path to JSB pickle (e.g. jsb-chorales-16th.pkl)")
    ap.add_argument("--out", dest="out_path", default="data/jsb_chords.pkl",
                    help="where to write the (key, RN) cache pickle")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: only process first N chorales per split "
                         "(for quick vocab stats)")
    args = ap.parse_args()

    try:
        import music21  # noqa: F401
    except ImportError:
        print("music21 not installed. Run: pip install music21", file=sys.stderr)
        sys.exit(1)

    print(f"Loading JSB from {args.in_path}", file=sys.stderr)
    blob = _load_pickle(args.in_path)

    extracted: dict = {}
    alias = {"val": "valid", "valid": "val"}
    for split in args.splits:
        src = split if split in blob else alias.get(split)
        if src is None or src not in blob:
            print(f"  [warn] split {split!r} not found, skipping", file=sys.stderr)
            continue
        chorales = blob[src]
        if args.limit is not None:
            chorales = chorales[: args.limit]
        print(f"  extracting split={src} ({len(chorales)} chorales)", file=sys.stderr)
        ex: list[tuple[str, str, list[str]]] = []
        for i, chorale in enumerate(chorales):
            ex.append(extract_chorale(chorale))
            _print_progress(i, len(chorales))
        extracted[split] = ex

    vocab = print_vocab_stats(extracted)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extracted": extracted,   # {split: [(tonic, mode, [rn, ...]), ...]}
        "vocab": vocab,            # sorted list of unique RN strings
        "sentinels": {"RN_REST": RN_REST, "RN_OTHER": RN_OTHER},
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
