"""
Per-voice pitch/duration diagnostics for sampled output.

Run on OSCAR from /users/zyang188/bach_transformer/:
    python sample_diagnostics.py

Auto-detects sample directories. If detection fails, pass them explicitly:
    python sample_diagnostics.py samples/m1 samples/m2 samples/m3 samples/m4
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------- config ----------
CANDIDATE_DIRS = {
    "M1": ["samples/m1", "samples_m1", "outputs/m1", "out/m1"],
    "M2": ["samples/m2", "samples_m2", "outputs/m2", "out/m2"],
    "M3": ["samples/m3", "samples_m3", "outputs/m3", "out/m3"],
    "M4": ["samples/m4", "samples_m4", "outputs/m4", "out/m4"],
}
VOICES = ["S", "A", "T", "B"]


# ---------- loaders ----------
def load_tokens_from_pt(path: Path) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(obj, dict):
        for k in ("tokens", "sample", "ids", "x"):
            if k in obj:
                return np.asarray(obj[k])
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def load_tokens_from_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def load_midi_pitches(path: Path) -> np.ndarray | None:
    """Return (T, 4) int array of MIDI pitches per voice, or None."""
    try:
        import pretty_midi
    except ImportError:
        return None
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None
    if len(pm.instruments) < 4:
        return None
    # sample at 16th-note grid for ~64 steps
    total_end = max(inst.get_end_time() for inst in pm.instruments[:4])
    n_steps = 64
    dt = total_end / n_steps if total_end > 0 else 0.25
    grid = np.arange(n_steps) * dt + dt / 2
    out = np.full((n_steps, 4), -1, dtype=np.int32)
    for v, inst in enumerate(pm.instruments[:4]):
        for note in inst.notes:
            mask = (grid >= note.start) & (grid < note.end)
            out[mask, v] = int(note.pitch)
    return out


def load_sample(path: Path) -> np.ndarray | None:
    if path.suffix == ".pt":
        arr = load_tokens_from_pt(path)
    elif path.suffix == ".npy":
        arr = load_tokens_from_npy(path)
    elif path.suffix in (".mid", ".midi"):
        arr = load_midi_pitches(path)
    else:
        return None
    if arr is None:
        return None
    # normalize to (T, 4)
    if arr.ndim == 1:
        # probably packed SATB: reshape (T*4,) -> (T, 4)
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ---------- diagnostics ----------
def summarize_condition(cond: str, paths: list[Path]) -> None:
    print(f"\n=== {cond} ({len(paths)} samples from {paths[0].parent}) ===")
    samples = []
    for p in sorted(paths)[:100]:
        arr = load_sample(p)
        if arr is None or arr.ndim != 2:
            continue
        samples.append(arr)
    if not samples:
        print("  no loadable samples")
        return

    # Stack to (N, T, V); truncate T to min across samples
    min_T = min(s.shape[0] for s in samples)
    stacked = np.stack([s[:min_T, :4] for s in samples], axis=0)  # (N, T, 4)
    N, T, V = stacked.shape
    print(f"  shape: N={N} samples, T={T} steps, V={V} voices")
    print(f"  token value range: min={int(stacked.min())}, max={int(stacked.max())}")

    for v in range(V):
        flat = stacked[:, :, v].ravel()
        unique = np.unique(flat)
        # consecutive repetition: fraction of adjacent-timestep pairs that match
        rep = np.mean(stacked[:, 1:, v] == stacked[:, :-1, v])
        # entropy over token distribution
        counts = Counter(flat.tolist())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        # most common 3 tokens
        top = counts.most_common(3)
        print(
            f"  voice {VOICES[v]}: "
            f"unique={len(unique):3d}  "
            f"repeat_rate={rep:.2f}  "
            f"entropy={entropy:.2f} bits  "
            f"top3_tokens={[(t, round(c/total, 2)) for t, c in top]}"
        )


def find_dirs() -> dict[str, Path]:
    found = {}
    for cond, candidates in CANDIDATE_DIRS.items():
        for rel in candidates:
            p = Path(rel)
            if p.exists() and any(p.iterdir()):
                found[cond] = p
                break
    return found


def main() -> None:
    if len(sys.argv) > 1:
        args = [Path(a) for a in sys.argv[1:]]
        dirs = dict(zip(["M1", "M2", "M3", "M4"], args))
    else:
        dirs = find_dirs()

    if not dirs:
        print("Could not auto-detect sample directories. Tried:")
        for cond, cands in CANDIDATE_DIRS.items():
            print(f"  {cond}: {cands}")
        print("\nPass them as args: python sample_diagnostics.py <m1_dir> <m2_dir> <m3_dir> <m4_dir>")
        sys.exit(1)

    print("Found sample directories:")
    for cond, d in dirs.items():
        exts = Counter(p.suffix for p in d.iterdir())
        print(f"  {cond}: {d}  ({dict(exts)})")

    for cond, d in dirs.items():
        # pick any sample files
        paths = [p for p in d.iterdir() if p.suffix in (".pt", ".npy", ".mid", ".midi")]
        if not paths:
            print(f"\n  {cond}: no .pt/.npy/.mid files in {d}")
            continue
        summarize_condition(cond, paths)


if __name__ == "__main__":
    main()
