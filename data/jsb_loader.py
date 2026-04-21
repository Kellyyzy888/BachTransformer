"""JSB Chorales PyTorch Dataset.

The standard Boulanger-Lewandowski split is shipped as a pickled dict:
    {'train': [...], 'valid': [...], 'test': [...]}
where each split is a list of chorales, and each chorale is a list of
timesteps, each timestep is a tuple of MIDI pitches (variable arity, but
typically 4 for the SATB voices).

Sources:
    https://github.com/czhuang/JSB-Chorales-dataset
    http://www-etud.iro.umontreal.ca/~boulanni/icml2012   (original release)

Run `python -m data.jsb_loader --download` to fetch and cache the pickle.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import ChoraleTokenizer, TokenizerConfig, _tonic_to_pc


# Default augmentation shift range: the symmetric interval {-5, ..., +6}
# gives us exactly 12 transpositions per source chorale without duplicate
# pitch classes (shift 0 and shift 12 would land on the same pitch-class
# sequence). The chosen span keeps most voices inside the tokenizer's
# [C2, C6] pitch window for typical JSB chorales, which tend to sit in
# the middle of that window.
DEFAULT_SHIFTS = list(range(-5, 7))           # 12 shifts, inclusive


# ---------------------------------------------------------------------------

def _normalize_chorale(raw: list[tuple[int, ...]], tok: ChoraleTokenizer) -> np.ndarray:
    """Convert a list-of-timesteps chorale to a (4, T) MIDI pitch array.

    - If a timestep has fewer than 4 pitches, pad with REST_RAW.
    - If more than 4, drop the extras (rare; usually a duplicated pitch).
    - Sort each timestep ascending so voices map to (S high ... B low) by
      reversing.
    - Clip to tokenizer pitch range; out-of-range pitches become REST.
    - Insert HOLD where a voice's pitch repeats from the previous timestep.
      We bring HOLD back (after briefly removing it 2026-04-20) because
      removing HOLD just swapped one shortcut for another: with raw
      repeated pitches, "predict the same-voice token from 4 positions
      ago" was 78.6% accurate (CE 0.241 nats), and the model collapsed
      onto that copy-prior shortcut instead. The real fix lives in
      train_m1.py — class-weighted CE that down-weights HOLD heavily so
      the model can't get free loss off either trivial pattern.
    """
    T = len(raw)
    out = np.full((4, T), tok.REST_RAW, dtype=np.int64)
    pmin, pmax = tok.cfg.pitch_min, tok.cfg.pitch_max

    for t, chord in enumerate(raw):
        pitches = sorted(int(p) for p in chord)
        # take the highest 4 if more than 4, lowest-padded if fewer
        if len(pitches) >= 4:
            pitches = pitches[-4:]
        else:
            pitches = [tok.REST_RAW] * (4 - len(pitches)) + pitches
        # SATB ordering: S highest -> B lowest
        pitches = list(reversed(pitches))
        for v, p in enumerate(pitches):
            if p == tok.REST_RAW:
                out[v, t] = tok.REST_RAW
            elif pmin <= p <= pmax:
                out[v, t] = p
            else:
                out[v, t] = tok.REST_RAW

    # mark holds (repeated pitches in the same voice across consecutive steps)
    for v in range(4):
        for t in range(1, T):
            if out[v, t] == out[v, t - 1] and out[v, t] not in (tok.REST_RAW,):
                out[v, t] = tok.HOLD_RAW
    return out


# ---------------------------------------------------------------------------

def _shift_chorale(arr: np.ndarray, shift: int, tok: ChoraleTokenizer) -> np.ndarray:
    """Transpose a (4, T) chorale by `shift` semitones.

    Out-of-range pitches after transposition are clamped by octave shifts
    *within* the voice — a C3 that would become B2 after shift -1 is left
    alone rather than moved to a different octave, because the rule loss
    / MIDI rendering depends on exact pitches. If the shift would push a
    pitch outside [pitch_min, pitch_max], we fall back to REST.
    Sentinel values (HOLD_RAW, REST_RAW) are preserved unchanged.
    """
    out = arr.copy()
    pmin, pmax = tok.cfg.pitch_min, tok.cfg.pitch_max
    for v in range(arr.shape[0]):
        for t in range(arr.shape[1]):
            p = int(arr[v, t])
            if p == tok.HOLD_RAW or p == tok.REST_RAW:
                continue
            np_ = p + shift
            if not (pmin <= np_ <= pmax):
                out[v, t] = tok.REST_RAW
            else:
                out[v, t] = np_
    return out


def _shift_key(tonic_name: str, shift: int) -> str:
    """Shift a tonic name by `shift` semitones. Mode is unchanged."""
    from .tokenizer import PITCH_CLASS_NAMES
    pc = (_tonic_to_pc(tonic_name) + shift) % 12
    return PITCH_CLASS_NAMES[pc]


# ---------------------------------------------------------------------------

class JSBChorales(Dataset):
    """SATB-packed JSB dataset (M1 / M1-aug).

    When `chord_cache_path` is set, the dataset additionally emits
    interleaved [KEY, RN, S, A, T, B, ...] sequences per chunk — the
    M4 training signal. In that mode the tokenizer must have been
    constructed with `chord_vocab=...`.
    """

    def __init__(
        self,
        pickle_path: str | Path,
        split: str = "train",
        piece_length: int = 64,
        stride: int = 32,
        tokenizer: ChoraleTokenizer | None = None,
        # --- M4 additions ------------------------------------------------
        chord_cache_path: str | Path | None = None,
        transpose_shifts: list[int] | None = None,
    ):
        super().__init__()
        with open(pickle_path, "rb") as f:
            try:
                blob = pickle.load(f)
            except UnicodeDecodeError:
                # The canonical B-L pickle was written under Python 2.
                f.seek(0)
                blob = pickle.load(f, encoding="latin1")
        if split not in blob:
            # the original pickle uses "valid" not "val"
            alt = {"val": "valid", "valid": "val"}.get(split)
            if alt and alt in blob:
                split = alt
            else:
                raise KeyError(f"split {split!r} not in pickle (have {list(blob)})")

        self.tok = tokenizer or ChoraleTokenizer()
        self.piece_length = piece_length
        self.stride = stride
        self.chord_mode = chord_cache_path is not None

        # Only augment training data. Caller passes `transpose_shifts=None`
        # (or [0]) for val/test.
        self.shifts: list[int] = (
            list(transpose_shifts) if transpose_shifts is not None else [0]
        )

        # Load chord cache if requested.
        chord_data_by_split: list[tuple[str, str, list[str]]] | None = None
        if self.chord_mode:
            with open(chord_cache_path, "rb") as f:
                chord_payload = pickle.load(f)
            chord_all = chord_payload["extracted"]
            key = split if split in chord_all else {"val": "valid", "valid": "val"}.get(split)
            if key is None or key not in chord_all:
                raise KeyError(f"chord cache missing split {split!r}")
            chord_data_by_split = chord_all[key]
            if len(chord_data_by_split) != len(blob[split]):
                raise ValueError(
                    f"chord cache has {len(chord_data_by_split)} chorales but "
                    f"JSB pickle has {len(blob[split])} for split {split!r}"
                )

        # (chunk_arr, chunk_rns or None, tonic_name, mode)
        self.chunks: list[tuple[np.ndarray, list[str] | None, str, str]] = []
        for ci, chorale in enumerate(blob[split]):
            arr = _normalize_chorale(chorale, self.tok)          # (4, T_full)
            rns_full: list[str] | None = None
            tonic_name, mode = "C", "major"
            if self.chord_mode:
                tonic_name, mode, rns_full = chord_data_by_split[ci]

            T = arr.shape[1]
            # Produce chunks (in the *untransposed* frame) first; we apply
            # shifts on-the-fly in __getitem__ so we don't blow up memory.
            if T < piece_length:
                pad = np.full((4, piece_length - T), self.tok.REST_RAW, dtype=np.int64)
                arr_p = np.concatenate([arr, pad], axis=1)
                rns_p: list[str] | None = None
                if rns_full is not None:
                    rns_p = list(rns_full) + ["RN_REST"] * (piece_length - T)
                self.chunks.append((arr_p, rns_p, tonic_name, mode))
                continue
            for start in range(0, T - piece_length + 1, stride):
                sub = arr[:, start : start + piece_length]
                sub_rns = (
                    rns_full[start : start + piece_length]
                    if rns_full is not None else None
                )
                self.chunks.append((sub, sub_rns, tonic_name, mode))

        self._n_base = len(self.chunks)
        self._n_shifts = len(self.shifts)

    def __len__(self) -> int:
        return self._n_base * self._n_shifts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = idx // self._n_shifts
        shift = self.shifts[idx % self._n_shifts]
        chorale, rns, tonic_name, mode = self.chunks[base]

        if shift != 0:
            chorale = _shift_chorale(chorale, shift, self.tok)
            tonic_name = _shift_key(tonic_name, shift)
            # RN list is key-relative, so it stays invariant under
            # transposition — the tonic changed but the degrees didn't.

        if self.chord_mode:
            tokens = self.tok.encode_with_chords(
                chorale, rns, tonic_name, mode
            )
        else:
            tokens = self.tok.encode(chorale)

        return {
            "input_ids": tokens[:-1],
            "target_ids": tokens[1:],
        }


# ---------------------------------------------------------------------------

def make_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders for M1 or M4.

    If `cfg["chord"]["enabled"]` is set and a chord cache is on disk, we
    build the M4 chord-conditioned dataset. Otherwise we build the plain
    SATB-packed M1 dataset. In both cases, training-time transposition is
    controlled by `cfg["chord"]["transpose_augment"]`.
    """
    chord_cfg = cfg.get("chord", {}) or {}
    chord_enabled = bool(chord_cfg.get("enabled", False))
    chord_cache = chord_cfg.get("cache_path")

    chord_vocab: list[str] | None = None
    if chord_enabled:
        if chord_cache is None:
            raise ValueError("chord.enabled=True but chord.cache_path is unset")
        with open(chord_cache, "rb") as f:
            chord_payload = pickle.load(f)
        chord_vocab = list(chord_payload["vocab"])

    tok_cfg = TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
        chord_vocab=chord_vocab,
    )
    tok = ChoraleTokenizer(tok_cfg)

    shifts_train = (
        DEFAULT_SHIFTS if chord_cfg.get("transpose_augment", False) else [0]
    )

    common = dict(
        pickle_path=cfg["data"]["jsb_path"],
        piece_length=cfg["data"]["piece_length"],
        stride=cfg["data"]["stride"],
        tokenizer=tok,
        chord_cache_path=chord_cache if chord_enabled else None,
    )
    train_ds = JSBChorales(
        split="train", transpose_shifts=shifts_train, **common
    )
    val_ds = JSBChorales(split="valid", transpose_shifts=[0], **common)
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw),
    )


# ---------------------------------------------------------------------------

def _download(out_path: str) -> None:
    """TODO: wire this up.

    Easiest source: the czhuang/JSB-Chorales-dataset GitHub repo, which mirrors
    the original B-L pickle as `Jsb16thSeparated.npz` or `jsb-chorales-16th.pkl`.
    For now, drop the pickle at the path in configs/base.yaml manually.
    """
    raise NotImplementedError(
        "Download path not implemented. "
        "Manually place the JSB pickle at the configured `data.jsb_path`."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="data/jsb_chorales.pkl")
    ap.add_argument("--inspect", help="path to a pickle to print stats for")
    args = ap.parse_args()

    if args.download:
        _download(args.out)

    if args.inspect:
        with open(args.inspect, "rb") as f:
            try:
                blob = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                blob = pickle.load(f, encoding="latin1")
        for split, chorales in blob.items():
            lens = [len(c) for c in chorales]
            print(f"{split}: {len(chorales)} chorales, "
                  f"min/median/max length = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
