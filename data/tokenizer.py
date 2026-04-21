"""SATB-packed tokenizer for four-part chorales.

A training example is a (4, T) array of MIDI pitches (or sentinel values for
rest / hold) at 16th-note quantization. We pack it into a 1D token sequence
of length 4*T in SATB order:

    [S_0, A_0, T_0, B_0, S_1, A_1, T_1, B_1, ...]

This layout is what makes the differentiable rule loss tractable: any pair
of consecutive timesteps' SATB pitches are adjacent groups of 4 in the
sequence, and the model emits one categorical distribution per token.

Vocabulary (default config: pitch_min=36, pitch_max=84):
    indices  0..48   -> MIDI pitches 36..84   (49 pitches)
    index    49      -> HOLD   (continue previous pitch in this voice)
    index    50      -> REST   (silence)
    index    51      -> BOS
    index    52      -> BAR    (downbeat marker, optional)
    index    53      -> PAD
    vocab_size = 54

Chord-conditioned (M4) extension
--------------------------------
When constructed with a `chord_vocab` list, the tokenizer grows additional
token ranges:
    KEY_MAJOR, KEY_MINOR              (2 key-mode tokens; tonic pc in a
                                       separate 12-entry range)
    PC_0 .. PC_11                     (12 tonic pitch-class tokens)
    RN_0 .. RN_{K-1}                  (K Roman-numeral tokens, incl.
                                       RN_REST / RN_OTHER sentinels)

The interleaved sequence layout becomes (per timestep, 5 tokens):
    [KEY_MODE, PC_tonic,
     RN_0, S_0, A_0, T_0, B_0,
     RN_1, S_1, A_1, T_1, B_1,
     ...]
The prefix is 2 tokens (mode + tonic pc); the body is 5 tokens per step.
The voice role for positional embedding is one of {CHORD, S, A, T, B}
(see `voice_index_chord`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F",
                     "F#", "G", "G#", "A", "A#", "B"]


def _tonic_to_pc(tonic_name: str) -> int:
    """Map music21 tonic name (e.g. 'C', 'F#', 'Bb') to pitch class 0..11."""
    name = tonic_name.strip()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0].upper()]
    for acc in name[1:]:
        if acc == "#":
            base += 1
        elif acc == "-" or acc.lower() == "b":
            base -= 1
    return base % 12


@dataclass
class TokenizerConfig:
    pitch_min: int = 36
    pitch_max: int = 84  # inclusive
    # Optional: sorted list of unique RN strings (including RN_REST / RN_OTHER).
    # When set, the tokenizer is in "chord mode" and exposes chord-aware
    # encode/decode methods.
    chord_vocab: list[str] | None = None

    @property
    def n_pitches(self) -> int:
        return self.pitch_max - self.pitch_min + 1

    @property
    def n_chord(self) -> int:
        return 0 if self.chord_vocab is None else len(self.chord_vocab)


class ChoraleTokenizer:
    """Encodes/decodes (4, T) chorale arrays to/from 1D token sequences.

    The "raw" array uses MIDI pitch ints directly, with two sentinel ints:
        -1  = HOLD
        -2  = REST
    All other entries must be in [pitch_min, pitch_max].
    """

    HOLD_RAW = -1
    REST_RAW = -2

    def __init__(self, cfg: TokenizerConfig | None = None):
        self.cfg = cfg or TokenizerConfig()
        n = self.cfg.n_pitches
        self.HOLD = n
        self.REST = n + 1
        self.BOS = n + 2
        self.BAR = n + 3
        self.PAD = n + 4
        base_vocab = n + 5

        # ----- chord-mode extensions ------------------------------------
        # Token id ranges (all contiguous after the base vocab):
        #   KEY_MAJOR, KEY_MINOR                          (2 ids)
        #   PC_0 .. PC_11                                 (12 ids, tonic pc)
        #   RN_0 .. RN_{K-1}                              (K ids)
        self.KEY_MAJOR = base_vocab
        self.KEY_MINOR = base_vocab + 1
        self.PC_BASE = base_vocab + 2          # PC_i = PC_BASE + i
        self.RN_BASE = base_vocab + 14         # RN_i = RN_BASE + i

        if self.cfg.chord_vocab is not None:
            self._rn_to_id = {rn: i for i, rn in enumerate(self.cfg.chord_vocab)}
            self._id_to_rn = list(self.cfg.chord_vocab)
            self.vocab_size = self.RN_BASE + self.cfg.n_chord
        else:
            self._rn_to_id = {}
            self._id_to_rn = []
            self.vocab_size = base_vocab        # no chord extensions

    # ----- pitch <-> token id ---------------------------------------------

    def pitch_to_token(self, pitch: int) -> int:
        if pitch == self.HOLD_RAW:
            return self.HOLD
        if pitch == self.REST_RAW:
            return self.REST
        if not (self.cfg.pitch_min <= pitch <= self.cfg.pitch_max):
            raise ValueError(
                f"pitch {pitch} out of range "
                f"[{self.cfg.pitch_min}, {self.cfg.pitch_max}]"
            )
        return pitch - self.cfg.pitch_min

    def token_to_pitch(self, token: int) -> int:
        if token == self.HOLD:
            return self.HOLD_RAW
        if token == self.REST:
            return self.REST_RAW
        if 0 <= token < self.cfg.n_pitches:
            return token + self.cfg.pitch_min
        # specials decode to REST
        return self.REST_RAW

    # ----- batch encode / decode ------------------------------------------

    def encode(self, chorale: np.ndarray) -> torch.Tensor:
        """(4, T) int array -> (4*T,) LongTensor of token ids."""
        if chorale.ndim != 2 or chorale.shape[0] != 4:
            raise ValueError(f"expected (4, T), got {chorale.shape}")
        T = chorale.shape[1]
        out = np.empty((T, 4), dtype=np.int64)
        for t in range(T):
            for v in range(4):
                out[t, v] = self.pitch_to_token(int(chorale[v, t]))
        return torch.from_numpy(out.reshape(-1))  # SATB-major flatten

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        """(4*T,) tokens -> (4, T) MIDI pitch array (with -1 / -2 sentinels)."""
        toks = tokens.detach().cpu().numpy().astype(np.int64)
        if toks.size % 4 != 0:
            toks = toks[: (toks.size // 4) * 4]
        T = toks.size // 4
        grid = toks.reshape(T, 4)
        out = np.empty((4, T), dtype=np.int64)
        for t in range(T):
            for v in range(4):
                out[v, t] = self.token_to_pitch(int(grid[t, v]))
        return out

    # ----- helpers used by positional embedding & rule loss ---------------

    def voice_index(self, position: int) -> int:
        """Which voice (0=S, 1=A, 2=T, 3=B) does flat-index `position` belong to?"""
        return position % 4

    def timestep_index(self, position: int) -> int:
        return position // 4

    def is_pitch_token(self, token: int) -> bool:
        return 0 <= token < self.cfg.n_pitches

    def resolve_holds(self, chorale: np.ndarray) -> np.ndarray:
        """Replace HOLD_RAW (-1) with the most recent pitch in that voice.

        Useful before feeding to rule_checker.py / pretty_midi, which expect
        actual MIDI pitches.
        """
        out = chorale.copy()
        for v in range(4):
            last = -2  # rest if voice starts on hold (shouldn't happen)
            for t in range(out.shape[1]):
                if out[v, t] == self.HOLD_RAW:
                    out[v, t] = last
                else:
                    last = out[v, t]
        return out

    # ================== chord-conditioned (M4) extensions =================

    def _require_chord_mode(self):
        if self.cfg.chord_vocab is None:
            raise RuntimeError(
                "tokenizer is not in chord mode; construct with "
                "TokenizerConfig(chord_vocab=[...]) to enable."
            )

    # ----- RN / key token conversions -------------------------------------

    def rn_to_token(self, rn: str) -> int:
        self._require_chord_mode()
        try:
            return self.RN_BASE + self._rn_to_id[rn]
        except KeyError:
            # graceful fallback: any unseen RN collapses to RN_OTHER, if
            # that sentinel is in the vocab (it should be).
            if "RN_OTHER" in self._rn_to_id:
                return self.RN_BASE + self._rn_to_id["RN_OTHER"]
            raise

    def token_to_rn(self, token: int) -> str:
        self._require_chord_mode()
        idx = token - self.RN_BASE
        if 0 <= idx < len(self._id_to_rn):
            return self._id_to_rn[idx]
        return "RN_OTHER"

    def key_tokens(self, tonic_name: str, mode: str) -> tuple[int, int]:
        """Return (mode_token, pc_token) for a (tonic, mode) pair."""
        self._require_chord_mode()
        mode_tok = self.KEY_MAJOR if mode.lower().startswith("maj") else self.KEY_MINOR
        pc_tok = self.PC_BASE + _tonic_to_pc(tonic_name)
        return mode_tok, pc_tok

    def is_chord_token(self, token: int) -> bool:
        """True for KEY_MAJOR/MINOR, PC_*, and RN_* tokens."""
        self._require_chord_mode()
        return self.KEY_MAJOR <= token < self.RN_BASE + self.cfg.n_chord

    def is_rn_token(self, token: int) -> bool:
        if self.cfg.chord_vocab is None:
            return False
        return self.RN_BASE <= token < self.RN_BASE + self.cfg.n_chord

    # ----- interleaved encode / decode ------------------------------------

    def encode_with_chords(
        self,
        chorale: np.ndarray,
        rn_list: list[str],
        tonic_name: str,
        mode: str,
    ) -> torch.Tensor:
        """Produce the chord-interleaved token sequence.

        Layout:
            [MODE, TONIC_PC,                       <- 2-token key prefix
             RN_0, S_0, A_0, T_0, B_0,              <- 5 tokens / timestep
             RN_1, S_1, A_1, T_1, B_1,
             ...]

        Length: 2 + 5 * T.
        """
        self._require_chord_mode()
        if chorale.ndim != 2 or chorale.shape[0] != 4:
            raise ValueError(f"expected (4, T), got {chorale.shape}")
        T = chorale.shape[1]
        if len(rn_list) != T:
            raise ValueError(f"rn_list length {len(rn_list)} != T={T}")

        mode_tok, pc_tok = self.key_tokens(tonic_name, mode)
        out = np.empty(2 + 5 * T, dtype=np.int64)
        out[0] = mode_tok
        out[1] = pc_tok
        for t in range(T):
            base = 2 + 5 * t
            out[base + 0] = self.rn_to_token(rn_list[t])
            for v in range(4):
                out[base + 1 + v] = self.pitch_to_token(int(chorale[v, t]))
        return torch.from_numpy(out)

    def decode_with_chords(
        self, tokens: torch.Tensor
    ) -> tuple[np.ndarray, list[str], str, str]:
        """Inverse of `encode_with_chords`.

        Returns `(chorale, rn_list, tonic_name, mode)`. Unknown / corrupted
        entries fall back to sentinels:
            - bad RN token -> "RN_OTHER"
            - bad key tokens -> ("C", "major")
            - bad pitch tokens -> REST (via token_to_pitch)
        """
        self._require_chord_mode()
        toks = tokens.detach().cpu().numpy().astype(np.int64).tolist()
        if len(toks) < 2:
            return np.zeros((4, 0), dtype=np.int64), [], "C", "major"

        mode_tok, pc_tok = toks[0], toks[1]
        mode = "major" if mode_tok == self.KEY_MAJOR else "minor"
        pc = pc_tok - self.PC_BASE
        tonic_name = PITCH_CLASS_NAMES[pc] if 0 <= pc < 12 else "C"

        body = toks[2:]
        # round DOWN to a multiple of 5 — strip partial trailing timestep
        T = len(body) // 5
        chorale = np.empty((4, T), dtype=np.int64)
        rns: list[str] = []
        for t in range(T):
            base = 5 * t
            rns.append(self.token_to_rn(body[base]))
            for v in range(4):
                chorale[v, t] = self.token_to_pitch(int(body[base + 1 + v]))
        return chorale, rns, tonic_name, mode

    # ----- positional helpers for the 5-wide layout -----------------------

    # Voice role ids for the chord-interleaved sequence:
    # 0 = CHORD (RN token AND the key prefix — both are chord-role), then
    # 1..4 = S, A, T, B.
    ROLE_CHORD = 0
    ROLE_S = 1
    ROLE_A = 2
    ROLE_T = 3
    ROLE_B = 4
    N_ROLES_CHORD = 5

    def voice_index_chord(self, position: int) -> int:
        """Role id (0..4) for flat-index `position` in the chord layout.

        position 0 (MODE) and 1 (TONIC_PC) -> ROLE_CHORD (0).
        After the 2-token prefix, timesteps are 5-wide; offset-0 is the
        RN slot (ROLE_CHORD), offsets 1..4 are S/A/T/B.
        """
        if position < 2:
            return self.ROLE_CHORD
        offset = (position - 2) % 5
        return offset  # 0=CHORD, 1=S, 2=A, 3=T, 4=B

    def timestep_index_chord(self, position: int) -> int:
        """Which timestep does flat-index `position` belong to?

        The 2-token key prefix is mapped to timestep 0 (shared with the
        first musical timestep) so positional embeddings don't waste
        capacity on two otherwise-dead positions.
        """
        if position < 2:
            return 0
        return (position - 2) // 5
