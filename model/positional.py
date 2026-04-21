"""Voice-aware positional embeddings.

M1 (default, 4-voice SATB-packed layout)
----------------------------------------
Given a flat token position p in [0, 4*T), the position decomposes into:
    timestep t = p // 4
    voice    v = p %  4    (0=S, 1=A, 2=T, 3=B)

We add two learned embeddings — one per timestep, one per voice — to the
token embedding before the first transformer block. The voice embedding is
the part the rule loss relies on: it tells the model "this slot is the
alto," which is what makes the per-voice constraints learnable.

M4 (chord-interleaved layout)
-----------------------------
When `chord_layout=True`, the module expects a 5-wide layout with a
2-token key prefix:
    [MODE, TONIC_PC, RN_0, S_0, A_0, T_0, B_0, RN_1, ...]
Voice roles are now 5 (0=CHORD, 1=S, 2=A, 3=T, 4=B). The MODE / TONIC_PC
prefix maps to ROLE_CHORD at timestep 0, so they don't claim extra
timestep-embedding capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VoiceAwarePositional(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_timesteps: int,
        n_voices: int = 4,
        chord_layout: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps
        self.chord_layout = chord_layout
        # In chord_layout mode we hard-code 5 roles so the checkpoint shape
        # is determined by the flag, not the caller — makes eval code
        # simpler (`M4Model.load_state_dict` doesn't need extra config).
        self.n_voices = 5 if chord_layout else n_voices
        self.timestep_emb = nn.Embedding(max_timesteps, d_model)
        self.voice_emb = nn.Embedding(self.n_voices, d_model)
        # small init so positional doesn't dominate the token embedding
        nn.init.normal_(self.timestep_emb.weight, std=0.02)
        nn.init.normal_(self.voice_emb.weight, std=0.02)

    # ----- index computation --------------------------------------------

    def _indices_flat(self, L: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-chord layout: positions 0..L-1 packed `n_voices` per timestep."""
        positions = torch.arange(L, device=device)
        t_idx = (positions // self.n_voices).clamp_max(self.max_timesteps - 1)
        v_idx = positions % self.n_voices
        return t_idx, v_idx

    def _indices_chord(self, L: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Chord-interleaved layout: 2-token prefix + 5-wide timesteps."""
        positions = torch.arange(L, device=device)
        # timestep: prefix (pos 0,1) both -> 0; after that (pos-2) // 5
        t_raw = torch.where(
            positions < 2,
            torch.zeros_like(positions),
            (positions - 2) // 5,
        ).clamp_max(self.max_timesteps - 1)
        # voice: prefix -> 0 (CHORD); after that (pos-2) % 5
        v_raw = torch.where(
            positions < 2,
            torch.zeros_like(positions),
            (positions - 2) % 5,
        )
        return t_raw, v_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model). Returns x with positional embeddings added."""
        _, L, _ = x.shape
        if self.chord_layout:
            t_idx, v_idx = self._indices_chord(L, x.device)
        else:
            t_idx, v_idx = self._indices_flat(L, x.device)
        return x + self.timestep_emb(t_idx) + self.voice_emb(v_idx)
