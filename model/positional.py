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

Chord attention bias (M4 improvement)
--------------------------------------
When `chord_attn_bias=True`, a learned per-head scalar bias is added to
attention logits wherever a pitch token (roles 1-4) attends to the RN
token (role 0) at the *same* timestep. This gives the model an explicit
structural hint that the chord label is the harmonic context for its
surrounding pitch tokens, rather than requiring the model to discover
this relationship purely from data.
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
        chord_attn_bias: bool = False,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps
        self.chord_layout = chord_layout
        self.chord_attn_bias = chord_attn_bias and chord_layout
        # In chord_layout mode we hard-code 5 roles so the checkpoint shape
        # is determined by the flag, not the caller — makes eval code
        # simpler (`M4Model.load_state_dict` doesn't need extra config).
        self.n_voices = 5 if chord_layout else n_voices
        self.timestep_emb = nn.Embedding(max_timesteps, d_model)
        self.voice_emb = nn.Embedding(self.n_voices, d_model)
        # small init so positional doesn't dominate the token embedding
        nn.init.normal_(self.timestep_emb.weight, std=0.02)
        nn.init.normal_(self.voice_emb.weight, std=0.02)

        # Per-head learned bias: when a pitch token attends to its own
        # timestep's RN token, add this bias to the pre-softmax score.
        # Initialized to a small positive value so the model starts with
        # a slight preference for attending to the chord context.
        if self.chord_attn_bias:
            self.n_heads = n_heads
            self.rn_bias = nn.Parameter(torch.full((n_heads,), 0.5))

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

    def get_chord_attn_bias_mask(self, L: int, device) -> torch.Tensor | None:
        """Build (1, n_heads, L, L) additive attention bias for chord layout.

        For each query position q that is a pitch token (role 1-4),
        and each key position k that is an RN token (role 0) at the
        same timestep as q, we add a learned per-head bias to the
        pre-softmax attention score. This is causality-safe because
        in the 5-wide layout [RN, S, A, T, B] the RN token always
        precedes its timestep's pitch tokens.

        Returns None when chord_attn_bias is disabled (M1 path).
        """
        if not self.chord_attn_bias:
            return None

        t_idx, v_idx = self._indices_chord(L, device)

        # is_pitch[q] = True if position q is a pitch token (role 1-4)
        is_pitch = v_idx > 0                                      # (L,)
        # is_rn[k] = True if position k is an RN/chord token (role 0)
        is_rn = v_idx == 0                                         # (L,)
        # same_timestep[q, k] = True if positions q and k share a timestep
        same_t = t_idx.unsqueeze(0) == t_idx.unsqueeze(1)          # (L, L)

        # mask[q, k] = True where pitch query q should get a bonus for
        # attending to same-timestep RN key k.
        # Causality is implicit: RN is at offset 0 in each 5-wide group,
        # so its position is always <= the pitch positions at offsets 1-4.
        mask = is_pitch.unsqueeze(1) & is_rn.unsqueeze(0) & same_t  # (L, L)

        # Build (1, H, L, L) bias tensor. Only the masked positions are
        # nonzero; everything else stays 0 (no effect on attention).
        # rn_bias is (H,) -> reshape to (1, H, 1, 1) for broadcasting.
        bias = torch.where(
            mask.unsqueeze(0).unsqueeze(0),                         # (1, 1, L, L)
            self.rn_bias.view(1, -1, 1, 1),                        # (1, H, 1, 1)
            torch.zeros(1, device=device),
        )  # (1, H, L, L)

        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model). Returns x with positional embeddings added."""
        _, L, _ = x.shape
        if self.chord_layout:
            t_idx, v_idx = self._indices_chord(L, x.device)
        else:
            t_idx, v_idx = self._indices_flat(L, x.device)
        return x + self.timestep_emb(t_idx) + self.voice_emb(v_idx)