"""Differentiable music-theory rule loss for training M2.

We implement the five rules from the `rule_checker.py` counter as
*expected-violation* tensors under the model's softmax distribution. This
makes them differentiable, so the rule loss can be added to the usual
next-token cross-entropy term.

Setup / conventions
-------------------
The model emits logits of shape (B, L, V) where L = 4*T and tokens are in
SATB-major flatten order. For teacher forcing we know:
  - the ground-truth previous-timestep pitches for each voice
  - the softmax distribution at each position (which we compute from logits)

The five rules each reduce to an inner product or outer product of softmax
distributions restricted to "pitch" tokens. Special tokens (HOLD, REST, BOS,
BAR, PAD) are excluded from all rule calculations — their probability mass
is simply not contributed to any violation.

Vocabulary layout (from data.tokenizer):
    indices 0..N_PITCHES-1 are pitch tokens corresponding to MIDI pitches
    [pitch_min, pitch_max].

All computations are batched over B and over timesteps t=1..T-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import ChoraleTokenizer


# ---------------------------------------------------------------------------
# Mask caches — precomputed once per tokenizer.
# ---------------------------------------------------------------------------

@dataclass
class RuleMasks:
    """Precomputed (N_pitches, N_pitches) masks over pitch pairs.

    All masks are float tensors with 1.0 at forbidden pairs, 0.0 otherwise.
    Indices are *token* indices (not MIDI). Converting is:
        midi = token + pitch_min
    so pitch-token arithmetic is identical to MIDI arithmetic up to an
    additive constant, which means interval differences are preserved.
    """

    voice_crossing: torch.Tensor      # (P, P): 1 if p_upper < p_lower
    spacing:        torch.Tensor      # (P, P): 1 if |pA - pB| > 12 (one octave)

    n_pitches: int
    device: torch.device


def build_rule_masks(tok: ChoraleTokenizer, device: torch.device) -> RuleMasks:
    P = tok.cfg.n_pitches
    idx = torch.arange(P, device=device)
    a = idx.unsqueeze(1)                          # (P, 1) = "upper voice candidate"
    b = idx.unsqueeze(0)                          # (1, P) = "lower voice candidate"

    # voice crossing: upper < lower (i.e. S < A, A < T, or T < B)
    voice_crossing = (a < b).float()

    # spacing: |a - b| > 12 semitones
    spacing = (torch.abs(a - b) > 12).float()

    return RuleMasks(voice_crossing=voice_crossing, spacing=spacing,
                     n_pitches=P, device=device)


# ---------------------------------------------------------------------------
# Parallel / hidden interval masks depend on the *previous* interval, so we
# build them on the fly (but only for the two relevant interval classes).
# ---------------------------------------------------------------------------

def _parallel_mask(
    n_pitches: int,
    prev_upper: torch.Tensor,          # (B,) pitch-token of upper voice at t-1
    prev_lower: torch.Tensor,          # (B,) pitch-token of lower voice at t-1
    interval_mod12: int,               # 7 for P5, 0 for P8
    device: torch.device,
) -> torch.Tensor:
    """Return (B, P, P) mask where entry (b, i, j) = 1 iff picking upper=i,
    lower=j would form a *parallel* interval of the given class relative to
    (prev_upper[b], prev_lower[b]).

    Conditions:
      1. (i - j) mod 12 == interval_mod12
      2. (prev_upper - prev_lower) mod 12 == interval_mod12
      3. both voices move in the same direction (same sign of pitch change)
    """
    B = prev_upper.shape[0]
    idx = torch.arange(n_pitches, device=device)
    a = idx.view(1, n_pitches, 1).expand(B, n_pitches, n_pitches)   # upper cand
    b = idx.view(1, 1, n_pitches).expand(B, n_pitches, n_pitches)   # lower cand

    pu = prev_upper.view(B, 1, 1)
    pl = prev_lower.view(B, 1, 1)

    same_interval_now  = ((a - b) % 12 == interval_mod12)
    same_interval_prev = ((pu - pl) % 12 == interval_mod12).expand_as(same_interval_now)
    # direction: sign of (current - previous) for each voice
    up_move  = a - pu
    low_move = b - pl
    same_direction = (torch.sign(up_move) == torch.sign(low_move)) & (up_move != 0)

    mask = same_interval_now & same_interval_prev & same_direction
    return mask.float()


def _hidden_mask(
    n_pitches: int,
    prev_upper: torch.Tensor,
    prev_lower: torch.Tensor,
    interval_mod12: int,
    leap_threshold: int,
    device: torch.device,
) -> torch.Tensor:
    """Hidden 5th/8ve between outer voices (soprano & bass):
    both voices move in the same direction AND the upper voice moves by a
    leap (> 2 semitones, per the checker doc — adjust if your definition
    differs) AND the arrival interval is a P5 or P8.

    (interval_mod12 need not match at t-1 — this is the hidden variant.)
    """
    B = prev_upper.shape[0]
    idx = torch.arange(n_pitches, device=device)
    a = idx.view(1, n_pitches, 1).expand(B, n_pitches, n_pitches)
    b = idx.view(1, 1, n_pitches).expand(B, n_pitches, n_pitches)
    pu = prev_upper.view(B, 1, 1)
    pl = prev_lower.view(B, 1, 1)

    arrival = ((a - b) % 12 == interval_mod12)
    up_move = a - pu
    low_move = b - pl
    same_direction = (torch.sign(up_move) == torch.sign(low_move)) & (up_move != 0)
    upper_leaps = torch.abs(up_move) > leap_threshold

    return (arrival & same_direction & upper_leaps).float()


# ---------------------------------------------------------------------------
# Main module.
# ---------------------------------------------------------------------------

class RuleLoss(nn.Module):
    """Compute expected-violation loss from logits + ground-truth prev pitches.

    Usage:
        rule_loss = RuleLoss(tokenizer, weights_dict, lambda_total)
        extra = rule_loss(logits, input_ids)      # scalar, differentiable
        total = ce_loss + extra
    """

    def __init__(
        self,
        tokenizer: ChoraleTokenizer,
        per_rule_weights: dict[str, float],
        lambda_total: float = 1.0,
        large_leap_threshold: int = 9,     # > major 6th
        hidden_leap_threshold: int = 2,    # upper voice leaps more than a step
    ):
        super().__init__()
        self.tok = tokenizer
        self.lambda_total = lambda_total
        self.w = per_rule_weights
        self.large_leap_threshold = large_leap_threshold
        self.hidden_leap_threshold = hidden_leap_threshold
        self._masks: RuleMasks | None = None
        # voice pair indices: (upper, lower)
        self.adjacent_pairs = [(0, 1), (1, 2), (2, 3)]   # S-A, A-T, T-B
        self.outer_pair = (0, 3)                         # S-B
        # all pairs (for parallels — rule applies between any two voices)
        self.all_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        ]

    # ---- lazy mask init so the module is device-agnostic at construction --

    def _ensure_masks(self, device: torch.device) -> RuleMasks:
        if self._masks is None or self._masks.device != device:
            self._masks = build_rule_masks(self.tok, device)
        return self._masks

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,           # (B, L, V) — L = 4T or 4T-1 (shifted)
        input_ids: torch.Tensor,        # (B, L) — the teacher-forced inputs
    ) -> dict[str, torch.Tensor]:
        """Return {'total': scalar, '<rule>': scalar, ...}.

        Caller does `loss = ce + out['total']`.
        """
        device = logits.device
        masks = self._ensure_masks(device)
        tok = self.tok
        P = tok.cfg.n_pitches

        B, L, V = logits.shape
        # Reshape the flat sequence back into (B, T, 4) by slicing voices.
        # The caller's convention is that logits[:, i, :] predicts input_ids[:, i+1].
        # We want the softmax over pitches per (t, voice). Truncate L so it's
        # a multiple of 4.
        L_use = (L // 4) * 4
        logits = logits[:, :L_use, :]
        # softmax over the full vocab, then restrict to pitch slice
        probs = F.softmax(logits, dim=-1)
        pitch_probs = probs[..., :P]                      # (B, L_use, P)
        # shape as (B, T, 4, P)
        T = L_use // 4
        pitch_probs = pitch_probs.view(B, T, 4, P)

        # ground-truth previous-timestep pitches from the *teacher-forced*
        # target. We pull from input_ids: the token at flat position (t*4+v)
        # is the voice v at timestep t (HOLD/REST get ignored by clamping to
        # the pitch range — we produce a sentinel and mask out their
        # contribution to the rule loss).
        ids = input_ids[:, :L_use].view(B, T, 4)          # (B, T, 4) Long
        is_pitch = (ids >= 0) & (ids < P)                 # (B, T, 4)
        prev_pitch_tok = ids.clamp(0, P - 1)              # safe indexing

        out: dict[str, torch.Tensor] = {}

        # ---- 1. voice crossing (adjacent pairs, single timestep) ---------
        # E[crossing between voice i (upper) and voice j (lower) at time t]
        # = sum_{pu, pl} P(v_i = pu) P(v_j = pl) * mask[pu, pl]
        vc_total = pitch_probs.new_zeros(())
        for (u, l) in self.adjacent_pairs:
            pu = pitch_probs[:, :, u, :]                    # (B, T, P)
            pl = pitch_probs[:, :, l, :]                    # (B, T, P)
            # bilinear: sum_{i,j} pu[...,i] * mask[i,j] * pl[...,j]
            # = pu @ mask @ pl.T, batched per (B, T)
            vc = torch.einsum("btp,pq,btq->bt", pu, masks.voice_crossing, pl)
            vc_total = vc_total + vc.mean()                 # average over B, T
        out["voice_crossing"] = vc_total

        # ---- 2. spacing (adjacent pairs, S-A and A-T only) ---------------
        sp_total = pitch_probs.new_zeros(())
        for (u, l) in [(0, 1), (1, 2)]:                     # S-A, A-T; skip T-B
            pu = pitch_probs[:, :, u, :]
            pl = pitch_probs[:, :, l, :]
            sp = torch.einsum("btp,pq,btq->bt", pu, masks.spacing, pl)
            sp_total = sp_total + sp.mean()
        out["spacing"] = sp_total

        # ---- 3. large leap (per voice, requires t-1) --------------------
        # E[|pitch_v(t) - pitch_v(t-1)| > threshold]
        # For t=1..T-1: look up prev pitch in is_pitch; mask out non-pitch.
        ll_total = pitch_probs.new_zeros(())
        for v in range(4):
            # shape (B, T-1, P): prob dist at time t>=1
            p_cur = pitch_probs[:, 1:, v, :]                # (B, T-1, P)
            prev = prev_pitch_tok[:, :-1, v]                # (B, T-1)
            prev_valid = is_pitch[:, :-1, v].float()        # (B, T-1)
            # build a (B, T-1, P) 0/1 mask over candidate pitches
            idx = torch.arange(P, device=device)
            # |cand - prev| > threshold
            diff = idx.view(1, 1, P) - prev.unsqueeze(-1)
            leap_mask = (diff.abs() > self.large_leap_threshold).float()
            leap = (p_cur * leap_mask).sum(dim=-1)          # (B, T-1)
            leap = leap * prev_valid
            ll_total = ll_total + leap.mean()
        out["large_leap"] = ll_total

        # ---- 4. parallel 5ths and 8ves (all pairs, requires t-1) --------
        # E[parallel interval] via the (B, P, P) mask built per (prev_u, prev_l)
        # We batch-build masks across (B, T-1) by flattening.
        p5_total = pitch_probs.new_zeros(())
        p8_total = pitch_probs.new_zeros(())
        for (u, l) in self.all_pairs:
            pu = pitch_probs[:, 1:, u, :].reshape(-1, P)     # (B*(T-1), P)
            pl = pitch_probs[:, 1:, l, :].reshape(-1, P)
            prev_u = prev_pitch_tok[:, :-1, u].reshape(-1)   # (B*(T-1),)
            prev_l = prev_pitch_tok[:, :-1, l].reshape(-1)
            valid = (is_pitch[:, :-1, u] & is_pitch[:, :-1, l]).float().reshape(-1)

            # P5 (interval mod 12 == 7)
            m5 = _parallel_mask(P, prev_u, prev_l, 7, device)     # (N, P, P)
            par5 = torch.einsum("np,npq,nq->n", pu, m5, pl) * valid
            p5_total = p5_total + par5.mean()

            # P8 (interval mod 12 == 0)
            m8 = _parallel_mask(P, prev_u, prev_l, 0, device)
            par8 = torch.einsum("np,npq,nq->n", pu, m8, pl) * valid
            p8_total = p8_total + par8.mean()

        out["parallel_fifth"] = p5_total
        out["parallel_octave"] = p8_total

        # ---- 5. hidden 5ths and 8ves (outer voices only, S vs B) --------
        u, l = self.outer_pair
        pu = pitch_probs[:, 1:, u, :].reshape(-1, P)
        pl = pitch_probs[:, 1:, l, :].reshape(-1, P)
        prev_u = prev_pitch_tok[:, :-1, u].reshape(-1)
        prev_l = prev_pitch_tok[:, :-1, l].reshape(-1)
        valid = (is_pitch[:, :-1, u] & is_pitch[:, :-1, l]).float().reshape(-1)

        h5 = _hidden_mask(P, prev_u, prev_l, 7, self.hidden_leap_threshold, device)
        hid5 = torch.einsum("np,npq,nq->n", pu, h5, pl) * valid
        out["hidden_fifth"] = hid5.mean()

        h8 = _hidden_mask(P, prev_u, prev_l, 0, self.hidden_leap_threshold, device)
        hid8 = torch.einsum("np,npq,nq->n", pu, h8, pl) * valid
        out["hidden_octave"] = hid8.mean()

        # ---- aggregate ---------------------------------------------------
        total = pitch_probs.new_zeros(())
        for name, val in out.items():
            total = total + self.w.get(name, 0.0) * val
        out["total"] = self.lambda_total * total
        return out


def build_rule_loss_from_config(cfg: dict, tokenizer: ChoraleTokenizer) -> RuleLoss:
    rc = cfg["rule_loss"]
    return RuleLoss(
        tokenizer=tokenizer,
        per_rule_weights=rc["per_rule_weights"],
        lambda_total=rc["lambda_total"],
    )
