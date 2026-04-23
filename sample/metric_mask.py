"""M3-metric: metric-weighted constrained decoding.

Extends sample.decode_m3.ConstrainedProcessor. The existing M3 applies a
uniform alpha to every parallel-5th/8ve violation regardless of where in
the bar the landing note falls. Species counterpoint teaches that
parallel motion across a strong beat is much more offensive than a
parallel across an off-beat passing tone. This processor scales the
*parallel-motion* penalty by the metric weight of the landing timestep
and leaves voice-crossing / large-leap penalties unchanged (those are
equally bad on any beat).

Works in two modes:
  - Voice-only layout (M1/M3 4-wide): the 4-wide position maps to
    timestep = (position - voice) // 4.
  - Chord-interleaved layout (M4+M3, 5-wide + 2-token prefix): the
    caller strips RN slots before passing to us, so we see a voice-only
    sequence exactly like M1/M3 — no layout special-casing needed here.

Assumed meter: 4/4 at 16th-note resolution (16 ticks per bar). For
3/4 chorales this over-weights the "beat 3" slot, but the two metric
categories that actually matter — downbeat vs. off-beat — stay correct
as long as the piece starts on a downbeat (it does, by tokenizer
convention).
"""

from __future__ import annotations

import torch

from data.tokenizer import ChoraleTokenizer
from sample.decode_m3 import ConstrainedProcessor


# Metric weights by 16th-note position within a 4/4 bar.
# Downbeat > beat 3 > beats 2 & 4 > eighth off-beats > sixteenth off-beats.
# These scale the parallel-motion penalty multiplicatively; 1.0 reproduces
# the uniform M3 behavior on that position.
DEFAULT_METRIC_WEIGHTS = {
    0:  1.50,   # bar downbeat
    8:  1.25,   # beat 3 of 4/4 (secondary strong)
    4:  1.00,   # beat 2
    12: 1.00,   # beat 4
    2:  0.50,   # "and" of 1
    6:  0.50,   # "and" of 2
    10: 0.50,   # "and" of 3
    14: 0.50,   # "and" of 4
    1:  0.25,   # 16th off-beats
    3:  0.25,
    5:  0.25,
    7:  0.25,
    9:  0.25,
    11: 0.25,
    13: 0.25,
    15: 0.25,
}


# Beat-aware HOLD bias. Added to the HOLD logit at every voice slot,
# modulated by the landing timestep's metric position.
#   negative  -> penalize HOLD (force a new attack / articulation)
#   zero      -> neutral (model's own preference)
#   positive  -> encourage HOLD (sustain)
#
# At HOLD emission rate 14% the model is ~6x too busy vs Bach's ~85%.
# We need HOLD to be ~10x more likely on off-beats, which in logit space is
# roughly +log(10) ≈ +2.3. On strong beats we want the opposite — force an
# attack — so we subtract a little.
#
# Tune via the `hold_bias` kwarg; DEFAULT_HOLD_BIAS is a decent starting
# point for 4/4 chorales.
DEFAULT_HOLD_BIAS = {
    0:  -2.0,   # downbeat: strongly penalize HOLD → force attack on beat 1
    8:  -1.0,   # beat 3: penalize
    4:   1.5,   # beat 2: mild HOLD preference
    12:  1.5,   # beat 4: mild HOLD preference
    2:   5.0,   # "and" of 1: strongly prefer HOLD
    6:   5.0,
    10:  5.0,
    14:  5.0,
    1:   7.0,   # 16th off-beats: overwhelmingly prefer HOLD
    3:   7.0,
    5:   7.0,
    7:   7.0,
    9:   7.0,
    11:  7.0,
    13:  7.0,
    15:  7.0,
}


# Maximum voices allowed to move on each beat position. "Move" = emit a
# pitch token rather than HOLD. Encodes the species-counterpoint rule:
# on off-beats at most ONE voice should be moving at a time. On strong
# beats all four voices may move (chord change). See the research notes
# in METRIC_ABLATION.md for the Bach-chorale texture motivation.
DEFAULT_MAX_MOVES = {
    0:  4, 8:  4,                   # beats 1 & 3: chord change, all voices
    4:  4, 12: 4,                   # beats 2 & 4: usually chord change too
    2:  1, 6:  1, 10: 1, 14: 1,     # 8th off-beats: at most 1 voice (passing)
    1:  1, 3:  1, 5:  1, 7:  1,     # 16th off-beats: at most 1 voice
    9:  1, 11: 1, 13: 1, 15: 1,
}


# Extra HOLD bias applied when the per-beat move budget is already spent.
# This is a HARD-ish force: on top of the soft DEFAULT_HOLD_BIAS, we add
# an additional +10 to HOLD's logit whenever the current timestep has
# already used up its DEFAULT_MAX_MOVES quota. Net effect: after one
# voice moves on an off-beat, the remaining voices are almost certain
# to hold.
HARD_HOLD_BONUS = 10.0


# REST penalty: Bach chorales don't have mid-phrase rests. This pushes
# REST's logit down by a huge amount on every voice slot so the model
# effectively can't emit REST. (The slot mask in decode_m4.py allows
# REST for generality, but we don't want it.)
REST_PENALTY = 20.0


class MetricConstrainedProcessor(ConstrainedProcessor):
    """Parallel-motion penalty scaled by the landing timestep's metric weight."""

    def __init__(
        self,
        tok: ChoraleTokenizer,
        alpha: float = 5.0,
        max_leap: int = 12,
        metric_weights: dict[int, float] | None = None,
        hold_bias: dict[int, float] | None = None,
        hold_bias_scale: float = 1.0,
        max_moves: dict[int, int] | None = None,
        bar_len: int = 16,
    ):
        super().__init__(tok, alpha=alpha, max_leap=max_leap)
        self.bar_len = bar_len
        self.metric_weights = metric_weights or DEFAULT_METRIC_WEIGHTS
        # hold_bias = None → no HOLD biasing (default off)
        # hold_bias = {}   → explicit zeros, also no-op
        # hold_bias = DEFAULT_HOLD_BIAS → beat-aware HOLD prior active
        # hold_bias_scale multiplies every entry (for CLI tuning)
        self.hold_bias = hold_bias
        self.hold_bias_scale = hold_bias_scale
        # max_moves = None → no cross-voice coupling (each voice independent)
        # max_moves = DEFAULT_MAX_MOVES → off-beats forced monophonic texture
        self.max_moves = max_moves

    # Override __call__ to apply a per-position scaling to the parallel rule
    # only. We replicate the parent's logic inline because the parent bakes
    # the three rules into one penalty tensor with a single alpha — easiest
    # to fork the method than to try to hook into it.
    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape
        position = generated.size(1)
        voice = position % 4
        base = position - voice
        prev_timestep_end = base
        if prev_timestep_end < 4:
            return logits

        timestep = prev_timestep_end // 4       # landing timestep (the one we're filling)
        beat_pos = timestep % self.bar_len
        m_weight = float(self.metric_weights.get(beat_pos, 1.0))

        device = logits.device
        # Separate penalty tensors so we can scale them independently.
        leap_pen = torch.zeros(B, V, device=device)
        cross_pen = torch.zeros(B, V, device=device)
        parallel_pen = torch.zeros(B, V, device=device)
        cand = torch.arange(self.P, device=device)

        for b in range(B):
            prev = self._effective_pitches(generated[b, :prev_timestep_end])
            cur  = self._effective_pitches(generated[b, :position])
            prev_v = prev[voice]

            # 1. large-leap (uniform — bad on any beat)
            if prev_v is not None:
                leap_mask = (cand - prev_v).abs() > self.max_leap
                leap_pen[b, : self.P] = leap_pen[b, : self.P] + leap_mask.float()

            # 2. voice-crossing (uniform — always an error)
            if voice >= 1 and cur[voice - 1] is not None:
                upper = cur[voice - 1]
                cross_pen[b, : self.P] = cross_pen[b, : self.P] + (cand > upper).float()

            # 3. parallel 5ths / 8ves — metric-weighted
            for u in range(voice):
                if prev[u] is None or prev_v is None or cur[u] is None:
                    continue
                prev_interval = (prev[u] - prev_v) % 12
                if prev_interval not in (0, 7):
                    continue
                cur_u = cur[u]
                for p_idx in range(self.P):
                    if ((cur_u - p_idx) % 12) != prev_interval:
                        continue
                    up_move = cur_u - prev[u]
                    low_move = p_idx - prev_v
                    if (up_move > 0 and low_move > 0) or (up_move < 0 and low_move < 0):
                        parallel_pen[b, p_idx] += 1.0

        total = leap_pen + cross_pen + m_weight * parallel_pen
        shaped = logits - self.alpha * total

        # Beat-aware HOLD prior (additive on the HOLD logit only).
        if self.hold_bias is not None:
            hb = float(self.hold_bias.get(beat_pos, 0.0)) * self.hold_bias_scale
            if hb != 0.0:
                shaped[:, self.tok.HOLD] = shaped[:, self.tok.HOLD] + hb

        # Cross-voice articulation coupling: if the per-beat move budget
        # has already been spent by earlier voices in this timestep, force
        # the remaining voices to hold. Encodes "at most N voices move
        # per off-beat" (species-counterpoint rule).
        #
        # We ALSO apply the budget preemptively at voice 0 by scaling the
        # HOLD bias by (n_voices_remaining / n_voices_total). Without this,
        # Soprano grabs the single off-beat move slot every time, leaving
        # A/T/B idle. The preemptive gate adds enough HOLD pressure on
        # early voices so the move slot is fairly distributed.
        if self.max_moves is not None:
            budget = int(self.max_moves.get(beat_pos, 4))
            voices_remaining = 4 - voice            # incl. current
            # Fair-share HOLD bias: if budget=1 and voices_remaining=4, we
            # want P(this voice moves) = 1/4, not whatever the model prefers.
            # HOLD bias = log((voices_remaining - budget) / budget + 1e-6)
            # but we cap it to avoid infinite values.
            if budget < voices_remaining and budget >= 1:
                fair_bias = 2.0 * (voices_remaining - budget)     # +2 per "extra" voice
                shaped[:, self.tok.HOLD] = shaped[:, self.tok.HOLD] + fair_bias
            for b in range(B):
                # count voices in this timestep (voices 0..voice-1) that
                # emitted a PITCH (moved). HOLD / REST don't count.
                n_moved = 0
                for v_prev in range(voice):
                    pos_prev = prev_timestep_end + v_prev
                    tid = int(generated[b, pos_prev].item())
                    if 0 <= tid < self.P:
                        n_moved += 1
                if n_moved >= budget:
                    # budget used up — force HOLD
                    shaped[b, self.tok.HOLD] = shaped[b, self.tok.HOLD] + HARD_HOLD_BONUS

        # Penalize REST everywhere — mid-phrase silence is not Bach.
        shaped[:, self.tok.REST] = shaped[:, self.tok.REST] - REST_PENALTY

        return shaped
