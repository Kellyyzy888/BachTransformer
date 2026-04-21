"""PPO helpers for M2 (RL fine-tuning of M1 with the rule checker as reward).

Pieces:
  - `score_tokens_np`  : in-memory rule checker that mirrors eval.rule_checker
                         but skips the MIDI-file round-trip (too slow to do
                         inside a rollout loop).
  - `compute_gae`      : generalized advantage estimation. Terminal reward is
                         sparse (applied at the last generated token), KL
                         penalty is per-token, GAE mixes them.
  - `ppo_losses`       : clipped surrogate + value loss + bookkeeping stats.

Design notes
------------
We compute *sample-based* KL: `log π_rl(a|s) − log π_ref(a|s)`, accumulated
per token. It's cheaper than full KL over the 54-vocab distribution and it's
the standard estimator used in RLHF (Stiennon et al., Ouyang et al.). If you
see the adaptive β go unstable, switch to full-distribution KL here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.tokenizer import ChoraleTokenizer
from eval.rule_checker import (
    RuleReport,
    check_parallel,
    check_voice_crossing,
    check_hidden_outer,
    check_spacing,
    check_leaps,
    harmonic_score,
)


# ---------------------------------------------------------------------------
# Reward: in-memory rule scoring
# ---------------------------------------------------------------------------

def score_tokens_np(tokens_np: np.ndarray, tok: ChoraleTokenizer) -> dict:
    """Pure-numpy equivalent of `rule_checker.score_midi` that operates
    directly on a token sequence, bypassing the MIDI write/read round-trip.

    tokens_np: 1D int array, length 4*T. Any tokens that aren't pitches or
    HOLDs become RESTs (per the tokenizer convention — this matches what
    tokens_to_midi → load_satb → diagnose does).

    Returns a dict with the per-rule counts and aggregate HarmonicScore.
    """
    # Decode: token ids -> (4, T) MIDI ints with HOLD_RAW (-1) / REST_RAW (-2) sentinels.
    grid = tok.decode(torch.from_numpy(tokens_np.astype(np.int64)))
    grid = tok.resolve_holds(grid)                # HOLDs -> carried-over pitch
    T = grid.shape[1]

    # Convert to per-voice pitch list with None for REST, matching rule_checker.load_satb.
    voices = []
    for v in range(4):
        row = [None if int(grid[v, t]) < 0 else int(grid[v, t]) for t in range(T)]
        voices.append(row)

    r = RuleReport(sample="rollout")
    r.n_steps = T
    r.n_transitions = max(0, T - 1)
    if T > 0:
        check_parallel(voices, r)
        check_voice_crossing(voices, r)
        check_hidden_outer(voices, r)
        check_spacing(voices, r)
        check_leaps(voices, r)

    return {
        "HarmonicScore":       harmonic_score(r),
        "parallel_5ths":       r.parallel_5ths,
        "parallel_8ves":       r.parallel_8ves,
        "voice_crossings":     r.voice_crossings,
        "hidden_5ths_outer":   r.hidden_5ths_outer,
        "hidden_8ves_outer":   r.hidden_8ves_outer,
        "spacing_violations":  r.spacing_violations,
        "large_leaps":         r.large_leaps,
        "augmented_leaps":     r.augmented_leaps,
    }


# ---------------------------------------------------------------------------
# GAE: Generalized Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,        # (B, L) per-token reward
    values:  torch.Tensor,        # (B, L+1) V(s_0)..V(s_L); V(s_L) = 0 at terminal
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE: A_t = Σ_k (γλ)^k δ_{t+k}, δ_t = r_t + γV_{t+1} − V_t.

    We run it backward over the sequence. Mask-out happens in the loss;
    it's safe to compute advantages at every token here because prompt-token
    rewards are zero and prompt-token advantages get filtered out by the
    mask in `ppo_losses`.

    Returns:
        advantages: (B, L)
        returns:    (B, L) = advantages + values[:, :L]
    """
    B, L = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(L)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    returns = advantages + values[:, :L]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO losses
# ---------------------------------------------------------------------------

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def ppo_losses(
    new_log_probs: torch.Tensor,   # (B, L)
    old_log_probs: torch.Tensor,   # (B, L)
    advantages:    torch.Tensor,   # (B, L)  (un-normalized; we normalize inside)
    new_values:    torch.Tensor,   # (B, L)
    returns:       torch.Tensor,   # (B, L)
    mask:          torch.Tensor,   # (B, L) 1 = generated token, 0 = prompt
    clip_eps:      float = 0.2,
    value_coef:    float = 0.5,
    entropy:       torch.Tensor | None = None,   # (B, L) or None
    entropy_coef:  float = 0.0,
    value_clip_eps: float | None = None,         # if set, clip value updates
    old_values:    torch.Tensor | None = None,   # required if value_clip_eps is set
) -> Tuple[torch.Tensor, dict]:
    """Clipped surrogate objective + value loss (+ optional entropy bonus).

    Returns (total_loss, stats_dict). The stats dict is CPU-scalar friendly for
    logging.
    """
    # Normalize advantages across valid tokens — standard PPO trick to
    # reduce variance when rewards have very different scales per rollout.
    valid = mask.bool()
    if valid.any():
        a = advantages[valid].detach()
        mean = a.mean()
        std = a.std().clamp_min(1e-8)
        adv_norm = (advantages - mean) / std
    else:
        adv_norm = advantages

    # Policy loss — clipped surrogate
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    policy_loss = _masked_mean(-torch.min(surr1, surr2), mask)

    # Value loss (optionally clipped, as in PPO2).
    if value_clip_eps is not None and old_values is not None:
        v_clipped = old_values + torch.clamp(
            new_values - old_values, -value_clip_eps, value_clip_eps,
        )
        vl_unclipped = (new_values - returns).pow(2)
        vl_clipped   = (v_clipped  - returns).pow(2)
        value_loss = 0.5 * _masked_mean(torch.max(vl_unclipped, vl_clipped), mask)
    else:
        value_loss = 0.5 * _masked_mean((new_values - returns).pow(2), mask)

    total = policy_loss + value_coef * value_loss

    if entropy is not None and entropy_coef > 0.0:
        # Entropy bonus (maximize entropy -> minimize -entropy).
        ent_loss = -_masked_mean(entropy, mask)
        total = total + entropy_coef * ent_loss
        ent_stat = _masked_mean(entropy, mask).detach().item()
    else:
        ent_stat = 0.0

    with torch.no_grad():
        approx_kl = _masked_mean(old_log_probs - new_log_probs, mask).item()
        clip_frac = _masked_mean(((ratio - 1).abs() > clip_eps).float(), mask).item()

    stats = {
        "policy_loss": policy_loss.detach().item(),
        "value_loss":  value_loss.detach().item(),
        "approx_kl":   approx_kl,
        "clip_frac":   clip_frac,
        "entropy":     ent_stat,
    }
    return total, stats
