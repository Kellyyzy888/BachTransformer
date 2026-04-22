"""M2 — PPO fine-tuning of the M1 checkpoint with the rule checker as reward.

This is the RL-as-rule-injection version of M2. It is a custom, from-scratch
PPO implementation targeting our symbolic chorale setup.

Pipeline per update:
  1. Rollout: sample a batch of full 64-step chorales from the current policy.
  2. Score: `train._ppo.score_tokens_np` runs the rule checker on each.
  3. Reward assembly:
       r_t = -β · (log π_rl(a_t|s_t) − log π_ref(a_t|s_t))        (per-token KL penalty)
       r_{T-1} += -HarmonicScore                                    (terminal rule reward)
  4. GAE: compute per-token advantages with γ, λ.
  5. PPO update: `n_epochs` of minibatched clipped-surrogate + value-loss steps.
  6. Adaptive β: nudge KL weight toward a target KL.

The frozen M1 weights are held in a second model (`ref`) — same architecture
but without the value head. The policy is a `ChoraleTransformer` built with
`use_value_head=True`; M1 weights load via `strict=False` so the value head
starts from its random init.

Run:
    python -m train.train_m2 --config configs/base.yaml \\
        --override ppo.init_ckpt=checkpoints/m1/best.pt \\
        --override ppo.ckpt_dir=checkpoints/m2
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import ChoraleTokenizer, TokenizerConfig
from model.transformer import build_model_from_config
from sample._midi_utils import make_prompt
from train._common import load_config, parse_args, save_checkpoint, set_seed
from train._ppo import compute_gae, ppo_losses, score_token_trajectory_np


# ---------------------------------------------------------------------------
# Rollout: generate samples and assemble per-token rewards.
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(
    policy: nn.Module,
    ref: nn.Module,
    tok: ChoraleTokenizer,
    cfg: dict,
    device: torch.device,
    beta_kl: float,
) -> dict:
    """Generate, score, and pack one rollout batch.

    Returns a dict with everything the PPO update needs.
    """
    B = cfg["ppo"]["rollout_batch"]
    piece_length = cfg["sample"]["piece_length"]
    prompt_len = 4
    T_full = piece_length * 4
    gen_len = T_full - prompt_len

    policy.eval()
    ref.eval()

    # Prompt with real openers (seed=None -> different every rollout, i.e. diverse starts)
    prompt = make_prompt(tok, batch_size=B, seed=None).to(device)

    # Rollout-time logit mask. REST is the mode-collapse attractor: under a
    # pure -HS reward, the policy will learn that silence = zero rule
    # violations = globally optimal reward (observed in run 1804127, where
    # M2 files collapsed to 171 bytes of near-silence vs M1's ~900 bytes of
    # musical content). We hard-mask REST during generation so the attractor
    # is physically unreachable. BOS/BAR/PAD are dataset-level tokens that
    # shouldn't appear mid-generation anyway. HOLD is preserved — sustained
    # notes are musically legitimate in Bach chorales.
    #
    # The log-prob bookkeeping below re-scores tokens under the *unmasked*
    # softmax (same as with nucleus sampling), so importance ratios remain
    # self-consistent across rollout and update.
    if cfg["ppo"].get("mask_rest", True):
        _forbidden = torch.tensor(
            [tok.REST, tok.BOS, tok.BAR, tok.PAD], device=device, dtype=torch.long,
        )
        def _mask_non_musical(next_logits: torch.Tensor, _generated: torch.Tensor):
            next_logits[:, _forbidden] = float("-inf")
            return next_logits
        logit_processor = _mask_non_musical
    else:
        logit_processor = None

    # Sample with the same nucleus we use for the M1 baseline eval (top_p=0.95).
    # Sampling at top_p=1.0 pulls in low-prob garbage tokens that spike HS at
    # update 0 (~14 vs M1's 4.7) and pollutes the whole RL signal. The log-prob
    # bookkeeping below is still exact: we gather log_softmax(full logits) at
    # the sampled tokens, which gives the correct importance-ratio denominator
    # for PPO regardless of whether sampling was done with a truncated nucleus.
    tokens = policy.generate(
        prompt,
        max_new_tokens=gen_len,
        temperature=cfg["ppo"].get("rollout_temperature", 1.0),
        top_p=cfg["ppo"].get("rollout_top_p", 0.95),
        logit_processor=logit_processor,
    )                                              # (B, T_full)

    # Single forward pass through both policy and ref on tokens[:, :-1] gives us
    # logits/values aligned to predict tokens[:, 1:]. We only need positions
    # where a generated token sits (p in [prompt_len, T_full)).
    #
    # The policy is in eval() here, which disables dropout. We also need AMP
    # off for the value head's small gradients to not underflow. Rollout is
    # under no_grad so this is fine anyway.
    logits_p, values_p = policy(tokens[:, :-1], return_values=True)   # (B, T_full-1, V), (B, T_full-1)
    logits_r = ref(tokens[:, :-1])                                    # (B, T_full-1, V)

    log_probs_p = F.log_softmax(logits_p, dim=-1)
    log_probs_r = F.log_softmax(logits_r, dim=-1)

    # Pull log-probs of the tokens actually generated.
    target = tokens[:, 1:].unsqueeze(-1)                              # (B, T_full-1, 1)
    gathered_p = log_probs_p.gather(-1, target).squeeze(-1)           # (B, T_full-1)
    gathered_r = log_probs_r.gather(-1, target).squeeze(-1)           # (B, T_full-1)

    # Repack to length T_full with index-0 unused (no one predicts token 0).
    old_log_probs = torch.zeros(B, T_full, device=device)
    ref_log_probs = torch.zeros(B, T_full, device=device)
    old_values    = torch.zeros(B, T_full, device=device)

    old_log_probs[:, 1:] = gathered_p
    ref_log_probs[:, 1:] = gathered_r
    # values_p[:, t] = V(tokens[:, :t+1]) = V(s_{t+1}). We want old_values[:, p] = V(s_p).
    # => old_values[:, p] = values_p[:, p-1] for p in 1..T_full-1.
    old_values[:, 1:] = values_p

    # Value for the terminal state (everything generated): 0, episodic MDP.
    values_for_gae = torch.zeros(B, T_full + 1, device=device)
    values_for_gae[:, :T_full] = old_values
    # values_for_gae[:, T_full] is already 0.

    # Mask: 1 at generated positions (prompt_len..T_full-1), 0 at prompt positions.
    mask = torch.zeros(B, T_full, device=device)
    mask[:, prompt_len:] = 1.0

    # Score each rollout via the in-memory rule checker.
    # Two signals are tracked:
    #   - harmonic_scores: unweighted HS, kept comparable to eval.
    #   - local_step_rewards: rule penalties assigned to the timestep where the
    #     violation first appears. This sharpens PPO credit assignment versus
    #     smearing one terminal reward across the whole sequence.
    harmonic_scores = np.zeros(B, dtype=np.float32)
    local_step_rewards = np.zeros((B, piece_length), dtype=np.float32)
    rule_reward_weights = cfg["ppo"].get("rule_reward_weights", None)
    local_reward_scale = float(cfg["ppo"].get("local_rule_reward_scale", 1.0))
    use_local_rule_reward = bool(cfg["ppo"].get("use_local_rule_reward", True))
    totals = {"parallel_5ths": 0, "parallel_8ves": 0, "voice_crossings": 0,
              "hidden_5ths_outer": 0, "hidden_8ves_outer": 0,
              "spacing_violations": 0, "large_leaps": 0, "augmented_leaps": 0}
    for b in range(B):
        stats = score_token_trajectory_np(
            tokens[b].cpu().numpy(),
            tok,
            rule_reward_weights=rule_reward_weights,
        )
        harmonic_scores[b] = stats["HarmonicScore"]
        local_step_rewards[b] = -local_reward_scale * stats["weighted_penalty_by_timestep"]
        for k in totals:
            totals[k] += stats[k]
    hs_t = torch.tensor(harmonic_scores, device=device)               # (B,)
    local_step_rewards_t = torch.tensor(local_step_rewards, device=device)  # (B, piece_length)

    # Reward assembly.
    #   KL term: per generated token, -β * (log π - log π_ref). Sample-based
    #            KL estimator. Positive means policy assigns more prob than ref.
    kl_per_token = old_log_probs - ref_log_probs                     # (B, T_full)
    rewards = -beta_kl * kl_per_token * mask                         # zero at prompt

    if use_local_rule_reward:
        # Map timestep penalties onto the four generated tokens of that step.
        # Timestep 0 is the prompt chord, so only timesteps 1..piece_length-1
        # participate in PPO reward shaping.
        local_token_rewards = torch.zeros(B, T_full, device=device)
        for t in range(1, piece_length):
            start = 4 * t
            end = start + 4
            local_token_rewards[:, start:end] = (
                local_step_rewards_t[:, t].unsqueeze(1) / 4.0
            )
        rewards = rewards + local_token_rewards * mask
    else:
        # Compatibility fallback: smear the weighted episode penalty across all
        # generated tokens, matching the previous dense-HS behavior.
        weighted_total = local_step_rewards_t[:, 1:].sum(dim=1).abs()
        hs_dense = (-weighted_total).unsqueeze(1) / float(gen_len)
        rewards = rewards + hs_dense * mask

    # Pitch-emission bonus — destroys the silence attractor. Under -HS alone,
    # "emit REST everywhere" is a global reward minimum (no notes = no rule
    # violations = HS 0). Giving every pitch token a small positive bonus
    # c = pitch_bonus (≈ 0.02) shifts that minimum: filling every slot with
    # a pitch is now worth +c·gen_len ≈ +5 per rollout, comparable to the
    # HS magnitude we're fighting. Combined with the hard REST mask above
    # this double-gates mode collapse. HOLD counts as non-pitch and gets
    # no bonus, so the policy also can't trivially game via "hold the first
    # chord forever." Token IDs 0..n_pitches-1 are pitches; HOLD = n_pitches.
    # (MusicRL 2024 arxiv:2402.04229; Bach2Bach 2018 arxiv:1812.01060.)
    pitch_bonus_c = float(cfg["ppo"].get("pitch_bonus", 0.0))
    if pitch_bonus_c > 0.0:
        is_pitch = (tokens >= 0) & (tokens < tok.cfg.n_pitches)      # (B, T_full) bool
        rewards = rewards + is_pitch.float() * pitch_bonus_c * mask

    # GAE.
    advantages, returns = compute_gae(
        rewards, values_for_gae,
        gamma=cfg["ppo"]["gamma"],
        lam=cfg["ppo"]["lam"],
    )

    # Token-class telemetry over generated positions only — this is the early
    # warning for mode collapse. If rest_frac climbs above ~0.1 or hold_frac
    # above ~0.7, the policy is drifting toward silence or chord-drone. With
    # mask_rest=true and pitch_bonus>0, expect pitch_frac ~ 0.5, hold_frac ~ 0.5.
    gen_tokens = tokens[:, prompt_len:]
    pitch_frac = float(((gen_tokens >= 0) & (gen_tokens < tok.cfg.n_pitches)).float().mean().item())
    hold_frac  = float((gen_tokens == tok.HOLD).float().mean().item())
    rest_frac  = float((gen_tokens == tok.REST).float().mean().item())

    return {
        "tokens":        tokens,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "old_values":    old_values,
        "advantages":    advantages,
        "returns":       returns,
        "rewards":       rewards,
        "mask":          mask,
        "harmonic_scores": hs_t,
        "local_step_rewards": local_step_rewards_t,
        "kl_per_token":    kl_per_token,
        "rule_totals":     totals,
        "token_frac":      {"pitch": pitch_frac, "hold": hold_frac, "rest": rest_frac},
    }


# ---------------------------------------------------------------------------
# PPO update: n_epochs over minibatches.
# ---------------------------------------------------------------------------

def ppo_update(
    policy: nn.Module,
    optim: torch.optim.Optimizer,
    batch: dict,
    cfg: dict,
) -> dict:
    tokens        = batch["tokens"]
    old_log_probs = batch["old_log_probs"]
    old_values    = batch["old_values"]
    advantages    = batch["advantages"]
    returns       = batch["returns"]
    mask          = batch["mask"]

    B = tokens.size(0)
    T_full = tokens.size(1)
    mb_size  = cfg["ppo"]["minibatch_size"]
    n_epochs = cfg["ppo"]["n_epochs"]
    clip_eps = cfg["ppo"]["clip_eps"]
    value_coef  = cfg["ppo"]["value_coef"]
    value_clip  = cfg["ppo"].get("value_clip_eps", None)
    entropy_coef = cfg["ppo"].get("entropy_coef", 0.0)
    grad_clip = cfg["train"]["grad_clip"]
    max_kl = cfg["ppo"].get("early_stop_kl", None)   # if mean KL exceeds, break epochs

    agg = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0,
           "clip_frac": 0.0, "entropy": 0.0, "n": 0}

    # Keep dropout OFF during PPO updates: rollout log-probs (old_log_probs)
    # were collected with dropout off, so enabling it here would make the
    # ratio `exp(new − old)` drift from 1 on epoch 0 due to dropout noise
    # alone (not actual policy change). Standard in RLHF-PPO implementations.
    policy.eval()
    # Re-enable grads on the policy parameters (eval() only flips dropout/BN).
    done = False
    for epoch in range(n_epochs):
        perm = torch.randperm(B, device=tokens.device)
        for i in range(0, B, mb_size):
            idx = perm[i : i + mb_size]
            mb_tokens  = tokens[idx]
            mb_old_lp  = old_log_probs[idx]
            mb_old_val = old_values[idx]
            mb_adv     = advantages[idx]
            mb_ret     = returns[idx]
            mb_mask    = mask[idx]

            logits, values = policy(mb_tokens[:, :-1], return_values=True)
            log_probs = F.log_softmax(logits, dim=-1)

            target = mb_tokens[:, 1:].unsqueeze(-1)                    # (mb, T_full-1, 1)
            new_lp_gather = log_probs.gather(-1, target).squeeze(-1)   # (mb, T_full-1)

            new_lp     = torch.zeros_like(mb_old_lp)
            new_values = torch.zeros_like(mb_old_lp)
            new_lp[:, 1:]     = new_lp_gather
            new_values[:, 1:] = values

            # Entropy of the predictive distribution at each step.
            # -sum p * log p, averaged over valid positions (added to loss below).
            if entropy_coef > 0.0:
                probs = log_probs.exp()
                ent_per_pos = -(probs * log_probs).sum(-1)             # (mb, T_full-1)
                entropy = torch.zeros_like(mb_old_lp)
                entropy[:, 1:] = ent_per_pos
            else:
                entropy = None

            loss, stats = ppo_losses(
                new_log_probs=new_lp,
                old_log_probs=mb_old_lp,
                advantages=mb_adv,
                new_values=new_values,
                returns=mb_ret,
                mask=mb_mask,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy=entropy,
                entropy_coef=entropy_coef,
                value_clip_eps=value_clip,
                old_values=mb_old_val if value_clip is not None else None,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optim.step()

            for k in ("policy_loss", "value_loss", "approx_kl", "clip_frac", "entropy"):
                agg[k] += stats[k]
            agg["n"] += 1

            if max_kl is not None and stats["approx_kl"] > max_kl:
                done = True
                break
        if done:
            break

    if agg["n"] > 0:
        for k in ("policy_loss", "value_loss", "approx_kl", "clip_frac", "entropy"):
            agg[k] /= agg["n"]
    return agg


# ---------------------------------------------------------------------------
# Value-head warmup (optional, strongly recommended).
# ---------------------------------------------------------------------------

def warmup_value_head(
    policy: nn.Module,
    ref: nn.Module,
    tok: ChoraleTokenizer,
    cfg: dict,
    device: torch.device,
    beta_kl: float,
    n_warmup: int,
    lr: float,
) -> None:
    """Pre-train the value head with the rest of the policy frozen.

    At step 0 of PPO, V(s) is random-init, so every advantage is essentially
    noise. PPO then makes `n_epochs` of policy-gradient updates against that
    noise, which shoves the policy in arbitrary directions before the critic
    has caught up. The standard fix is a critic warmup: freeze the policy,
    run a few rollouts, fit V(s) to the observed returns via MSE.

    With the detached value head (see model/transformer.py) this is doubly
    safe — even without freezing, the critic's gradients can't reach the
    backbone. We freeze explicitly anyway to make intent obvious and to skip
    unnecessary optimizer work on the transformer params.
    """
    # Freeze everything, then re-enable grads only on value_head.
    for p in policy.parameters():
        p.requires_grad_(False)
    for p in policy.value_head.parameters():
        p.requires_grad_(True)

    v_optim = torch.optim.AdamW(policy.value_head.parameters(), lr=lr)
    policy.eval()
    grad_clip = cfg["train"]["grad_clip"]

    print(f"[warmup] training value head only for {n_warmup} updates @ lr={lr:.0e}")
    for step in range(n_warmup):
        # rollout() is under torch.no_grad, so it returns tokens/returns with
        # no grad history. We redo a grad-enabled forward below just for V.
        batch = rollout(policy, ref, tok, cfg, device, beta_kl=beta_kl)
        tokens  = batch["tokens"]
        returns = batch["returns"]
        mask    = batch["mask"]

        B = tokens.size(0)
        T_full = tokens.size(1)

        _, values = policy(tokens[:, :-1], return_values=True)   # (B, T_full-1)
        new_values = torch.zeros(B, T_full, device=device)
        new_values[:, 1:] = values                                # V(s_p) = values[:, p-1]

        denom = mask.sum().clamp_min(1.0)
        v_loss = ((new_values - returns).pow(2) * mask).sum() / denom

        v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        nn.utils.clip_grad_norm_(policy.value_head.parameters(), grad_clip)
        v_optim.step()

        if step % max(1, n_warmup // 10) == 0 or step == n_warmup - 1:
            mean_hs = float(batch["harmonic_scores"].mean().item())
            mean_r  = float((returns * mask).sum().item() / denom.item())
            print(
                f"[warmup {step:03d}] v_loss={v_loss.item():.3f}  "
                f"mean_return={mean_r:+.2f}  mean_hs={mean_hs:.2f}"
            )

    # Unfreeze the full policy for the main PPO loop.
    for p in policy.parameters():
        p.requires_grad_(True)
    print("[warmup] done — full policy unfrozen, entering PPO.")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])

    assert "ppo" in cfg, "M2 requires a `ppo:` section in the config."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))

    # --- Policy: built WITH value head. Load M1 weights (strict=False skips the value head).
    policy = build_model_from_config(
        cfg, vocab_size=tok.vocab_size, use_value_head=True,
    ).to(device)
    init_ckpt = cfg["ppo"]["init_ckpt"]
    blob = torch.load(init_ckpt, map_location=device)
    missing, unexpected = policy.load_state_dict(blob["model"], strict=False)
    # Expected: missing should include "value_head.*", unexpected should be empty.
    print(f"policy loaded from {init_ckpt}")
    print(f"  missing keys (expected: value_head.*): {missing}")
    print(f"  unexpected keys (expected: none):      {unexpected}")

    # --- Reference: same M1 weights, frozen, NO value head (we only need log-probs).
    ref = build_model_from_config(
        cfg, vocab_size=tok.vocab_size, use_value_head=False,
    ).to(device)
    ref.load_state_dict(blob["model"], strict=True)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"policy: {n_params/1e6:.2f}M params ({n_trainable/1e6:.2f}M trainable)")
    print(f"ref: frozen M1, {sum(p.numel() for p in ref.parameters())/1e6:.2f}M params")

    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg["ppo"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )

    beta_kl      = float(cfg["ppo"]["beta_kl"])
    target_kl    = float(cfg["ppo"]["target_kl"])
    adaptive_beta = bool(cfg["ppo"].get("adaptive_beta", True))
    beta_lo, beta_hi = cfg["ppo"].get("beta_range", [1e-4, 1.0])

    n_updates = int(cfg["ppo"]["n_updates"])
    log_every = int(cfg["ppo"]["log_every"])
    ckpt_every = int(cfg["ppo"]["ckpt_every"])
    ckpt_dir  = cfg["ppo"]["ckpt_dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # --- Value-head warmup. Default 30 updates; disable with 0.
    n_warmup = int(cfg["ppo"].get("value_warmup_updates", 30))
    warmup_lr = float(cfg["ppo"].get("value_warmup_lr", 1e-4))
    if n_warmup > 0:
        warmup_value_head(
            policy, ref, tok, cfg, device,
            beta_kl=beta_kl,
            n_warmup=n_warmup,
            lr=warmup_lr,
        )

    best_mean_hs = math.inf
    running_mean_hs = None

    for step in range(n_updates):
        t0 = time.time()
        batch = rollout(policy, ref, tok, cfg, device, beta_kl=beta_kl)
        ps = ppo_update(policy, optim, batch, cfg)
        dt = time.time() - t0

        mean_hs = float(batch["harmonic_scores"].mean().item())
        # Per-token KL averaged over generated positions.
        m = batch["mask"]
        mean_kl = float(
            (batch["kl_per_token"] * m).sum().item() / m.sum().clamp_min(1.0).item()
        )
        totals = batch["rule_totals"]
        tfrac  = batch["token_frac"]

        running_mean_hs = mean_hs if running_mean_hs is None else 0.9 * running_mean_hs + 0.1 * mean_hs

        # Adaptive β (Schulman-style).
        if adaptive_beta:
            if mean_kl > 1.5 * target_kl:
                beta_kl = min(beta_hi, beta_kl * 1.5)
            elif mean_kl < 0.5 * target_kl:
                beta_kl = max(beta_lo, beta_kl / 1.5)

        if step % log_every == 0 or step == n_updates - 1:
            print(
                f"[upd {step:04d}] "
                f"hs={mean_hs:.2f} (ema={running_mean_hs:.2f}) "
                f"kl={mean_kl:+.4f} β={beta_kl:.4f} "
                f"pit={tfrac['pitch']:.2f} hld={tfrac['hold']:.2f} rst={tfrac['rest']:.2f} "
                f"P5={totals['parallel_5ths']} P8={totals['parallel_8ves']} "
                f"VC={totals['voice_crossings']} Sp={totals['spacing_violations']} "
                f"Lp={totals['large_leaps']} Tri={totals['augmented_leaps']} "
                f"| pl={ps['policy_loss']:+.3f} vl={ps['value_loss']:.3f} "
                f"cf={ps['clip_frac']:.2f} ent={ps['entropy']:.2f} "
                f"t={dt:.1f}s"
            )

        if (step + 1) % ckpt_every == 0 or step == n_updates - 1:
            # Best by EMA of mean HS (smoother than per-rollout).
            if running_mean_hs < best_mean_hs:
                best_mean_hs = running_mean_hs
                save_checkpoint(policy, optim, step, running_mean_hs, ckpt_dir, name="best.pt")
                print(f"  [ckpt] new best EMA mean HS = {best_mean_hs:.3f} -> best.pt")
            save_checkpoint(policy, optim, step, running_mean_hs, ckpt_dir, name="last.pt")

    print(f"done. best EMA mean HS = {best_mean_hs:.3f}")


if __name__ == "__main__":
    main()
