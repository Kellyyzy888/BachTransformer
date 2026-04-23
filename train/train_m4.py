"""M4 — chord-conditioned trainer: vanilla CE on the interleaved sequence.

Differences from train_m1.py
----------------------------
- Expects `chord.enabled=true` in the config. The dataloader yields
  interleaved [KEY, RN, S, A, T, B, ...] sequences (length 2 + 5*T).
- Tokenizer is in chord mode (extended vocab: n_pitches + 5 specials +
  2 key-mode + 12 tonic-pc + K Roman-numeral tokens).
- Class weights now differentiate HOLD / KEY / RN / pitch (see the
  `train_m4.class_weights` block in base.yaml). KEY tokens appear twice
  per chorale (2 prefix tokens) and would otherwise dominate loss at
  position 0 / 1; we down-weight them accordingly.
- The model is built with `chord_layout=True`, which switches the
  positional module to 5-voice-role and the max_seq_len to 2 + T*5.
- If `chord.transpose_augment=true`, every chorale yields 12
  transpositions per epoch (the dataloader handles this; see
  `DEFAULT_SHIFTS` in data/jsb_loader.py). Roman numerals are
  key-relative and stay invariant under transposition — the tonic
  pitch-class token is the one that shifts.

Run:
    # Step 1 (one-time): extract chord labels from the JSB pickle.
    python -m data.chord_extractor \
        --in  JSB-Chorales-dataset/jsb-chorales-16th.pkl \
        --out data/jsb_chords.pkl

    # Step 2: train.
    python -m train.train_m4 --config configs/base.yaml \
        --override chord.enabled=true
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.jsb_loader import make_dataloaders
from data.tokenizer import ChoraleTokenizer, TokenizerConfig
from model.transformer import build_model_from_config
from train._common import (load_config, parse_args, save_checkpoint, set_seed,
                           warmup_cosine)
from train.rule_loss import build_rule_loss_from_config


def _build_class_weights(
    tok: ChoraleTokenizer,
    device: torch.device,
    cfg_weights: dict,
) -> torch.Tensor:
    """Per-class CE weights with the M4 token-family breakdown.

    Families (all indexable off the tokenizer):
      - pitch tokens  [0, n_pitches)                -> `pitch`
      - HOLD                                         -> `hold`
      - REST, BOS, BAR, PAD                          -> `pitch` (same scale)
      - KEY_MAJOR, KEY_MINOR, PC_0..PC_11            -> `key`
      - RN_0..RN_{K-1}                               -> `rn`

    PAD stays at the default 1.0 but is later zeroed via `ignore_index`.
    """
    w = torch.ones(tok.vocab_size, device=device)
    w[tok.HOLD] = float(cfg_weights.get("hold", 0.1))

    # Key block (2 mode tokens + 12 tonic-pc tokens).
    key_w = float(cfg_weights.get("key", 0.05))
    w[tok.KEY_MAJOR] = key_w
    w[tok.KEY_MINOR] = key_w
    for i in range(12):
        w[tok.PC_BASE + i] = key_w

    # RN block.
    rn_w = float(cfg_weights.get("rn", 1.0))
    for i in range(tok.cfg.n_chord):
        w[tok.RN_BASE + i] = rn_w

    # pitch_w is the implicit default (already 1.0). Scale down
    # explicitly if the user asked for < 1 in config.
    pitch_w = float(cfg_weights.get("pitch", 1.0))
    for i in range(tok.cfg.n_pitches):
        w[i] = pitch_w
    return w


def _extract_voice_only_view(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert M4's chord-interleaved sequence into a voice-only SATB view."""
    full_seq = torch.cat([input_ids[:, :1], target_ids], dim=1)
    full_len = full_seq.size(1)
    positions = torch.arange(full_len, device=full_seq.device)
    voice_mask = (positions >= 3) & (((positions - 2) % 5) != 0)
    voice_pos = positions[voice_mask]
    logit_pos = voice_pos - 1
    voice_logits = logits.index_select(1, logit_pos)
    voice_ids = full_seq.index_select(1, voice_pos)
    return voice_logits, voice_ids


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    # M4 requires chord mode; enforce here so we fail fast if someone runs
    # `python -m train.train_m4` without flipping the flag.
    if not cfg.get("chord", {}).get("enabled", False):
        print(
            "[train_m4] chord.enabled=false in config; did you mean to run "
            "train_m1? Either flip it in base.yaml or pass "
            "--override chord.enabled=true"
        )
    # Force-enable for the rest of this script regardless, so the
    # dataloader / tokenizer / model are in a consistent configuration.
    cfg.setdefault("chord", {})["enabled"] = True

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the RN vocab from the chord cache *once* so we can pass it to
    # both the tokenizer and the model vocab_size consistently.
    chord_cache = cfg["chord"]["cache_path"]
    with open(chord_cache, "rb") as f:
        chord_payload = pickle.load(f)
    chord_vocab = list(chord_payload["vocab"])
    print(f"loaded chord vocab: {len(chord_vocab)} RN symbols "
          f"(cache={chord_cache})")

    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
        chord_vocab=chord_vocab,
    ))
    print(f"tokenizer vocab_size={tok.vocab_size} "
          f"(pitch={tok.cfg.n_pitches} + specials=5 + key=14 + RN={len(chord_vocab)})")

    train_loader, val_loader = make_dataloaders(cfg)
    print(f"train chunks: {len(train_loader.dataset)}  "
          f"val chunks: {len(val_loader.dataset)}  "
          f"(batch={cfg['data']['batch_size']})")

    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, max_seq_len={model.cfg.max_seq_len}")

    m4_cfg = cfg.get("train_m4", {})
    class_weights = _build_class_weights(
        tok, device, m4_cfg.get("class_weights", {})
    )
    print(
        "CE class weights: HOLD={:.3f}, KEY={:.3f}, RN={:.3f}, pitch={:.3f}"
        .format(
            class_weights[tok.HOLD].item(),
            class_weights[tok.KEY_MAJOR].item(),
            class_weights[tok.RN_BASE].item() if tok.cfg.n_chord > 0 else float("nan"),
            class_weights[0].item(),
        )
    )

    use_rule_loss = bool(cfg.get("rule_loss", {}).get("enabled", False))
    rule_loss = None
    if use_rule_loss:
        rule_loss = build_rule_loss_from_config(cfg, tok).to(device)
        print(
            f"soft rule loss enabled: lambda={cfg['rule_loss']['lambda_total']} "
            f"weights={cfg['rule_loss']['per_rule_weights']}"
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"])

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    peak_lr = cfg["train"]["lr"]
    warmup = cfg["train"]["warmup_steps"]
    grad_clip = cfg["train"]["grad_clip"]
    log_every = cfg["train"]["log_every"]
    val_every = cfg["train"]["val_every"]
    ckpt_dir = m4_cfg.get("ckpt_dir", "checkpoints/m4")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["target_ids"].to(device, non_blocking=True)

            lr_now = warmup_cosine(step, warmup, total_steps, peak_lr)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["train"]["mixed_precision"]):
                logits = model(ids)                                    # (B, L, V)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    weight=class_weights,
                    ignore_index=tok.PAD,
                )
                if use_rule_loss:
                    voice_logits, voice_ids = _extract_voice_only_view(logits, ids, tgt)
                    rl = rule_loss(voice_logits, voice_ids)
                    loss = ce + rl["total"]
                else:
                    rl = None
                    loss = ce

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            step += 1
            if step % log_every == 0:
                if rl is None:
                    pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_now:.2e}")
                else:
                    pbar.set_postfix(
                        ce=f"{ce.item():.3f}",
                        rule=f"{rl['total'].item():.3f}",
                        loss=f"{loss.item():.3f}",
                        lr=f"{lr_now:.2e}",
                    )
            if step % val_every == 0:
                val_w, val_pitch, val_rn, val_rule = _validate(
                    model, val_loader, device, tok, class_weights, rule_loss
                )
                msg = (
                    f"[step {step}] val_weighted={val_w:.4f} "
                    f"pitch_only={val_pitch:.4f}  rn_only={val_rn:.4f}"
                )
                if val_rule is not None:
                    msg += f"  rule_only={val_rule:.4f}"
                print(msg)
                # Track best checkpoint by pitch-only CE so we're comparing
                # apples to apples with M1's metric. rn_only is reported
                # for diagnostic purposes — if it's suspiciously low,
                # the model is probably overfitting the RN prior.
                if val_pitch < best_val:
                    best_val = val_pitch
                    save_checkpoint(model, optim, step, val_pitch, ckpt_dir)
                model.train()

    save_checkpoint(model, optim, step, best_val, ckpt_dir, name="last.pt")


@torch.no_grad()
def _validate(model, loader, device, tok, class_weights, rule_loss=None) -> tuple[float, float, float, float | None]:
    """Returns (weighted_val_loss, pitch_only_val_loss, rn_only_val_loss, rule_only).

    The weighted CE matches training. The two unweighted slices:
      - pitch_only: only where target is a real pitch token (0..n_pitches-1)
      - rn_only:    only where target is a Roman-numeral token
    Together they tell us whether the model is learning *music* (pitch CE
    should be comparable to M1) AND *harmony* (rn CE should be low if
    chord conditioning is actually being used).
    """
    model.eval()
    tot_w, cnt_w = 0.0, 0
    tot_p, cnt_p = 0.0, 0
    tot_r, cnt_r = 0.0, 0
    tot_rule, cnt_rule = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        tgt = batch["target_ids"].to(device)
        logits = model(ids)
        V = logits.size(-1)

        loss_w = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            weight=class_weights,
            ignore_index=tok.PAD,
            reduction="sum",
        )
        tot_w += loss_w.item()
        cnt_w += (tgt != tok.PAD).sum().item()

        tgt_flat = tgt.reshape(-1)
        logits_flat = logits.reshape(-1, V)

        # Pitch-only.
        pitch_mask = tgt_flat < tok.cfg.n_pitches
        if pitch_mask.any():
            lp = F.cross_entropy(
                logits_flat[pitch_mask],
                tgt_flat[pitch_mask],
                reduction="sum",
            )
            tot_p += lp.item()
            cnt_p += int(pitch_mask.sum().item())

        # RN-only.
        rn_mask = (tgt_flat >= tok.RN_BASE) & (tgt_flat < tok.RN_BASE + tok.cfg.n_chord)
        if rn_mask.any():
            lr = F.cross_entropy(
                logits_flat[rn_mask],
                tgt_flat[rn_mask],
                reduction="sum",
            )
            tot_r += lr.item()
            cnt_r += int(rn_mask.sum().item())

        if rule_loss is not None:
            voice_logits, voice_ids = _extract_voice_only_view(logits, ids, tgt)
            rl = rule_loss(voice_logits, voice_ids)
            tot_rule += float(rl["total"].item())
            cnt_rule += 1

    return (
        tot_w / max(1, cnt_w),
        tot_p / max(1, cnt_p),
        tot_r / max(1, cnt_r),
        (tot_rule / max(1, cnt_rule)) if rule_loss is not None else None,
    )


if __name__ == "__main__":
    main()
