"""M1 — baseline trainer: vanilla next-token cross-entropy on JSB Chorales.

Run:
    python -m train.train_m1 --config configs/base.yaml
"""

from __future__ import annotations

import math
import time
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))
    train_loader, val_loader = make_dataloaders(cfg)
    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, vocab={tok.vocab_size}")

    # Class-weighted CE. HOLD is ~49.6% of training tokens; if we let it
    # contribute to the loss at weight 1.0, "always predict HOLD" is an
    # argmin-CE shortcut (CE ~0.70 nats, no learning). Down-weighting HOLD
    # to 0.1 forces the gradient signal onto the pitch distribution.
    # Tune in [0.05, 0.2] — lower if generation emits too many HOLDs.
    hold_weight = float(cfg["train"].get("hold_class_weight", 0.1))
    class_weights = torch.ones(tok.vocab_size, device=device)
    class_weights[tok.HOLD] = hold_weight
    print(f"CE class weights: HOLD={hold_weight}, all others=1.0")

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
    ckpt_dir = cfg["train"]["ckpt_dir"]

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
                logits = model(ids)                              # (B, L, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    weight=class_weights,
                    ignore_index=tok.PAD,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_now:.2e}")
            if step % val_every == 0:
                val_loss, val_pitch = _validate(
                    model, val_loader, device, tok, class_weights
                )
                print(
                    f"[step {step}] val_loss={val_loss:.4f} "
                    f"pitch_only={val_pitch:.4f}"
                )
                # Track checkpoint quality by pitch-only loss — val_loss is
                # weighted and conflates "learned HOLD" with "learned pitch".
                if val_pitch < best_val:
                    best_val = val_pitch
                    save_checkpoint(model, optim, step, val_pitch, ckpt_dir)
                model.train()

    # final checkpoint
    save_checkpoint(model, optim, step, best_val, ckpt_dir, name="last.pt")


@torch.no_grad()
def _validate(model, loader, device, tok, class_weights) -> tuple[float, float]:
    """Returns (weighted_val_loss, pitch_only_val_loss).

    - weighted_val_loss: same CE as training (class-weighted). Matches the
      optimization target and should trend down during training.
    - pitch_only_val_loss: unweighted CE computed ONLY on pitch-token targets
      (PAD, HOLD, REST, BOS, BAR, PAD excluded). This is the number that
      actually tells us whether the model is learning music. Healthy target
      ~1.5-2.2 nats; if it stays ≥2.4, model is near-random on pitches.
    """
    model.eval()
    tot_w, cnt_w = 0.0, 0
    tot_p, cnt_p = 0.0, 0
    pitch_max_tid = tok.cfg.n_pitches  # pitch tokens are ids 0..n_pitches-1
    for batch in loader:
        ids = batch["input_ids"].to(device)
        tgt = batch["target_ids"].to(device)
        logits = model(ids)
        V = logits.size(-1)

        # Weighted CE (matches training loss)
        loss_w = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            weight=class_weights,
            ignore_index=tok.PAD,
            reduction="sum",
        )
        tot_w += loss_w.item()
        cnt_w += (tgt != tok.PAD).sum().item()

        # Pitch-only unweighted CE: mask to positions where target is a real
        # pitch (not HOLD/REST/BOS/BAR/PAD). This is the interpretable metric.
        tgt_flat = tgt.reshape(-1)
        pitch_mask = tgt_flat < pitch_max_tid  # pitch tokens only
        if pitch_mask.any():
            loss_p = F.cross_entropy(
                logits.reshape(-1, V)[pitch_mask],
                tgt_flat[pitch_mask],
                reduction="sum",
            )
            tot_p += loss_p.item()
            cnt_p += int(pitch_mask.sum().item())

    return tot_w / max(1, cnt_w), tot_p / max(1, cnt_p)


if __name__ == "__main__":
    main()
