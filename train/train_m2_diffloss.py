"""M2 (differentiable rule loss variant) — legacy / comparison baseline.

This is the ORIGINAL M2 plan: CE loss + differentiable rule loss (see
train/rule_loss.py for the soft-rule implementation). We kept it around as
an ablation: it lets us compare "rules as a soft gradient signal applied
everywhere during supervised training" against the PPO/RL version in
train_m2.py.

Run:
    python -m train.train_m2_diffloss --config configs/base.yaml \
        --override rule_loss.enabled=true \
        --override train.ckpt_dir=checkpoints/m2_diffloss
"""

from __future__ import annotations

import math
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])

    assert cfg["rule_loss"]["enabled"], (
        "train_m2 requires rule_loss.enabled=true (use --override)."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))
    train_loader, val_loader = make_dataloaders(cfg)
    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    rule_loss = build_rule_loss_from_config(cfg, tok).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, vocab={tok.vocab_size}")
    print(f"rule_loss λ={cfg['rule_loss']['lambda_total']}, "
          f"weights={cfg['rule_loss']['per_rule_weights']}")

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
                logits = model(ids)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=tok.PAD,
                )
                rl = rule_loss(logits, ids)
                loss = ce + rl["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(
                    ce=f"{ce.item():.3f}",
                    rule=f"{rl['total'].item():.3f}",
                    lr=f"{lr_now:.2e}",
                )
            if step % val_every == 0:
                val_loss = _validate(model, val_loader, device, tok)
                print(f"[step {step}] val_loss={val_loss:.4f}  "
                      f"per_rule="
                      f"{ {k: round(v.item(),4) for k,v in rl.items() if k!='total'} }")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, optim, step, val_loss, ckpt_dir)
                model.train()

    save_checkpoint(model, optim, step, best_val, ckpt_dir, name="last.pt")


@torch.no_grad()
def _validate(model, loader, device, tok) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        tgt = batch["target_ids"].to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=tok.PAD,
            reduction="sum",
        )
        tot += loss.item()
        cnt += (tgt != tok.PAD).sum().item()
    return tot / max(1, cnt)


if __name__ == "__main__":
    main()
