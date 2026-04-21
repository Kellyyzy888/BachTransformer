"""Shared training utilities: config loading, override parsing, schedulers,
checkpointing. Imported by train_m1 and train_m2.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[],
                    help="dotted-key=value, e.g. train.lr=1e-4")
    return ap.parse_args()


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ovr in overrides:
        if "=" not in ovr:
            raise ValueError(f"bad --override {ovr!r}, want key=value")
        key, val = ovr.split("=", 1)
        _set_dotted(cfg, key, _coerce(val))
    return cfg


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _coerce(val: str) -> Any:
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    except ValueError:
        return val


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(step: int, warmup: int, total: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    if step >= total:
        return peak_lr * 0.1
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(model, optim, step: int, val_loss: float, ckpt_dir: str,
                    name: str = "best.pt") -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        os.path.join(ckpt_dir, name),
    )


def load_checkpoint(model, optim, path: str) -> dict:
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["model"])
    if optim is not None and "optim" in blob:
        optim.load_state_dict(blob["optim"])
    return blob
