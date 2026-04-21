"""M2 sampler — vanilla autoregressive generation with the SAME logit mask
that was used during PPO rollouts in train_m2.py.

Why a separate file? `decode_m1.py` samples from the full vocabulary (pitches,
HOLD, REST, BOS, BAR, PAD). M2 was PPO-trained with REST/BOS/BAR/PAD hard-masked
to -inf at every generation step (cfg.ppo.mask_rest=true), so its policy
distribution was never exposed to those tokens at training time. Sampling from
the unmasked distribution at eval is a distribution mismatch: the model has
mass on tokens it was never rewarded or penalized for, which inflates eval HS
relative to what the training loop actually optimized.

This sampler applies the same mask at inference, so eval matches training.

Run:
    python -m sample.decode_m2 --ckpt checkpoints/m2/best.pt --n 50 \\
        --out_dir samples/m2_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.tokenizer import ChoraleTokenizer, TokenizerConfig
from model.transformer import build_model_from_config
from train._common import load_config
from sample._midi_utils import make_prompt, tokens_to_midi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=None, help="number of samples")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument(
        "--no_mask_rest",
        action="store_true",
        help="disable the REST/BOS/BAR/PAD logit mask (for debugging only)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    n = args.n or cfg["sample"]["n_samples"]
    out_dir = Path(args.out_dir or "samples/m2")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))
    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    blob = torch.load(args.ckpt, map_location=device)
    # strict=False so M2 value_head.* keys load cleanly.
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if unexpected:
        print(f"  ignored unexpected keys (expected value_head.*): {unexpected}")
    if missing:
        print(f"  missing keys (should be empty): {missing}")
    model.eval()

    # Build the same non-musical-token mask the rollout loop uses.
    mask_rest = (not args.no_mask_rest) and cfg["ppo"].get("mask_rest", True)
    if mask_rest:
        forbidden = torch.tensor(
            [tok.REST, tok.BOS, tok.BAR, tok.PAD],
            device=device, dtype=torch.long,
        )

        def _mask_non_musical(next_logits: torch.Tensor, _generated: torch.Tensor):
            next_logits[:, forbidden] = float("-inf")
            return next_logits

        logit_processor = _mask_non_musical
        print(f"  REST/BOS/BAR/PAD masked at inference (matches PPO rollout)")
    else:
        logit_processor = None
        print(f"  no logit mask (vanilla sampling — distribution mismatch vs PPO)")

    piece_length = cfg["sample"]["piece_length"]
    new_tokens = piece_length * 4 - 4   # 4 prompt tokens (one real chord)
    temperature = cfg["sample"]["temperature"]
    top_p = cfg["sample"]["top_p"]

    file_list = out_dir / "file_list.txt"
    with open(file_list, "w") as flist:
        for i in tqdm(range(n), desc="sample"):
            prompt = make_prompt(tok, seed=i).to(device)
            out = model.generate(
                prompt,
                max_new_tokens=new_tokens,
                temperature=temperature,
                top_p=top_p,
                logit_processor=logit_processor,
            )
            tokens = out[0]
            fname = f"sample_{i:04d}.mid"
            tokens_to_midi(tokens, tok, out_dir / fname)
            flist.write(f"{i}\t{fname}\n")

    print(f"wrote {n} samples to {out_dir}")


if __name__ == "__main__":
    main()
