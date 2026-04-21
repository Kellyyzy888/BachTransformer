"""M1 sampler — vanilla autoregressive generation.

Run:
    python -m sample.decode_m1 --ckpt checkpoints/m1/best.pt --n 100
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
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    n = args.n or cfg["sample"]["n_samples"]
    out_dir = Path(args.out_dir or cfg["sample"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))
    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    blob = torch.load(args.ckpt, map_location=device)
    # strict=False so M2 checkpoints (which carry value_head.* keys from the
    # PPO value head) load cleanly into this sampler — we only need the LM.
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if unexpected:
        print(f"  ignored unexpected keys (expected value_head.*): {unexpected}")
    if missing:
        print(f"  missing keys (should be empty): {missing}")
    model.eval()

    piece_length = cfg["sample"]["piece_length"]
    # Prompt is now 4 real SATB tokens (one chord), not 1 BOS. We want the
    # final sequence to be exactly piece_length*4 tokens, so generate the
    # remaining piece_length*4 - 4.
    new_tokens = piece_length * 4 - 4
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
            )
            # keep the prompt — it's a real first chord, not a sentinel
            tokens = out[0]
            fname = f"sample_{i:04d}.mid"
            tokens_to_midi(tokens, tok, out_dir / fname)
            flist.write(f"{i}\t{fname}\n")

    print(f"wrote {n} samples to {out_dir}")


if __name__ == "__main__":
    main()
