"""M3 sampler — constrained decoding via logit shaping.

At each step we look at the current partial sequence and down-weight
logits that would create an immediate voice-leading violation. This is
the inference-time analogue of the training-time rule loss in M2.

Run:
    python -m sample.decode_m3 --ckpt checkpoints/m1/best.pt --n 100 \
        --out_dir samples/m3 --alpha 5.0
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


class ConstrainedProcessor:
    """Callable that maps (logits, generated_so_far) -> shaped_logits.

    IMPORTANT: resolves HOLD tokens to their underlying sustained pitch before
    applying rules. An earlier version only checked raw token values; since
    HOLD is ~50% of training tokens, ~87% of parallel-check opportunities were
    silently skipped (any of {prev_upper, prev_self, cur_upper} being HOLD
    short-circuited the rule), which made alpha tuning ineffective.

    Three rules implemented here (the rest are training-only via M2):
      1. voice crossing
      2. large leaps (> max_leap semitones)
      3. parallel 5ths / 8ves
    """

    def __init__(self, tok: ChoraleTokenizer, alpha: float = 5.0,
                 max_leap: int = 12):
        self.tok = tok
        self.alpha = alpha
        self.max_leap = max_leap
        self.P = tok.cfg.n_pitches
        self.pmin = tok.cfg.pitch_min

    def _effective_pitches(self, seq: torch.Tensor) -> list:
        """For each voice 0..3, walk backward through `seq` and return the
        most recent pitch token at a position p where p % 4 == voice.
        Returns None for voices that never emitted a pitch, or whose last
        non-HOLD event was a REST.
        """
        result = [None] * 4
        L = seq.size(0)
        for voice in range(4):
            p = L - 1
            # skip positions not belonging to this voice
            while p >= 0 and (p % 4) != voice:
                p -= 1
            while p >= 0:
                tid = int(seq[p].item())
                if 0 <= tid < self.P:           # real pitch token
                    result[voice] = tid
                    break
                if tid == self.tok.REST:        # rest wipes the voice
                    break
                # HOLD / BOS / BAR / PAD — keep walking backward
                p -= 4
        return result

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape
        position = generated.size(1)        # next index to fill
        voice = position % 4                # 0=S 1=A 2=T 3=B
        base = position - voice             # start of current (incomplete) chord
        prev_timestep_end = base            # exclusive end of last complete chord
        if prev_timestep_end < 4:
            return logits                   # no prior chord yet

        device = logits.device
        penalty = torch.zeros(B, V, device=device)
        cand = torch.arange(self.P, device=device)

        for b in range(B):
            # prev[v] = voice v's pitch at the end of the last complete timestep
            # cur[v]  = voice v's pitch right now (for v < voice: just emitted
            #           this chord; for v >= voice: inherited from prev timestep)
            prev = self._effective_pitches(generated[b, :prev_timestep_end])
            cur  = self._effective_pitches(generated[b, :position])

            prev_v = prev[voice]

            # 1. large leap vs prev pitch in same voice
            if prev_v is not None:
                leap_mask = (cand - prev_v).abs() > self.max_leap
                penalty[b, : self.P] = penalty[b, : self.P] + leap_mask.float()

            # 2. voice crossing: this voice's pitch must be <= upper voice
            #    just-emitted in the current (incomplete) chord
            if voice >= 1 and cur[voice - 1] is not None:
                upper = cur[voice - 1]
                penalty[b, : self.P] = penalty[b, : self.P] + (cand > upper).float()

            # 3. parallel 5ths / 8ves with already-emitted voices this tstep
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
                        penalty[b, p_idx] += 1.0

        return logits - self.alpha * penalty


def _to_pitch_tok(tok_id: int, tok: ChoraleTokenizer) -> int | None:
    """Return pitch-token index (0..P-1) or None if not a pitch.

    Kept for backward compatibility with anything else that imports it; the
    ConstrainedProcessor now uses its own HOLD-resolving helper instead.
    """
    if 0 <= tok_id < tok.cfg.n_pitches:
        return tok_id
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    n = args.n or cfg["sample"]["n_samples"]
    alpha = args.alpha if args.alpha is not None else cfg["decode_m3"]["alpha"]
    out_dir = Path(args.out_dir or "samples/m3")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
    ))
    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    blob = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()

    processor = ConstrainedProcessor(tok, alpha=alpha)
    piece_length = cfg["sample"]["piece_length"]
    # Prompt is now 4 real SATB tokens (one chord), not 1 BOS.
    new_tokens = piece_length * 4 - 4

    file_list = out_dir / "file_list.txt"
    with open(file_list, "w") as flist:
        for i in tqdm(range(n), desc="sample"):
            prompt = make_prompt(tok, seed=i).to(device)
            out = model.generate(
                prompt,
                max_new_tokens=new_tokens,
                temperature=cfg["sample"]["temperature"],
                top_p=cfg["sample"]["top_p"],
                logit_processor=processor,
            )
            tokens = out[0]
            fname = f"sample_{i:04d}.mid"
            tokens_to_midi(tokens, tok, out_dir / fname)
            flist.write(f"{i}\t{fname}\n")

    print(f"wrote {n} constrained samples to {out_dir}")


if __name__ == "__main__":
    main()
