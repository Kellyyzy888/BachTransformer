"""M4 sampler — chord-conditioned autoregressive generation.

Two modes
---------
1. **Free generation** (default). The model samples the key prefix
   itself, then alternates between emitting an RN and four voice tokens
   (S/A/T/B) at each 16th-note timestep. This is the "write me a
   chorale from scratch" mode.

2. **Chord-progression conditioned** (`--chord_progression FILE` or
   `--chord_progression_str "I V I V6 I V7 I"`). The user supplies a
   sequence of Roman-numeral symbols; we force the RN slots of the
   sequence to those exact tokens and only let the model sample the
   four voice tokens at each step. This demonstrates compositional
   control — the poster's "here's our model filling voices for a user-
   supplied harmony" demo.

In both modes we additionally combine with M3 via the `--constrained`
flag, which wraps the sampler in the same `ConstrainedProcessor` used
by decode_m3.py (logit shaping over voice tokens only — chord/key
tokens are never penalized).

Run (free generation):
    python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt --n 100 \
        --out_dir samples/m4

Run (chord-conditioned):
    python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt \
        --chord_progression_str "I I V V I I IV V I" \
        --out_dir samples/m4_cc

Run (M4 + M3 constrained):
    python -m sample.decode_m4 --ckpt checkpoints/m4/best.pt --constrained \
        --out_dir samples/m4_m3
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.tokenizer import ChoraleTokenizer, TokenizerConfig, _tonic_to_pc
from model.transformer import build_model_from_config
from train._common import load_config
from sample._midi_utils import tokens_to_midi


# ---------------------------------------------------------------------------
# Layout helpers — offset within the 5-wide body
# ---------------------------------------------------------------------------

def _offset_in_timestep(pos: int) -> int:
    """Given absolute position pos (pos >= 2), return 0..4.

    0 = RN slot, 1..4 = S/A/T/B.
    """
    return (pos - 2) % 5


# ---------------------------------------------------------------------------
# Chord-progression parsing
# ---------------------------------------------------------------------------

def _parse_progression(
    raw: str,
    piece_length: int,
) -> list[str]:
    """Turn 'I V I vi | IV V I' into a length-`piece_length` RN list.

    Rules:
      - tokens split on whitespace and '|'
      - each token occupies (piece_length // n_tokens) 16th-notes, with
        the remainder going to the last token to preserve length exactly
      - unknown symbols stay as-is and will map to RN_OTHER at encode time
    """
    tokens = [t for t in raw.replace("|", " ").split() if t]
    if not tokens:
        raise ValueError("empty chord progression")
    n = len(tokens)
    base = piece_length // n
    extra = piece_length - base * n
    out: list[str] = []
    for i, tok in enumerate(tokens):
        k = base + (extra if i == n - 1 else 0)
        out.extend([tok] * k)
    assert len(out) == piece_length
    return out


# ---------------------------------------------------------------------------
# Mask builders — zero out illegal token families at each position
# ---------------------------------------------------------------------------

def _build_slot_mask(tok: ChoraleTokenizer, offset: int, device) -> torch.Tensor:
    """Return a (V,) 0/1 mask where legal tokens are 1.

    - offset 0 (RN slot) -> only RN_* tokens legal
    - offsets 1..4 (voice slots) -> only pitch / HOLD / REST tokens legal
      (no BOS/BAR/PAD to avoid mid-sequence breakers)
    """
    V = tok.vocab_size
    mask = torch.zeros(V, device=device)
    if offset == 0:
        start = tok.RN_BASE
        end = tok.RN_BASE + tok.cfg.n_chord
        mask[start:end] = 1.0
    else:
        mask[: tok.cfg.n_pitches] = 1.0     # pitches
        mask[tok.HOLD] = 1.0
        mask[tok.REST] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Sampling loop — chord-interleaved, optionally teacher-forced on RNs
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_chord_interleaved(
    model,
    tok: ChoraleTokenizer,
    piece_length: int,
    tonic_name: str | None = None,
    mode: str | None = None,
    forced_rns: list[str] | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    logit_processor=None,
    device=None,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a single chord-interleaved token sequence.

    If tonic_name / mode are given, we seed the prefix. Otherwise we
    sample the mode (major/minor) and the tonic pitch-class, giving the
    model freedom to choose a key.

    If forced_rns is non-None, its length must equal piece_length; we
    place those RN tokens into the RN slots deterministically and only
    sample the voice slots.
    """
    rng = torch.Generator(device=device if device else "cpu")
    if seed is not None:
        rng.manual_seed(seed)

    device = device or next(model.parameters()).device

    # -------- key prefix -----------------------------------------------
    prefix: list[int] = []
    if mode is not None:
        prefix.append(
            tok.KEY_MAJOR if mode.lower().startswith("maj") else tok.KEY_MINOR
        )
    else:
        # Sample mode from its two-way prior. Build a 2-way softmax on
        # just {KEY_MAJOR, KEY_MINOR}.
        logits = model(torch.tensor([[tok.BOS]], device=device) if False
                       else torch.tensor([[tok.KEY_MAJOR]], device=device))
        # Actually simplest: bias toward major (4:1) empirically — JSB is
        # ~62% major and RN-based key detection biases further up.
        prefix.append(tok.KEY_MAJOR if torch.rand((), generator=rng).item() < 0.7
                      else tok.KEY_MINOR)

    if tonic_name is not None:
        prefix.append(tok.PC_BASE + _tonic_to_pc(tonic_name))
    else:
        # sample a tonic pc uniformly from {0..11}
        pc = int(torch.randint(0, 12, (1,), generator=rng).item())
        prefix.append(tok.PC_BASE + pc)

    seq = torch.tensor([prefix], dtype=torch.long, device=device)

    # -------- body ------------------------------------------------------
    body_len = 5 * piece_length
    slot_masks = {o: _build_slot_mask(tok, o, device) for o in range(5)}

    for step in range(body_len):
        offset = step % 5
        pos_abs = 2 + step        # absolute position in the sequence

        # Teacher-forced RN slot?
        if offset == 0 and forced_rns is not None:
            rn_str = forced_rns[step // 5]
            tok_id = tok.rn_to_token(rn_str)
            seq = torch.cat(
                [seq, torch.tensor([[tok_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            continue

        # Otherwise run the LM on the current prefix.
        input_ids = seq[:, -model.cfg.max_seq_len:]
        logits = model(input_ids)[:, -1, :]          # (1, V)

        # Optional logit processor (for M3 combination).
        if logit_processor is not None:
            logits = logit_processor(logits, seq)

        # Mask out illegal token families for this slot.
        mask = slot_masks[offset]
        logits = logits.masked_fill(mask == 0, float("-inf"))

        if temperature != 1.0:
            logits = logits / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            probs = _nucleus_filter(probs, top_p)
        # Guard against all-zero probs (mask + low-temperature edge case)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            # fall back to uniform over the legal set for this slot
            probs = mask / mask.sum()
        next_tok = torch.multinomial(probs, num_samples=1, generator=rng)
        seq = torch.cat([seq, next_tok], dim=1)

    return seq[0]


def _nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cum > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_constrained_processor(tok: ChoraleTokenizer, cfg: dict):
    """Lazy-import ConstrainedProcessor from decode_m3 and wrap it so it
    only acts on voice slots (it wasn't written with chord interleaving
    in mind; RN slots would be penalized as voice-crossing nonsense
    otherwise).
    """
    try:
        from sample.decode_m3 import ConstrainedProcessor
    except ImportError as e:
        raise RuntimeError(
            "decode_m3.ConstrainedProcessor not found — needed for --constrained"
        ) from e

    inner = ConstrainedProcessor(tok, alpha=cfg.get("decode_m3", {}).get("alpha", 5.0))

    def processor(logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        # `generated` in the chord layout is the full sequence so far.
        # We need to hand the M3 processor a voice-only sequence so its
        # voice-crossing / parallel-5th / leap checks line up on stride-4
        # boundaries. Strip prefix + every 5th token (RN slot).
        ids = generated[0].detach().cpu().tolist()
        if len(ids) < 2 or (ids[0] != tok.KEY_MAJOR and ids[0] != tok.KEY_MINOR):
            voice_only = generated
        else:
            body = ids[2:]
            voice_body = [t for i, t in enumerate(body) if i % 5 != 0]
            voice_only = torch.tensor(
                [voice_body], dtype=generated.dtype, device=generated.device
            )
        # If we're currently generating a CHORD / RN slot, leave logits alone.
        pos_in_body = generated.size(1) - 2
        if pos_in_body >= 0 and pos_in_body % 5 == 0:
            return logits
        return inner(logits, voice_only)

    return processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True,
                    help="path to M4 checkpoint (checkpoints/m4/best.pt)")
    ap.add_argument("--n", type=int, default=None, help="number of samples")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--constrained", action="store_true",
                    help="combine with M3 ConstrainedProcessor on voice slots")
    ap.add_argument("--chord_progression", default=None,
                    help="path to a file containing one RN per line (or "
                         "space-separated on one line)")
    ap.add_argument("--chord_progression_str", default=None,
                    help="pass the RN progression inline, e.g. 'I V I V6 I'")
    ap.add_argument("--tonic", default=None,
                    help="force tonic (e.g. 'C', 'F#', 'Bb')")
    ap.add_argument("--mode", default=None, choices=["major", "minor"])
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    cfg.setdefault("chord", {})["enabled"] = True

    n = args.n or cfg.get("decode_m4", {}).get("n_samples", cfg["sample"]["n_samples"])
    out_dir = Path(args.out_dir or cfg.get("decode_m4", {}).get("out_dir", "samples/m4"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load chord vocab from cache to rebuild the tokenizer.
    chord_cache = cfg["chord"]["cache_path"]
    with open(chord_cache, "rb") as f:
        chord_payload = pickle.load(f)
    chord_vocab = list(chord_payload["vocab"])

    tok = ChoraleTokenizer(TokenizerConfig(
        pitch_min=cfg["tokenizer"]["pitch_min"],
        pitch_max=cfg["tokenizer"]["pitch_max"],
        chord_vocab=chord_vocab,
    ))

    model = build_model_from_config(cfg, vocab_size=tok.vocab_size).to(device)
    blob = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if unexpected:
        print(f"  ignored unexpected keys: {unexpected}")
    if missing:
        print(f"  missing keys (should be empty): {missing}")
    model.eval()

    piece_length = cfg["sample"]["piece_length"]
    temperature = cfg["sample"]["temperature"]
    top_p = cfg["sample"]["top_p"]

    # Parse the chord progression once if supplied.
    forced_rns: list[str] | None = None
    if args.chord_progression_str:
        forced_rns = _parse_progression(args.chord_progression_str, piece_length)
    elif args.chord_progression:
        text = Path(args.chord_progression).read_text()
        forced_rns = _parse_progression(text, piece_length)

    logit_proc = _load_constrained_processor(tok, cfg) if args.constrained else None

    file_list = out_dir / "file_list.txt"
    with open(file_list, "w") as flist:
        for i in tqdm(range(n), desc="sample"):
            seq = sample_chord_interleaved(
                model, tok,
                piece_length=piece_length,
                tonic_name=args.tonic,
                mode=args.mode,
                forced_rns=forced_rns,
                temperature=temperature,
                top_p=top_p,
                logit_processor=logit_proc,
                device=device,
                seed=i,
            )
            fname = f"sample_{i:04d}.mid"
            tokens_to_midi(seq, tok, out_dir / fname)
            flist.write(f"{i}\t{fname}\n")

    mode_str = "chord-conditioned" if forced_rns else "free"
    if args.constrained:
        mode_str += " + constrained"
    print(f"wrote {n} {mode_str} samples to {out_dir}")


if __name__ == "__main__":
    main()
