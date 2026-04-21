"""A/B listening-study harness for the poster's qualitative row.

What this produces
------------------
Given two sample directories A and B (e.g. M1 vs M4, or M1+M3 vs M4+M3),
we:
  1. Match A[i] <-> B[i] by filename order (both dirs are produced by
     the decode_*.py scripts and saved as `sample_{i:04d}.mid`).
  2. Pick `n_pairs` pairs using `rule_checker.harmonic_score` to bias
     toward samples where either arm produces a *musically plausible*
     result (HS < HS_MAX). Pure garbage pairs would wash out the
     signal — listeners just score both sides low.
  3. Randomize within-pair presentation order (A-first vs B-first) and
     across-pair ordering, so no listener sees a consistent first/second
     pattern.
  4. Render every selected MIDI to a WAV (optional; requires fluidsynth
     on PATH) and copy both .mid and .wav into `out_dir/stimuli/`.
  5. Emit `study_manifest.csv` with the ground truth (which option is A,
     which is B, which pair, randomized letter labels) — DO NOT SHOW
     THIS to listeners; it's for analysis only.
  6. Emit `listener_form.csv` — the unblinded form template with
     randomized A/B mapped to per-pair "Option 1" / "Option 2", ready
     to paste into a Google Form / Qualtrics.
  7. Optionally include ground-truth Bach samples from the JSB test set
     as calibration stimuli (--include_bach).

Analysis helper
---------------
After collecting responses as a CSV in `responses.csv` with one row per
(listener, pair) containing columns
    listener_id, pair_id, preferred ("A"|"B"|"tie"),
    voice_leading_errors ("yes"|"no")
run:
    python -m eval.ab_study analyze --responses responses.csv \
        --manifest out_dir/study_manifest.csv
to get the Wilcoxon signed-rank test + effect sizes. With n=10 listeners
and 6 pairs this gives ~60 paired comparisons — enough for a directional
poster claim, not enough for a paper.

Run (prepare)
-------------
    python -m eval.ab_study prepare \
        --arm_a samples/m1 --name_a "M1 baseline" \
        --arm_b samples/m4 --name_b "M4 chord-conditioned" \
        --n_pairs 6 --out_dir eval/ab_study_out
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path

# rule_checker.score_midi returns a dict with at least 'HarmonicScore'.
# We tolerate the case where it's missing / not importable and fall back
# to scoring everything as 0 (meaning "include all pairs") rather than
# hard-failing the pipeline.
try:
    from eval.rule_checker import score_midi as _score_midi
except Exception:
    _score_midi = None


HS_MAX_DEFAULT = 8.0     # pairs whose max(HS_a, HS_b) > this are excluded


# ---------------------------------------------------------------------------
# Stimulus selection
# ---------------------------------------------------------------------------

def _list_mids(dir_: Path) -> list[Path]:
    return sorted(p for p in dir_.iterdir() if p.suffix.lower() == ".mid")


def _safe_hs(path: Path) -> float:
    if _score_midi is None:
        return 0.0
    try:
        m = _score_midi(str(path))
        return float(m.get("HarmonicScore", 0.0))
    except Exception:
        return float("inf")          # exclude on error


def select_pairs(
    arm_a_dir: Path,
    arm_b_dir: Path,
    n_pairs: int,
    hs_max: float,
    rng: random.Random,
) -> list[tuple[int, Path, Path, float, float]]:
    """Return a list of (idx, path_a, path_b, hs_a, hs_b) tuples.

    We pair by index order, filter by max(hs_a, hs_b) <= hs_max, then
    sample `n_pairs` uniformly from the survivors.
    """
    a_files = _list_mids(arm_a_dir)
    b_files = _list_mids(arm_b_dir)
    n = min(len(a_files), len(b_files))
    candidates: list[tuple[int, Path, Path, float, float]] = []
    for i in range(n):
        hs_a = _safe_hs(a_files[i])
        hs_b = _safe_hs(b_files[i])
        if max(hs_a, hs_b) <= hs_max:
            candidates.append((i, a_files[i], b_files[i], hs_a, hs_b))
    if len(candidates) < n_pairs:
        print(f"[warn] only {len(candidates)} survivors below hs_max={hs_max}; "
              f"using all of them.")
        return candidates
    return rng.sample(candidates, n_pairs)


# ---------------------------------------------------------------------------
# Rendering MIDI -> WAV (via fluidsynth if available)
# ---------------------------------------------------------------------------

def _render_wav(midi_path: Path, wav_path: Path, soundfont: str | None) -> bool:
    if soundfont is None:
        return False
    try:
        subprocess.run(
            ["fluidsynth", "-ni", soundfont, str(midi_path),
             "-F", str(wav_path), "-r", "44100"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# `prepare` command
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    stimuli = out / "stimuli"
    stimuli.mkdir(parents=True, exist_ok=True)

    pairs = select_pairs(
        Path(args.arm_a), Path(args.arm_b),
        n_pairs=args.n_pairs, hs_max=args.hs_max, rng=rng,
    )
    if not pairs:
        raise SystemExit("no pair survived selection — lower --hs_max or "
                         "sample more generations first")

    # Shuffle pair order; within each pair, randomize which arm is shown
    # first ("Option 1" vs "Option 2"). Keep the ground truth in manifest.
    rng.shuffle(pairs)

    manifest_rows = []
    listener_rows = []
    for display_idx, (orig_idx, path_a, path_b, hs_a, hs_b) in enumerate(pairs, 1):
        a_first = rng.random() < 0.5
        if a_first:
            opt1_src, opt1_arm, opt1_hs = path_a, args.name_a, hs_a
            opt2_src, opt2_arm, opt2_hs = path_b, args.name_b, hs_b
        else:
            opt1_src, opt1_arm, opt1_hs = path_b, args.name_b, hs_b
            opt2_src, opt2_arm, opt2_hs = path_a, args.name_a, hs_a

        pair_id = f"pair_{display_idx:02d}"
        dst1_mid = stimuli / f"{pair_id}_option1.mid"
        dst2_mid = stimuli / f"{pair_id}_option2.mid"
        shutil.copy(opt1_src, dst1_mid)
        shutil.copy(opt2_src, dst2_mid)

        wav1 = stimuli / f"{pair_id}_option1.wav"
        wav2 = stimuli / f"{pair_id}_option2.wav"
        r1 = _render_wav(dst1_mid, wav1, args.soundfont)
        r2 = _render_wav(dst2_mid, wav2, args.soundfont)

        manifest_rows.append({
            "pair_id": pair_id,
            "orig_index": orig_idx,
            "option1_arm": opt1_arm, "option1_hs": f"{opt1_hs:.2f}",
            "option2_arm": opt2_arm, "option2_hs": f"{opt2_hs:.2f}",
            "option1_src": str(opt1_src), "option2_src": str(opt2_src),
            "rendered_wav": "yes" if (r1 and r2) else "no",
        })
        listener_rows.append({
            "pair_id": pair_id,
            "option1_file": dst1_mid.name,
            "option2_file": dst2_mid.name,
            "question_preference": "Which sounds more Bach-like?",
            "scale": "1=Option1 much more, 2=Option1 somewhat more, "
                     "3=tie, 4=Option2 somewhat more, 5=Option2 much more",
            "question_errors": "Any obvious voice-leading errors? (yes/no)",
        })

    # Ground-truth Bach calibration stimuli (optional).
    if args.include_bach and args.bach_dir:
        bach_files = _list_mids(Path(args.bach_dir))[: args.n_bach]
        for j, bf in enumerate(bach_files, 1):
            dst = stimuli / f"calibration_{j:02d}.mid"
            shutil.copy(bf, dst)
            _render_wav(dst, dst.with_suffix(".wav"), args.soundfont)
            manifest_rows.append({
                "pair_id": f"calibration_{j:02d}",
                "orig_index": -1,
                "option1_arm": "ground_truth_bach", "option1_hs": "0.00",
                "option2_arm": "", "option2_hs": "",
                "option1_src": str(bf), "option2_src": "",
                "rendered_wav": "yes",
            })

    # Write manifest (ground truth; keep private).
    with open(out / "study_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # Write listener form template (blinded — no arm labels).
    with open(out / "listener_form.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(listener_rows[0].keys()))
        w.writeheader()
        w.writerows(listener_rows)

    # Instructions page for listeners.
    (out / "INSTRUCTIONS.md").write_text(_INSTRUCTIONS_TEMPLATE.format(
        n_pairs=args.n_pairs, name_a=args.name_a, name_b=args.name_b,
    ))

    print(f"Wrote {len(pairs)} pairs to {stimuli}/")
    print(f"Manifest (ground truth, DO NOT SHOW LISTENERS): "
          f"{out / 'study_manifest.csv'}")
    print(f"Listener form template: {out / 'listener_form.csv'}")


_INSTRUCTIONS_TEMPLATE = """# Bach Chorale A/B Listening Study

You'll hear {n_pairs} pairs of short (~16-second) four-part chorale
excerpts. Both clips in each pair are generated by deep-learning
models trained on Bach's chorale corpus.

**For each pair, please:**

1. Listen to **Option 1** and **Option 2**.
2. Rate which one sounds more "Bach-like" on a 5-point scale:
   1 = Option 1 much more Bach-like
   2 = Option 1 somewhat more
   3 = Tie / can't tell
   4 = Option 2 somewhat more
   5 = Option 2 much more
3. Note whether you heard any obvious voice-leading errors (yes/no).
   (Voice-leading errors = things like parallel fifths, voice
   crossings, or awkward leaps that sound "off" for a chorale.)

You may listen to each clip as many times as you need. There is no
time pressure. Background: {name_a} vs {name_b}. Labels are hidden.

Do you have any formal music training? (yes/no, and if yes, what level)
"""


# ---------------------------------------------------------------------------
# `analyze` command
# ---------------------------------------------------------------------------

def cmd_analyze(args):
    # Lazy import scipy so `prepare` doesn't require it.
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not installed; analyze requires `pip install scipy`")
        return

    # Load manifest (ground-truth arm for each pair).
    manifest: dict[str, tuple[str, str]] = {}   # pair_id -> (arm1, arm2)
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            if row["pair_id"].startswith("pair_"):
                manifest[row["pair_id"]] = (row["option1_arm"], row["option2_arm"])

    # Load responses. Each row: listener_id, pair_id, rating (1-5), errors (y/n)
    rows = []
    with open(args.responses) as f:
        for row in csv.DictReader(f):
            if row["pair_id"] not in manifest:
                continue
            try:
                rating = float(row["rating"])
            except ValueError:
                continue
            rows.append(row | {"rating": rating})

    if not rows:
        print("no valid response rows; check column names "
              "(listener_id, pair_id, rating, errors)")
        return

    # Convert rating into "preference for arm B over arm A". The preference
    # direction depends on whether Option1/Option2 ↔ A/B was swapped.
    # Rating scale: 1 = Option1 much more; 5 = Option2 much more.
    # Center at 3 so the sign is intuitive.
    diffs_by_pair: dict[str, list[float]] = {}
    for row in rows:
        arm1, arm2 = manifest[row["pair_id"]]
        signed = row["rating"] - 3.0
        # We want positive = prefer arm_B. If arm_B is on option2, signed is
        # already correct; else flip.
        if arm2 == args.arm_b_name:
            diffs_by_pair.setdefault(row["pair_id"], []).append(signed)
        elif arm1 == args.arm_b_name:
            diffs_by_pair.setdefault(row["pair_id"], []).append(-signed)

    # Per-pair mean, then Wilcoxon across listeners (treat each
    # listener's per-pair averaged preference as a signed sample).
    per_listener: dict[str, list[float]] = {}
    for row in rows:
        arm1, arm2 = manifest[row["pair_id"]]
        signed = row["rating"] - 3.0
        if arm1 == args.arm_b_name:
            signed = -signed
        per_listener.setdefault(row["listener_id"], []).append(signed)

    listener_means = [sum(v) / len(v) for v in per_listener.values()]
    n = len(listener_means)

    print(f"\n=== A/B study — {args.arm_a_name} (A) vs {args.arm_b_name} (B) ===")
    print(f"listeners: {n}, total responses: {len(rows)}")
    print(f"mean preference (positive = prefer B): "
          f"{sum(listener_means)/n:+.3f}  (1pt scale)")

    if n >= 6 and any(abs(x) > 1e-9 for x in listener_means):
        stat, p = wilcoxon(listener_means, alternative="two-sided")
        print(f"Wilcoxon signed-rank (matched-pair, listener-level): "
              f"W={stat:.2f}  p={p:.4f}")
    else:
        print("Too few listeners (or all ties) for Wilcoxon — reporting "
              "descriptive stats only.")

    # Error-rate breakdown.
    err_by_arm: dict[str, list[int]] = {args.arm_a_name: [], args.arm_b_name: []}
    for row in rows:
        arm1, arm2 = manifest[row["pair_id"]]
        flag = 1 if str(row.get("errors", "")).lower().startswith("y") else 0
        # "errors" in the form is per-pair (did either side have errors?) —
        # so attribute to both. If you run a two-question form (errors per
        # option) adjust this.
        err_by_arm.setdefault(arm1, []).append(flag)
        err_by_arm.setdefault(arm2, []).append(flag)
    for arm, flags in err_by_arm.items():
        if flags:
            print(f"  voice-leading-error rate in pairs containing {arm}: "
                  f"{sum(flags)/len(flags):.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--arm_a", required=True)
    p_prep.add_argument("--arm_b", required=True)
    p_prep.add_argument("--name_a", default="arm_a")
    p_prep.add_argument("--name_b", default="arm_b")
    p_prep.add_argument("--n_pairs", type=int, default=6)
    p_prep.add_argument("--hs_max", type=float, default=HS_MAX_DEFAULT)
    p_prep.add_argument("--out_dir", default="eval/ab_study_out")
    p_prep.add_argument("--seed", type=int, default=1470)
    p_prep.add_argument("--soundfont", default=None,
                        help="path to a .sf2 file for WAV rendering; "
                             "requires `fluidsynth` on PATH")
    p_prep.add_argument("--include_bach", action="store_true")
    p_prep.add_argument("--bach_dir", default=None,
                        help="dir containing ground-truth Bach MIDIs "
                             "(only used with --include_bach)")
    p_prep.add_argument("--n_bach", type=int, default=2)
    p_prep.set_defaults(func=cmd_prepare)

    p_ana = sub.add_parser("analyze")
    p_ana.add_argument("--responses", required=True)
    p_ana.add_argument("--manifest", required=True)
    p_ana.add_argument("--arm_a_name", default="arm_a")
    p_ana.add_argument("--arm_b_name", default="arm_b")
    p_ana.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
