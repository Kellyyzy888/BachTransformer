"""Run rule_checker.py over a folder of generated MIDI samples and emit a
CSV + summary of per-rule deltas between two conditions.

Expected folder layout:
    samples/m1/sample_*.mid       # baseline
    samples/m2/sample_*.mid       # rule-aware loss
    samples/m3/sample_*.mid       # constrained decoding
    samples/m4/sample_*.mid       # rerank

Run:
    # Single condition
    python -m eval.run_eval --samples_dir samples/m1 --out eval/m1.csv

    # Compare two conditions
    python -m eval.run_eval --samples_dir samples/m1 --compare_to samples/m2 \
        --out eval/m1_vs_m2.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


RULE_COUNTERS = [
    "parallel_5ths",
    "parallel_8ves",
    "voice_crossings",
    "hidden_5ths_outer",
    "hidden_8ves_outer",
    "spacing_violations",
    "large_leaps",
    "augmented_leaps",
]


def _load_checker(path: str):
    spec = importlib.util.spec_from_file_location("rule_checker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rule_checker"] = mod
    spec.loader.exec_module(mod)
    return mod


def _score_folder(folder: Path, checker) -> pd.DataFrame:
    rows = []
    for midi_path in sorted(folder.glob("*.mid")):
        metrics = checker.score_midi(str(midi_path))
        row = {"file": midi_path.name}
        row.update({k: metrics.get(k, 0) for k in RULE_COUNTERS})
        row["HarmonicScore"] = metrics.get(
            "HarmonicScore", sum(row[k] for k in RULE_COUNTERS)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--compare_to", default=None,
                    help="second folder; reports per-rule delta (samples_dir - compare_to)")
    ap.add_argument("--checker", default="eval/rule_checker.py")
    ap.add_argument("--out", default="eval/out.csv")
    args = ap.parse_args()

    checker = _load_checker(args.checker)
    df_a = _score_folder(Path(args.samples_dir), checker)
    print(f"\n== {args.samples_dir} ==")
    print(df_a[RULE_COUNTERS + ["HarmonicScore"]].agg(["mean", "std", "sum"]))
    df_a.to_csv(args.out, index=False)

    if args.compare_to:
        df_b = _score_folder(Path(args.compare_to), checker)
        print(f"\n== {args.compare_to} ==")
        print(df_b[RULE_COUNTERS + ["HarmonicScore"]].agg(["mean", "std", "sum"]))

        deltas = {
            r: df_a[r].sum() - df_b[r].sum()
            for r in RULE_COUNTERS
        }
        deltas["HarmonicScore"] = df_a["HarmonicScore"].sum() - df_b["HarmonicScore"].sum()
        avg_delta = sum(deltas[r] for r in RULE_COUNTERS) / len(RULE_COUNTERS)

        print("\n== Deltas (A - B, positive = A has more violations) ==")
        for r, v in deltas.items():
            print(f"  {r:24s}  {v:+d}" if isinstance(v, int) else f"  {r:24s}  {v:+.2f}")
        print(f"  avg_delta (8 rules)        {avg_delta:+.2f}")


if __name__ == "__main__":
    main()
