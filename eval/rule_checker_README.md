# `rule_checker.py` lives elsewhere

This directory expects the team's existing `rule_checker.py` — the script
that reads a MIDI file and counts voice-leading violations.

Copy or symlink it here:

```bash
cp /path/to/rule_checker.py bach_transformer/eval/rule_checker.py
```

## Required interface

`run_eval.py` and `decode_m4.py` both call:

```python
rule_checker.score_midi(path: str) -> dict
```

The dict must contain (at minimum) the eight per-rule counters and a
`HarmonicScore`:

```python
{
    "parallel_5ths":      int,
    "parallel_8ves":      int,
    "voice_crossings":    int,
    "hidden_5ths_outer":  int,
    "hidden_8ves_outer":  int,
    "spacing_violations": int,
    "large_leaps":        int,
    "augmented_leaps":    int,
    "HarmonicScore":      int,   # sum of the eight, by spec
}
```

If your existing `rule_checker.py` exposes a different entry point (e.g. a
CLI, or a per-rule function set), add a thin wrapper:

```python
def score_midi(path: str) -> dict:
    counts = check_voice_leading(path)        # whatever your existing API is
    counts["HarmonicScore"] = sum(counts[k] for k in [
        "parallel_5ths", "parallel_8ves", "voice_crossings",
        "hidden_5ths_outer", "hidden_8ves_outer",
        "spacing_violations", "large_leaps", "augmented_leaps",
    ])
    return counts
```
