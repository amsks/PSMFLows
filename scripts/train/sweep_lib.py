"""scripts/train/sweep_lib.py — pure helpers for grid-sweep selection (unit-tested)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_SUCC = re.compile(r"eval/success[=:\s]+([0-9.]+)")


def parse_final_success(log_path: str) -> Optional[float]:
    """Last eval/success value in a run log, or None if absent."""
    last = None
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = _SUCC.search(line)
                if m:
                    last = float(m.group(1))
    except FileNotFoundError:
        return None
    return last


def aggregate_grid(rows: Dict[Tuple[str, str], List[float]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """(knob_a, knob_b) -> {mean, std, n} over per-seed final successes (None dropped)."""
    out = {}
    for key, vals in rows.items():
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[key] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    return out


def pick_winner(agg: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[str, str]:
    """Grid point with the highest mean success (NaN treated as -inf)."""
    def score(kv):
        m = kv[1]["mean"]
        return -np.inf if np.isnan(m) else m
    return max(agg.items(), key=score)[0]
