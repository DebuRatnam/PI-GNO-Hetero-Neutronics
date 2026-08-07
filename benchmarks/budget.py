"""Parameter-budget matching and the shared learning-rate grid.

Two objections kill an architecture comparison, and both are answered here
rather than in prose:

  "your model is just bigger"   -> every model is sized to the same parameter
                                   budget by bisecting ONE width knob, and the
                                   achieved count is recorded in summary.json.
  "you undertuned the baselines" -> every model gets the SAME learning-rate grid,
                                   selected on val, and the winner is logged.

Width is the knob rather than depth because depth changes the receptive field,
which is part of what is under study for the graph models. Bisecting a width
leaves the architecture's character intact.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import _paths  # noqa: F401
from interface import build_model

# Same grid for every model. Deliberately small: the point is equal treatment,
# not an exhaustive search that would favour whoever got more compute.
LR_GRID = (3e-4, 1e-3, 3e-3)

# The single width knob bisected per model to hit the budget.
SCALE_KNOB: Dict[str, str] = {
    "pigno": "latent_dim",
    "deeponet": "branch_hidden",
    "mgn": "hidden_dim",
    "fno": "latent_channels",
}

# Widths that must stay a multiple of something (attention heads, FFT channels).
KNOB_STEP: Dict[str, int] = {"pigno": 8, "deeponet": 8, "mgn": 8, "fno": 4}


def count_params(model_name: str, meta: dict, hparams: dict) -> int:
    return build_model(model_name, meta, **hparams).n_params()


def match_budget(model_name: str, meta: dict, target: int,
                 base_hparams: Optional[dict] = None, knob: Optional[str] = None,
                 lo: int = 8, hi: int = 1024, tol: float = 0.20,
                 verbose: bool = True) -> Tuple[dict, int]:
    """Bisect one width knob so `model_name` lands within `tol` of `target`.

    Returns (hparams, achieved_params). Raises if the budget is unreachable in
    [lo, hi] -- silently returning a model 3x the budget would invalidate the
    comparison it exists to protect.
    """
    hp = dict(base_hparams or {})
    knob = knob or SCALE_KNOB.get(model_name)
    if knob is None:
        raise KeyError(f"no width knob registered for '{model_name}'; "
                       f"add one to budget.SCALE_KNOB")
    step = KNOB_STEP.get(model_name, 1)

    def n_at(w: int) -> int:
        return count_params(model_name, meta, {**hp, knob: int(w)})

    lo_w = max(step, (lo // step) * step)
    hi_w = (hi // step) * step
    n_lo, n_hi = n_at(lo_w), n_at(hi_w)
    if not (n_lo <= target <= n_hi):
        raise ValueError(
            f"{model_name}: target {target:,} params is outside the reachable "
            f"range [{n_lo:,}, {n_hi:,}] for {knob} in [{lo_w}, {hi_w}]. "
            f"Widen the search or pick a different knob.")

    while hi_w - lo_w > step:
        mid = ((lo_w + hi_w) // 2 // step) * step
        mid = max(mid, lo_w + step)
        if n_at(mid) <= target:
            lo_w = mid
        else:
            hi_w = mid

    best_w = min((lo_w, hi_w), key=lambda w: abs(n_at(w) - target))
    hp[knob] = int(best_w)
    got = n_at(best_w)
    rel = abs(got - target) / target
    if verbose:
        print(f"budget {model_name}: {knob}={best_w} -> {got:,} params "
              f"(target {target:,}, off by {rel:.1%})")
    if rel > tol:
        raise ValueError(
            f"{model_name}: closest reachable is {got:,} params vs target "
            f"{target:,} ({rel:.1%} off, tolerance {tol:.0%}). The knob is too "
            f"coarse near the budget -- adjust KNOB_STEP or the base hparams.")
    return hp, got


if __name__ == "__main__":
    import argparse
    import json
    from data import GraphDataset
    from sensors import domain_from_dataset

    ap = argparse.ArgumentParser(
        description="size every model to a common parameter budget")
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=["pigno", "deeponet"])
    ap.add_argument("--probe", type=int, default=8)
    a = ap.parse_args()

    ds = GraphDataset(a.data, "train", limit=a.probe)
    meta = ds.metadata()
    dom = tuple(domain_from_dataset(ds, a.probe))
    out = {}
    for m in a.models:
        hp, got = match_budget(m, meta, a.target, {"domain": dom})
        hp.pop("domain", None)
        out[m] = {"hparams": hp, "n_params": got}
    print(json.dumps(out, indent=2))
