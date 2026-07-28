"""Reader and interpolator for the OpenMC branch table (xs_openmc.py, schema v2).

The generator needs cross sections at arbitrary (burnup, temperature, control
insertion) per node, but transport can only afford a coarse grid of branch cases.
This module does the standard lattice-physics thing: tabulate the branches, then
interpolate between them.

Interpolation rules (all documented in the sample metadata):

  burnup / temperature : linear between bracketing branch points, CLAMPED at the
      ends (never extrapolated). Macroscopic cross sections are close to linear in
      both over the tabulated spacing.
  control insertion    : linear between the rod-out and rod-in branches. For a
      control/shutdown LABEL the two endpoints are physically different materials
      (follower vs B4C absorber), so this is the gray-rod blend already used by the
      generator -- except that both endpoints are now MEASURED. For every other
      material the same axis carries the spectrum shift a nearby inserted absorber
      causes, evaluated at the CORE-AVERAGE insertion fraction.

Every blend goes through xs_common.blend_xs, so Sr / scatter / nuSf combine linearly
while D combines through its transport cross section (harmonic in D) -- averaging D
directly is not physical.

Absent table -> the caller keeps its committed hand-tuned library, so the pipeline
still runs without OpenMC.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from xs_common import (MultiGroupXS, blend_xs, multigroupxs_from_dict, n_scatter)


# FEM material labels whose rod-out endpoint is the follower material rather than
# their own record. Covers both reactors' control families.
CONTROL_LABELS = frozenset({
    "primary_control", "secondary_control",     # hex / Natrium
    "control_element", "shutdown_element",      # fhr / KP-FHR
})
FOLLOWER_KEY = "control_follower"

# lookups are quantized before caching; finer than any physical variation we resolve
_BURNUP_Q = 0.25       # MWd/kgHM
_TEMP_Q = 1.0          # K
_FRAC_Q = 1.0e-3


def _lerp(a: float, b: float, w: float) -> float:
    return (1.0 - w) * a + w * b


class BranchTable:
    """Tabulated OpenMC group constants over (burnup, temperature, rod)."""

    def __init__(self, blob: dict):
        if int(blob.get("schema_version", 1)) < 2:
            raise ValueError("branch table requires schema_version >= 2; regenerate "
                             "with xs_openmc.py")
        self.blob = blob
        self.reactor_type: str = blob["reactor_type"]
        self.n_groups: int = int(blob["n_groups"])
        self.group_boundaries_ev: List[float] = list(blob["group_boundaries_ev"])
        self.source: str = blob.get("source", "unknown")
        self.provenance: dict = blob.get("provenance", {})

        self.burnups: List[float] = sorted({float(b["burnup_mwd_kg"])
                                            for b in blob["branches"]})
        self.temperatures: List[float] = sorted({float(b["temperature_k"])
                                                 for b in blob["branches"]})
        self.rods: List[str] = sorted({str(b["rod"]) for b in blob["branches"]})

        # (burnup, temperature, rod) -> branch
        self._by_key: Dict[Tuple[float, float, str], dict] = {}
        self._xs: Dict[Tuple[float, float, str, str], MultiGroupXS] = {}
        self._up: Dict[Tuple[float, float, str, str], np.ndarray] = {}
        for b in blob["branches"]:
            key = (float(b["burnup_mwd_kg"]), float(b["temperature_k"]), str(b["rod"]))
            self._by_key[key] = b
            for name, rec in b["materials"].items():
                self._xs[key + (name,)] = multigroupxs_from_dict(rec)
                self._up[key + (name,)] = np.asarray(
                    rec.get("upscatter", [0.0] * n_scatter(self.n_groups)), float)

        self.materials = sorted({k[3] for k in self._xs})
        self._cache: Dict[tuple, Tuple[MultiGroupXS, np.ndarray]] = {}

    # --- axis helpers ---------------------------------------------------------

    @staticmethod
    def _bracket(axis: List[float], v: float) -> Tuple[float, float, float]:
        """(lo, hi, weight) bracketing `v` on a sorted axis; clamped at the ends."""
        if len(axis) == 1:
            return axis[0], axis[0], 0.0
        v = float(min(max(v, axis[0]), axis[-1]))
        j = int(np.searchsorted(axis, v, side="right") - 1)
        j = max(0, min(j, len(axis) - 2))
        lo, hi = axis[j], axis[j + 1]
        w = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return lo, hi, w

    def _material_key(self, material: str, rod: str) -> Optional[str]:
        """Which stored record represents `material` on a given rod branch."""
        if rod == "out" and material in CONTROL_LABELS:
            return FOLLOWER_KEY
        return material

    def _at(self, bu: float, T: float, rod: str, material: str
            ) -> Optional[Tuple[MultiGroupXS, np.ndarray]]:
        key = self._material_key(material, rod)
        k = (bu, T, rod, key)
        if k not in self._xs:
            return None
        return self._xs[k], self._up[k]

    def _at_rod(self, rod: str, bu: float, T: float, material: str
                ) -> Optional[Tuple[MultiGroupXS, np.ndarray]]:
        """Bilinear over (burnup, temperature) on one rod branch."""
        b0, b1, wb = self._bracket(self.burnups, bu)
        t0, t1, wt = self._bracket(self.temperatures, T)
        corners = []
        for b, w_b in ((b0, 1.0 - wb), (b1, wb)):
            for t, w_t in ((t0, 1.0 - wt), (t1, wt)):
                got = self._at(b, t, rod, material)
                if got is None:
                    return None
                corners.append((got, w_b * w_t))
        # collapse pairwise so every combination still goes through blend_xs
        xs, up = corners[0][0]
        acc_w = corners[0][1]
        for (nxt_xs, nxt_up), w in corners[1:]:
            if w <= 0.0:
                continue
            tot = acc_w + w
            f = w / tot if tot > 0 else 0.0
            xs = blend_xs(xs, nxt_xs, f)
            up = (1.0 - f) * up + f * nxt_up
            acc_w = tot
        return xs, up

    # --- public lookup --------------------------------------------------------

    def has(self, material: str) -> bool:
        if material in CONTROL_LABELS:
            return material in self.materials or FOLLOWER_KEY in self.materials
        return material in self.materials

    def lookup(self, material: str, *, burnup_mwd_kg: float = 0.0,
               temperature_k: Optional[float] = None,
               insert_frac: float = 0.0,
               core_rod_frac: float = 0.0
               ) -> Optional[Tuple[MultiGroupXS, np.ndarray]]:
        """(MultiGroupXS, up-scatter block) for one material at one state.

        insert_frac   : THIS element's control insertion depth in [0,1]. Only used
                        for control/shutdown labels (0 = follower, 1 = absorber).
        core_rod_frac : the CORE-AVERAGE insertion fraction, used for every other
                        material -- it is the spectrum shift caused by absorbers
                        elsewhere in the core, which is exactly what a branch case
                        measures. Ignored when the table has a single rod state.

        Returns None if the material is not in the table (caller falls back).
        """
        T = float(self.temperatures[0] if temperature_k is None else temperature_k)
        w = float(insert_frac if material in CONTROL_LABELS else core_rod_frac)
        w = min(max(w, 0.0), 1.0)

        ck = (material, round(burnup_mwd_kg / _BURNUP_Q), round(T / _TEMP_Q),
              round(w / _FRAC_Q))
        hit = self._cache.get(ck)
        if hit is not None:
            return hit

        if "out" in self.rods and "in" in self.rods:
            lo = self._at_rod("out", burnup_mwd_kg, T, material)
            hi = self._at_rod("in", burnup_mwd_kg, T, material)
            if lo is None and hi is None:
                return None
            if lo is None:
                out = hi
            elif hi is None:
                out = lo
            else:
                out = (blend_xs(lo[0], hi[0], w), (1.0 - w) * lo[1] + w * hi[1])
        else:
            rod = self.rods[0]
            out = self._at_rod(rod, burnup_mwd_kg, T, material)
            if out is None:
                return None

        self._cache[ck] = out
        return out

    def chi(self, *, burnup_mwd_kg: float = 0.0,
            temperature_k: Optional[float] = None,
            core_rod_frac: float = 0.0) -> Optional[Tuple[float, ...]]:
        """Core-average fission spectrum at a state (goes into F, never into node
        features). Interpolated on the same axes as the cross sections."""
        T = float(self.temperatures[0] if temperature_k is None else temperature_k)
        b0, b1, wb = self._bracket(self.burnups, burnup_mwd_kg)
        t0, t1, wt = self._bracket(self.temperatures, T)
        w = min(max(float(core_rod_frac), 0.0), 1.0)
        rods = (("out", 1.0 - w), ("in", w)) if len(self.rods) > 1 else ((self.rods[0], 1.0),)

        acc = np.zeros(self.n_groups)
        tot = 0.0
        for rod, wr in rods:
            for b, w_b in ((b0, 1.0 - wb), (b1, wb)):
                for t, w_t in ((t0, 1.0 - wt), (t1, wt)):
                    br = self._by_key.get((b, t, rod))
                    weight = wr * w_b * w_t
                    if br is None or weight <= 0.0 or not br.get("chi"):
                        continue
                    acc += weight * np.asarray(br["chi"], float)
                    tot += weight
        if tot <= 0.0:
            return None
        chi = acc / tot
        s = chi.sum()
        return tuple(float(v) for v in (chi / s if s > 0 else chi))

    # --- reporting ------------------------------------------------------------

    def max_rel_std(self) -> float:
        """Largest Monte Carlo relative standard deviation in the table.

        Report this in the paper: it bounds the statistical noise the labels inherit
        from the transport calculation.
        """
        worst = 0.0
        for b in self.blob["branches"]:
            for rec in b["materials"].values():
                for vals in rec.get("rel_std", {}).values():
                    if len(vals):
                        worst = max(worst, float(np.max(np.abs(vals))))
        return worst

    def is_converged(self) -> bool:
        """True when Monte Carlo noise is small enough to call the table physics."""
        return self.max_rel_std() <= UNCONVERGED_REL_STD

    def metadata(self) -> dict:
        """Compact provenance block for geometry_metadata."""
        return {
            "source": self.source,
            "converged": self.is_converged(),
            "schema_version": int(self.blob.get("schema_version", 2)),
            "n_groups": self.n_groups,
            "group_boundaries_ev": self.group_boundaries_ev,
            "branch_axes": {
                "burnup_mwd_kg": self.burnups,
                "temperature_k": self.temperatures,
                "rod": self.rods,
            },
            "branch_keff": {
                f"bu{b['burnup_mwd_kg']:g}_T{b['temperature_k']:g}_rod-{b['rod']}":
                    b["k_eff"] for b in self.blob["branches"]},
            "max_rel_std": self.max_rel_std(),
            "interpolation": ("linear in burnup and temperature (clamped, never "
                              "extrapolated); gray-rod blend on the insertion axis "
                              "with D combined through Sigma_tr"),
            **{k: v for k, v in self.provenance.items() if k != "depletion"},
            "depletion": self.provenance.get("depletion"),
        }


# A branch table whose worst constant carries more than this relative Monte Carlo
# standard deviation is a smoke test, not physics. Tables like that are produced by
# short runs (a few hundred histories) and are indistinguishable from converged ones
# by inspection -- they carry source="openmc" and a real OpenMC version -- so the
# loader flags them loudly rather than letting them silently label a dataset.
UNCONVERGED_REL_STD = 0.05


def load_branch_table(path: str, *, warn_unconverged: bool = True
                      ) -> Optional[BranchTable]:
    """Load a branch table if present and schema-v2, else None.

    Emits a warning for statistically under-converged tables (see
    UNCONVERGED_REL_STD). The table still loads -- short runs are legitimate while
    debugging the pipeline -- but the warning fires on every import, and
    `BranchTable.metadata()` records `converged: false` so the provenance embedded in
    every sample says so too.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    if int(blob.get("schema_version", 1)) < 2:
        return None
    table = BranchTable(blob)
    if warn_unconverged and not table.is_converged():
        import warnings
        warnings.warn(
            f"{os.path.basename(path)}: max Monte Carlo relative std dev is "
            f"{table.max_rel_std():.1%} (> {UNCONVERGED_REL_STD:.0%}). This table "
            f"looks like a short smoke-test run "
            f"({blob.get('provenance', {}).get('particles_per_batch', '?')} particles "
            f"x {blob.get('provenance', {}).get('batches', '?')} batches). Datasets "
            f"labelled with it are NOT publication grade -- rerun xs_openmc.py with "
            f"production statistics.", stacklevel=2)
    return table
