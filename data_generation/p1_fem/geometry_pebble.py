"""Core geometry: a Kairos KP-FHR PEBBLE-BED core, meshed for P1 FEM.

Parallel to geometry.py (Natrium hex lattice). Builds a 2D radial slice of a
pebble-bed core and returns the SAME `CoreGeometry` struct, so operators.py /
solver.py / power.py / graph_build.py run unchanged.

Physical model (2D radial, frozen state; each pebble homogenized to one node),
faithful to the real KP-FHR / generic-FHR (gFHR) design:

  - ANNULAR core (radial build out from center): central graphite reflector column
    -> inner unfueled pebble zone (graphite pebbles) -> FUELED pebble annulus (fuel
    + a fraction of graphite moderator pebbles) -> outer unfueled pebble zone ->
    outer graphite reflector -> steel vessel.
  - Each pebble center is ONE node (4 cm dia). FLiBe `coolant` fills the bed gaps.
  - 4 `control_element` rigid cylinders sit in the OUTER graphite reflector -- NRC
    KP-FHR: control elements insert into the side graphite reflector, NOT the bed.
    Each is a rigid 16-node circle inside a graphite-lined channel.
  - 3 `shutdown_element` rigid X-shapes insert into the inner fueled bed (NRC), each
    in a graphite-lined channel separating the B4C from the pebbles/FLiBe.
  - Insertion toggles absorber vs FLiBe-follower XS (moves k_eff). The fuel:moderator
    pebble ratio is also a reactivity lever.

Group 2 is a genuine THERMAL group (graphite + FLiBe moderation). N varies per
sample. Cross sections come from materials_fhr (OpenMC-upgraded when cached).

Sources: NRC KP-FHR Core Design & Analysis Methodology (ML21272A383); Kairos gFHR
benchmark (Satvat et al. 2021). Exact Hermes control-element diameter is
proprietary; r_ctrl is a documented representative value (gFHR 5.2 cm rod proxy).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from datagen_config import PebbleCoreConfig
from geometry import CoreGeometry, triangle_areas, nodal_volumes
from materials_fhr import BRANCH, MATERIAL_IDS, xs_for_id, up_scatter_for_id
from xs_common import n_xs_cols, nusf_slice, sr_slice


# --- footprint tests ---------------------------------------------------------

def _in_cross(p, center, arm_len: float, arm_w: float, angle: float) -> bool:
    """True if p is inside an X (two crossed bars) centered at `center`, rotated
    by `angle`."""
    dx, dy = p[0] - center[0], p[1] - center[1]
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    bar1 = (abs(lx) <= arm_len) and (abs(ly) <= arm_w / 2.0)
    bar2 = (abs(ly) <= arm_len) and (abs(lx) <= arm_w / 2.0)
    return bar1 or bar2


# --- rigid structure node generators ----------------------------------------

def _disk_nodes(center, r: float, n: int) -> List[tuple]:
    """Rigid filled circle: centre + an inner ring (n//2) + the n-node boundary ring,
    all on exact circles (x-cx)^2+(y-cy)^2=r^2 via cos/sin at even angles. Filling
    the interior makes every control-element triangle pure control material (no
    majority-vote spikes) and gives a clean FEM disk."""
    pts = [(center[0], center[1])]
    for rad, nn in ((0.5 * r, max(6, n // 2)), (r, n)):
        for t in np.linspace(0.0, 2 * np.pi, nn, endpoint=False):
            pts.append((center[0] + rad * np.cos(t), center[1] + rad * np.sin(t)))
    return pts


def _ring_nodes(center, r: float, n: int) -> List[tuple]:
    """n evenly-spaced nodes on a circle of radius r (channel-wall lining)."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return [(center[0] + r * np.cos(t), center[1] + r * np.sin(t)) for t in th]


def _cross_fill_nodes(center, arm_len, arm_w, angle, step) -> List[tuple]:
    """Rigid X interior: evenly-spaced (un-jittered) grid points inside the X."""
    pts = []
    g = np.arange(-arm_len, arm_len + step, step)
    for gx in g:
        for gy in g:
            p = (center[0] + gx, center[1] + gy)
            if _in_cross(p, center, arm_len, arm_w, angle):
                pts.append(p)
    return pts


# --- structure placement -----------------------------------------------------

def _structure_centers(cfg: PebbleCoreConfig
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """(control_centers [n_control,2] in the OUTER REFLECTOR,
        shutdown_centers [n_shutdown,2] in the INNER FUELED BED)."""
    Rc = cfg.R_bed + cfg.control_offset            # into the graphite reflector
    ang_c = np.linspace(0.0, 2 * np.pi, cfg.n_control, endpoint=False) + np.pi / 4.0
    ctrl = np.stack([Rc * np.cos(ang_c), Rc * np.sin(ang_c)], axis=1)

    Rs = cfg.R_fuel_out * cfg.shutdown_ring_frac   # inner fueled bed
    ang_s = np.linspace(0.0, 2 * np.pi, cfg.n_shutdown, endpoint=False) + np.pi / 2.0
    shut = np.stack([Rs * np.cos(ang_s), Rs * np.sin(ang_s)], axis=1)
    return ctrl, shut


def _rsa_pebbles(cfg: PebbleCoreConfig, shut: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """RSA pack non-overlapping pebble-center disks in the ANNULAR bed
    (R_center_refl .. R_bed), avoiding the shutdown-channel footprints."""
    r_in = cfg.R_center_refl + cfg.r_peb
    r_out = cfg.R_bed - cfg.r_peb
    min_sep2 = (2.0 * cfg.r_peb) ** 2
    area = np.pi * (r_out ** 2 - r_in ** 2)
    target = int(cfg.packing_fraction * area / (np.pi * cfg.r_peb ** 2))
    max_trials = 60 * max(target, 1)
    # keep pebbles just clear of the shutdown blade (no channel liner) so pebbles
    # sit directly against it
    sh_len = cfg.x_arm_len + cfg.r_peb
    sh_w = cfg.x_arm_w + 2 * cfg.r_peb

    def blocked(x, y):
        for sc in shut:
            if _in_cross((x, y), sc, sh_len, sh_w, np.pi / 4.0):
                return True
        return False

    centers = np.empty((target, 2))
    n = 0
    for _ in range(max_trials):
        if n >= target:
            break
        rr = np.sqrt(rng.random() * (r_out ** 2 - r_in ** 2) + r_in ** 2)
        th = 2 * np.pi * rng.random()
        x, y = rr * np.cos(th), rr * np.sin(th)
        if blocked(x, y):
            continue
        if n and np.min((centers[:n, 0] - x) ** 2 + (centers[:n, 1] - y) ** 2) < min_sep2:
            continue
        centers[n] = (x, y)
        n += 1
    return centers[:n]


def _pebble_materials(cfg: PebbleCoreConfig, pebbles: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Assign each pebble fuel vs graphite by radial zone: fuel only in the fueled
    annulus (R_fuel_in..R_fuel_out); a graphite_pebble_frac of those are moderator
    pebbles; the inner/outer unfueled bands are all graphite pebbles."""
    r = np.hypot(pebbles[:, 0], pebbles[:, 1])
    fueled = (r >= cfg.R_fuel_in) & (r <= cfg.R_fuel_out)
    is_mod = rng.random(pebbles.shape[0]) < cfg.graphite_pebble_frac
    mat = np.where(fueled & ~is_mod, MATERIAL_IDS["fuel_pebble"],
                   MATERIAL_IDS["graphite_pebble"])
    return mat.astype(np.int64)


def _free_boundary_edges(elements: np.ndarray) -> np.ndarray:
    """Edges belonging to exactly one triangle (the domain boundary)."""
    e = np.concatenate([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def make_pebble_core(cfg: PebbleCoreConfig, *, layout_name: str = "kpfhr",
                     insert_control: float = 1.0, insert_shutdown: float = 0.0,
                     graphite_pebble_frac: Optional[float] = None,
                     burnup_field: Optional[np.ndarray] = None,
                     rng: Optional[np.random.Generator] = None) -> CoreGeometry:
    """Build one frozen KP-FHR pebble-bed sample -> CoreGeometry.

    insert_control / insert_shutdown are the gray-rod insertion DEPTH in [0,1] of the
    control / shutdown bank: 0 = withdrawn (FLiBe follower), 1 = full B4C absorber,
    between = partially inserted (axially-averaged gray absorber, materials_fhr.xs_for).
    graphite_pebble_frac overrides the config default (per-sample reactivity lever).
    """
    rng = rng or np.random.default_rng()
    if graphite_pebble_frac is not None:
        from dataclasses import replace
        cfg = replace(cfg, graphite_pebble_frac=float(graphite_pebble_frac))

    ctrl, shut = _structure_centers(cfg)
    pebbles = _rsa_pebbles(cfg, shut, rng)
    peb_mat = _pebble_materials(cfg, pebbles, rng)
    n_peb = pebbles.shape[0]
    ptree = cKDTree(pebbles) if n_peb else None

    coords: List[tuple] = [tuple(p) for p in pebbles]
    mats: List[int] = list(peb_mat)

    ctrl_id = MATERIAL_IDS["control_element"]
    shut_id = MATERIAL_IDS["shutdown_element"]
    refl_id = MATERIAL_IDS["reflector"]
    cool_id = MATERIAL_IDS["coolant"]
    vess_id = MATERIAL_IDS["vessel"]

    # --- rigid control elements (in the outer reflector) + graphite channel wall
    ctrl_wall_r = cfg.r_ctrl + cfg.channel_wall
    for cc in ctrl:
        for p in _disk_nodes(cc, cfg.r_ctrl, cfg.n_circle_nodes):
            coords.append(p); mats.append(ctrl_id)
        for p in _ring_nodes(cc, ctrl_wall_r, cfg.n_circle_nodes):
            coords.append(p); mats.append(refl_id)          # graphite lining

    # --- rigid shutdown elements: blades inserted DIRECTLY into the packed bed.
    # Mark-1 PB-FHR / KP-FHR: shutdown blades push straight into the pebbles (no
    # graphite thimble) -- direct pebble/FLiBe contact is what gives the high
    # shutdown worth. So NO channel lining here (unlike the reflector control rods).
    estep = cfg.r_peb
    for sc in shut:
        for p in _cross_fill_nodes(sc, cfg.x_arm_len, cfg.x_arm_w, np.pi / 4.0, estep):
            coords.append(p); mats.append(shut_id)

    # --- background grid for the SOLID regions: central reflector, outer reflector,
    # steel vessel. The pebble bed is handled separately (target-count coolant below).
    step = cfg.r_peb * cfg.coolant_step_frac
    g = np.arange(-cfg.R_vessel, cfg.R_vessel + step, step)
    ctree = cKDTree(ctrl) if len(ctrl) else None
    for gx in g:
        for gy in g:
            x = gx + (rng.random() - 0.5) * step * 0.35
            y = gy + (rng.random() - 0.5) * step * 0.35
            r = float(np.hypot(x, y))
            if r > cfg.R_vessel:
                continue
            p = (x, y)
            if r > cfg.R_refl:
                coords.append(p); mats.append(vess_id)       # steel vessel shell
            elif r > cfg.R_bed:
                # outer graphite reflector; control elements own their channels
                if ctree is not None and ctree.query(p)[0] < ctrl_wall_r + 0.5 * step:
                    continue
                coords.append(p); mats.append(refl_id)
            elif r <= cfg.R_center_refl:
                coords.append(p); mats.append(refl_id)       # central reflector column
            # (R_center_refl < r < R_bed) is the bed -> coolant placed below

    # --- interstitial FLiBe: place coolant nodes to hit the real pebble:FLiBe VOLUME
    # ratio (~60:40). coolant_per_pebble=1.0 is calibrated to ~60% pebble volume
    # (coolant sits in tight inter-pebble gaps, so count != volume 1:1).
    n_cool = int(round(n_peb * cfg.coolant_per_pebble))
    r_in2, r_out2 = cfg.R_center_refl ** 2, cfg.R_bed ** 2
    placed, trials = 0, 0
    while placed < n_cool and trials < 60 * max(n_cool, 1):
        trials += 1
        rr = np.sqrt(rng.random() * (r_out2 - r_in2) + r_in2)
        th = 2 * np.pi * rng.random()
        p = (rr * np.cos(th), rr * np.sin(th))
        if ptree is not None and ptree.query(p)[0] < cfg.r_peb * 0.9:
            continue                                          # on a pebble
        if any(_in_cross(p, sc, cfg.x_arm_len, cfg.x_arm_w, np.pi / 4.0) for sc in shut):
            continue                                          # in a shutdown blade
        coords.append(p); mats.append(cool_id); placed += 1

    coords = np.asarray(coords, dtype=float)
    material_state = np.asarray(mats, dtype=np.int64)
    N = coords.shape[0]

    # control / shutdown insertion DEPTHS (gray-rod, in [0,1]). insert_control may be
    # a SCALAR (ganged bank: all n_control elements at one height -- normal symmetric
    # operation) OR a per-element SEQUENCE of length n_control (independent insertion:
    # tilt / stuck-rod off-normal states). insert_shutdown likewise for the bed bank.
    # depth=0 -> withdrawn (FLiBe follower), depth=1 -> full B4C absorber, between ->
    # partially inserted as an axially-averaged gray absorber (materials_fhr.xs_for).
    ctrl_depths = np.clip(np.broadcast_to(
        np.asarray(insert_control, float), (cfg.n_control,)), 0.0, 1.0)
    shut_depths = np.clip(np.broadcast_to(
        np.asarray(insert_shutdown, float), (cfg.n_shutdown,)), 0.0, 1.0)
    ctree_d = cKDTree(ctrl) if len(ctrl) else None    # node -> nearest element index
    stree_d = cKDTree(shut) if len(shut) else None

    # triangulate the full cloud (convex disk -> no exterior holes)
    tri = Delaunay(coords)
    elements = tri.simplices.astype(np.int64)
    areas = triangle_areas(coords, elements)
    keep = areas > 1e-9
    elements, areas = elements[keep], areas[keep]

    boundary_edges = _free_boundary_edges(elements)
    boundary_mask = np.zeros(N, bool)
    boundary_mask[np.unique(boundary_edges)] = True
    V = nodal_volumes(coords, elements, areas)

    # --- per-node cross sections -------------------------------------------------
    # Insertion toggles absorber XS for control/shutdown. Burnup and temperature vary
    # PER PEBBLE: recirculation means pebbles of every burnup coexist at every radius,
    # which is the physical state this dataset covers.
    fuel_mask = material_state == MATERIAL_IDS["fuel_pebble"]
    G = len(xs_for_id(0).D)
    ncol = n_xs_cols(G)
    nf, sr = nusf_slice(G), sr_slice(G)
    xs = np.zeros((N, ncol))

    # core-average insertion over BOTH control families: the spectrum shift inserted
    # absorbers impose on every other material (measured by the rod branch cases).
    all_depths = np.concatenate([ctrl_depths, shut_depths]) if (
        len(ctrl_depths) or len(shut_depths)) else np.zeros(0)
    core_rod_frac = float(all_depths.mean()) if all_depths.size else 0.0

    if BRANCH is not None:
        # OpenMC branch table: physical burnup [MWd/kgHM] + temperature [K] per node.
        burn_mwd = rng.uniform(*cfg.burnup_mwd_kg_range, N)
        temp_k = rng.uniform(*cfg.temperature_k_range, N)
        burn_mwd[~fuel_mask] = 0.0
        legacy_perturb = False
    else:
        burn = rng.uniform(0.0, cfg.burnup_perturb, N)
        temp = 1.0 + rng.uniform(-cfg.temp_perturb, cfg.temp_perturb, N)
        legacy_perturb = True

    depth_per_node = np.zeros(N)
    for i in range(N):
        m = int(material_state[i])
        if m == ctrl_id and ctree_d is not None:
            d = float(ctrl_depths[ctree_d.query(coords[i])[1]])   # this element's depth
        elif m == shut_id and stree_d is not None:
            d = float(shut_depths[stree_d.query(coords[i])[1]])
        else:
            d = 1.0                                               # ignored by non-control
        depth_per_node[i] = d
        if legacy_perturb:
            xs[i] = xs_for_id(m, insert_frac=d).as_row()
        else:
            xs[i] = xs_for_id(m, insert_frac=d, burnup_mwd_kg=float(burn_mwd[i]),
                              temperature_k=float(temp_k[i]),
                              core_rod_frac=core_rod_frac).as_row()

    if legacy_perturb:
        # fallback (no OpenMC table): DIRECTIONAL fissile depletion lowers nuSf, and
        # fission-product poison (Xe/Sm) raises the THERMAL removal Sr2; temp is a
        # symmetric +/- wiggle on nuSf. See PebbleCoreConfig. NOT publication-grade.
        xs[fuel_mask, nf] *= ((1.0 - burn[fuel_mask]) * temp[fuel_mask])[:, None]
        xs[fuel_mask, sr.start + 1] *= (
            1.0 + cfg.burnup_poison_coeff * burn[fuel_mask])   # Sr2 poison

    control_rod_cells = np.where(np.isin(material_state, [ctrl_id, shut_id]))[0].astype(np.int64)

    # per-node thermal up-scatter Ss_{g2->g1}; assembled into A by
    # operators.assemble_AF. Kept out of the per-node XS row (schema down-scatter-only).
    # With a branch table this is the tallied g2->g1 transfer at the node's own
    # temperature -- up-scatter IS a thermal-motion effect, so that matters.
    if legacy_perturb:
        upscatter = np.array([up_scatter_for_id(int(m)) for m in material_state],
                             dtype=float)
    else:
        upscatter = np.array([
            up_scatter_for_id(int(material_state[i]), insert_frac=depth_per_node[i],
                              burnup_mwd_kg=float(burn_mwd[i]),
                              temperature_k=float(temp_k[i]),
                              core_rod_frac=core_rod_frac)
            for i in range(N)], dtype=float)

    meta = {
        "reactor_type": "fhr",
        "core_area_cm2": float(areas.sum()),
        "n_pebbles": int(n_peb),
        "n_fuel_pebbles": int(fuel_mask.sum()),
        "n_graphite_pebbles": int((material_state == MATERIAL_IDS["graphite_pebble"]).sum()),
        "graphite_pebble_frac": float(cfg.graphite_pebble_frac),
        "packing_fraction_actual": float(
            n_peb * cfg.r_peb ** 2 / (cfg.R_bed ** 2 - cfg.R_center_refl ** 2)),
        "n_control": int(cfg.n_control),
        "n_control_inserted": int((ctrl_depths > 0.0).sum()),
        "n_shutdown": int(cfg.n_shutdown),
        "n_shutdown_inserted": int((shut_depths > 0.0).sum()),
        "control_depths": [float(d) for d in ctrl_depths],
        "shutdown_depths": [float(d) for d in shut_depths],
        "control_ganged": bool(np.ptp(ctrl_depths) < 1e-12) if len(ctrl_depths) else True,
        "mean_control_depth": float(ctrl_depths.mean()) if len(ctrl_depths) else 0.0,
        "mean_shutdown_depth": float(shut_depths.mean()) if len(shut_depths) else 0.0,
        "R_center_refl": cfg.R_center_refl, "R_fuel_in": cfg.R_fuel_in,
        "R_fuel_out": cfg.R_fuel_out, "R_bed": cfg.R_bed,
        "R_refl": cfg.R_refl, "R_vessel": cfg.R_vessel,
        "upscatter_included": True,
        "upscatter_note": "thermal->fast Ss21, operator-level (in A), not in node XS row",
    }

    return CoreGeometry(
        material_state=material_state, coordinates=coords, boundary_mask=boundary_mask,
        cross_sections=xs, control_rod_cells=control_rod_cells, elements=elements,
        boundary_edges=boundary_edges, nodal_volume=V, mesh=None, upscatter=upscatter,
        layout_name=layout_name, assembly_metadata=meta,
    )
