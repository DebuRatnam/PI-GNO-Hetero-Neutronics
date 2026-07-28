"""Core geometry: a Kairos KP-FHR PEBBLE-BED core, meshed for P1 FEM.

Parallel to geometry.py (Natrium hex lattice). Builds a 2D radial slice of a
pebble-bed core and returns the SAME `CoreGeometry` struct, so operators.py /
solver.py / power.py / graph_build.py run unchanged.

Physical model (2D radial, frozen state; each pebble homogenized to one node),
following the published gFHR dimensions and the licensed Hermes control layout:

  - CYLINDRICAL core (radial build out from center): a full-diameter pebble bed
    (R_bed = 120 cm) -> 60 cm graphite side reflector -> SS316H barrel + FLiBe
    downcomer + SS316H vessel, carried as one homogenized `vessel` ring.
    There is NO central reflector column: gFHR/KP-FHR pebbles are buoyant and float
    up through the whole bed cross section. (PebbleCoreConfig still exposes
    R_center_refl / R_fuel_in / R_fuel_out so an inner column or an unfueled radial
    band can be modelled; they default to off.)
  - Each pebble center is ONE node (4 cm dia). FLiBe `coolant` fills the bed gaps.
    A fraction of the pebbles are moderator-only `graphite_pebble` (NRC Hermes
    docket: a portion of the core pebbles are moderator pebbles).
  - `control_element` rigid cylinders sit in the OUTER graphite reflector -- KP-FHR
    reactivity control system (RCS): control elements insert into engineered channels
    in the side reflector, NOT the bed. Each is a rigid 16-node circle in a lined
    channel.
  - `shutdown_element` rigid X-shapes insert DIRECTLY into the packed bed -- KP-FHR
    reactivity shutdown system (RSS), no graphite thimble; direct pebble/FLiBe
    contact is what gives them their shutdown worth.
  - Insertion toggles absorber vs FLiBe-follower XS (moves k_eff). The fuel:moderator
    pebble ratio is also a reactivity lever -- it sets the carbon-to-heavy-metal
    (CHM) atom ratio, the KP-FHR's analogue of an LWR moderator-to-fuel ratio.

Group 2 is a genuine THERMAL group (graphite + FLiBe moderation). N varies per
sample. Cross sections come from materials_fhr (OpenMC-upgraded when cached).

MODELLING ASSUMPTIONS -- where this model knowingly departs from the licensed
methodology (KP-TR-024-NP). All four are echoed into every sample's
`modelling_assumptions` metadata block so a reviewer sees them without reading this:

  1. BURNUP FIELD. Kairos runs DEM pebble-flow -> ZONER -> spectral zones and
     generates constants per zone, so burnup is CORRELATED along pebble flow paths.
     Here burnup is drawn i.i.d. per pebble by default (`burnup_radial_weight = 0`),
     which is uncorrelated in space. Most of the real correlation is AXIAL and is
     invisible to a radial slice; the radial part -- slower near-wall pebble flow ->
     longer residence -> higher burnup at the bed edge -- can be switched on with
     `burnup_radial_weight > 0`. The magnitude is proprietary, hence opt-in.
  2. AXIAL SHAPE. This is a slice of the CYLINDRICAL section only. The real core adds
     upper and lower conic regions, a defueling chute and a fuel insertion region;
     the transverse buckling in PhysicsConfig.axial_buckling assumes a straight
     cylinder of the active height and does not represent those.
  3. REFLECTOR. Modelled as solid graphite. The real reflector is blocks with axial
     coolant channels, instrumentation penetrations, inter-block gaps and keys, all
     of which carry FLiBe and soften the reflector's return current.
  4. SOLUTION METHOD. The licensed methodology has no diffusion step at all: it is
     full-core 3D explicit Serpent 2 Monte Carlo with burnup. Multigroup diffusion is
     this project's choice; validate_openmc.py measures the resulting reactivity bias
     and radial power-shape error, and that measurement is what justifies the labels.

Sources: NRC KP-FHR Core Design and Analysis Methodology topical report KP-TR-024-NP
Rev 0 (ML24095A258, April 2024; earlier revisions ML21272A383, ML23195A130) for the
RCS/RSS layout, the fuel-annulus pebble form, buoyancy, moderator pebbles and CHM
ratio; the Hermes PSAR for the 4 control + 3 shutdown element count; the Kairos gFHR
benchmark (Satvat et al., Nucl. Eng. Des. 384 (2021) 111461; INL Virtual Test Bed
gFHR description) for every dimension. Exact Hermes element geometry is proprietary;
r_ctrl (2.6 cm) and control_offset (7.9 cm) are the published gFHR rod.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from datagen_config import PebbleCoreConfig
from geometry import CoreGeometry, triangle_areas, nodal_volumes
from materials_fhr import BRANCH, MATERIAL_IDS, xs_for_id, up_scatter_for_id
from xs_branch import check_axis_coverage
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
    """RSA pack non-overlapping pebble-center disks across the bed
    (R_center_refl .. R_bed), avoiding the shutdown-channel footprints.

    R_center_refl is 0 on the gFHR/KP-FHR build, so the bed is a full disk; the
    annular form is kept only for cores that do configure a central column."""
    r_in = cfg.R_center_refl + cfg.r_peb if cfg.R_center_refl > 0.0 else 0.0
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


def _apply_radial_burnup(burn: np.ndarray, coords: np.ndarray,
                         cfg: PebbleCoreConfig, lo: float, hi: float) -> np.ndarray:
    """Blend a radial ramp into an i.i.d. burnup draw (see burnup_radial_weight).

    b <- (1 - w) * b_iid + w * (lo + (hi - lo) * r / R_bed)

    w = 0 returns `burn` untouched, so the default path and its random stream are
    bit-identical to the uncorrelated model. w = 1 makes burnup a pure function of
    radius, which is unphysical on its own -- multi-pass recirculation keeps every
    burnup present at every radius -- so intermediate values are the meaningful ones.
    Higher burnup goes to the bed EDGE: near-wall pebbles flow slower, reside longer
    and burn deeper (the radial shadow of the DEM flow field ZONER resolves properly).

    (lo, hi) is the range `burn` was drawn over, so this works unchanged for the
    branch-table path (physical MWd/kgHM) and the legacy fallback (burnup fraction).
    """
    w = float(getattr(cfg, "burnup_radial_weight", 0.0))
    if w <= 0.0:
        return burn
    w = min(w, 1.0)
    rho = np.clip(np.hypot(coords[:, 0], coords[:, 1]) / max(cfg.R_bed, 1e-12), 0.0, 1.0)
    return (1.0 - w) * burn + w * (lo + (hi - lo) * rho)


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
    burnup_radial_weight = float(getattr(cfg, "burnup_radial_weight", 0.0))

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

    # --- background grid for the SOLID regions outside the bed. Two resolutions:
    # the 60 cm graphite reflector has no sub-pebble structure and is meshed coarse
    # (structure_step_frac), while the barrel / downcomer / vessel stack is only a few
    # cm per layer and keeps the fine (coolant) step so each layer is still resolved.
    # The pebble bed itself is handled separately (target-count coolant below).
    fine_step = cfg.r_peb * cfg.coolant_step_frac
    refl_step = cfg.r_peb * cfg.structure_step_frac
    ctree = cKDTree(ctrl) if len(ctrl) else None

    def _fill(r_lo: float, r_hi: float, step: float, mat_id: int,
              avoid_control: bool = False) -> None:
        """Jittered square grid clipped to the annulus r_lo < r <= r_hi."""
        if r_hi <= r_lo:
            return
        g = np.arange(-r_hi, r_hi + step, step)
        for gx in g:
            for gy in g:
                x = gx + (rng.random() - 0.5) * step * 0.35
                y = gy + (rng.random() - 0.5) * step * 0.35
                r = float(np.hypot(x, y))
                if not (r_lo < r <= r_hi):
                    continue
                p = (x, y)
                if avoid_control and ctree is not None and (
                        ctree.query(p)[0] < ctrl_wall_r + 0.5 * step):
                    continue                    # control elements own their channels
                coords.append(p); mats.append(mat_id)

    # central graphite column, when one is configured (gFHR/KP-FHR: none)
    _fill(0.0, cfg.R_center_refl, fine_step, refl_id)
    # outer graphite reflector (holds the control channels)
    _fill(cfg.R_bed, cfg.R_refl, refl_step, refl_id, avoid_control=True)
    # barrel + downcomer + vessel, homogenized into one `vessel` ring
    _fill(cfg.R_refl, cfg.R_vessel, fine_step, vess_id)

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
        check_axis_coverage(BRANCH, "burnup", *cfg.burnup_mwd_kg_range, label="fhr")
        check_axis_coverage(BRANCH, "temperature", *cfg.temperature_k_range, label="fhr")
        burn_mwd = rng.uniform(*cfg.burnup_mwd_kg_range, N)
        temp_k = rng.uniform(*cfg.temperature_k_range, N)
        burn_mwd = _apply_radial_burnup(burn_mwd, coords, cfg,
                                        *cfg.burnup_mwd_kg_range)
        burn_mwd[~fuel_mask] = 0.0
        legacy_perturb = False
    else:
        burn = rng.uniform(0.0, cfg.burnup_perturb, N)
        burn = _apply_radial_burnup(burn, coords, cfg, 0.0, cfg.burnup_perturb)
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
        "R_refl": cfg.R_refl, "R_barrel": cfg.R_barrel,
        "R_downcomer": cfg.R_downcomer, "R_vessel": cfg.R_vessel,
        "core_shape": ("cylindrical bed (no central reflector column)"
                       if cfg.R_center_refl <= 0.0 else "annular bed"),
        "upscatter_included": True,
        "upscatter_note": "thermal->fast Ss21, operator-level (in A), not in node XS row",
        # Where this model knowingly departs from the licensed KP-FHR methodology
        # (NRC KP-TR-024-NP Rev 0, ML24095A258). Carried in every sample so the
        # departures are auditable without reading the source. See the module
        # docstring for the long form.
        "methodology_reference": ("NRC KP-FHR Core Design and Analysis Methodology, "
                                  "KP-TR-024-NP Rev 0 (ML24095A258, April 2024)"),
        "modelling_assumptions": {
            "burnup_field": (
                "i.i.d. per pebble, spatially uncorrelated" if burnup_radial_weight
                <= 0.0 else
                f"radial ramp blended at weight {burnup_radial_weight:g} "
                f"(higher burnup toward the bed edge)"),
            "burnup_radial_weight": float(burnup_radial_weight),
            "burnup_field_departure": (
                "licensed method derives burnup zones from DEM pebble flow via ZONER, "
                "so burnup is correlated along flow paths; most of that correlation "
                "is axial and invisible to a 2D radial slice"),
            "axial_shape": (
                "2D slice of the cylindrical bed section only; upper/lower conic "
                "regions, defueling chute and fuel insertion region not represented, "
                "and the transverse buckling assumes a straight cylinder"),
            "reflector": (
                "solid graphite annulus; real reflector is blocks with axial coolant "
                "channels, instrumentation penetrations, inter-block gaps and keys"),
            "solution_method": (
                "multigroup P1-FEM diffusion; the licensed methodology uses full-core "
                "3D explicit Serpent 2 Monte Carlo with burnup and no diffusion step. "
                "Run validate_openmc.py for the resulting reactivity and power-shape "
                "bias"),
        },
    }

    return CoreGeometry(
        material_state=material_state, coordinates=coords, boundary_mask=boundary_mask,
        cross_sections=xs, control_rod_cells=control_rod_cells, elements=elements,
        boundary_edges=boundary_edges, nodal_volume=V, mesh=None, upscatter=upscatter,
        layout_name=layout_name, assembly_metadata=meta,
    )
