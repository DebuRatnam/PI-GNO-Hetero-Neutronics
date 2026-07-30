"""Core geometry: a Natrium-inspired HEXAGONAL-duct lattice, meshed for P1 FEM.

Physical model (2D radial, homogenized-assembly diffusion, DIF3D-style):
  - Assemblies sit on a hex lattice (rings 0..R). Radial roles: inner-enrichment
    fuel -> outer-enrichment fuel -> reflector rings -> shield rings. 9 primary +
    4 secondary control positions replace selected fuel assemblies.
  - Each hexagon gets an IDENTICAL structured submesh (spatial invariance across
    physically-identical assemblies): a fan interior (assembly material) wrapped by
    a homogenized duct ring (HT9 wall + gap, `duct` material).
  - The thin inter-assembly SODIUM GAPS are stitched with a Delaunay triangulation
    over the hex-boundary nodes plus explicit gap nodes (`coolant`), so the gap is a
    real physical region, not a blurred gradient.
  - Control insertion toggles absorber vs sodium-follower cross sections; a small
    per-assembly XS perturbation stands in for burnup/temperature spread.

Domain boundary (for the Marshak vacuum BC) is the CONVEX HULL of the node cloud --
robust to the mixed structured/Delaunay interior.

One NODE = one mesh vertex. N varies per sample. Both groups are FAST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

from datagen_config import HexCoreConfig
from materials import MATERIAL_IDS, ID_TO_MATERIAL, branch, xs_for_id
from xs_branch import check_axis_coverage
from xs_common import n_xs_cols

SQRT3 = np.sqrt(3.0)
DUCT_RING_FRAC = 0.18   # radial fraction of each hex homogenized as duct (HT9+gap)


@dataclass
class CoreGeometry:
    material_state: np.ndarray   # [N] int material id per node
    coordinates: np.ndarray      # [N, 2] node (x, y) in cm
    boundary_mask: np.ndarray    # [N] bool, True on domain boundary nodes
    cross_sections: np.ndarray   # [N, 7] XS (D1,D2,Sr1,Sr2,Ss12,nuSf1,nuSf2)
    control_rod_cells: np.ndarray  # [Ncr] node ids of control-assembly nodes
    elements: np.ndarray         # [T, 3] triangle -> node indices (P1 elements)
    boundary_edges: np.ndarray   # [B, 2] mesh edges on the domain boundary
    nodal_volume: np.ndarray     # [N] lumped nodal volume
    mesh: object                 # kept for compatibility (may be None)
    layout_name: str
    assembly_metadata: Dict = field(default_factory=dict)
    # Optional per-node thermal up-scatter Ss_{g2->g1} [N] (FHR only). Applied at the
    # operator level in assemble_AF; None (hex) -> pure down-scatter, unchanged.
    upscatter: Optional[np.ndarray] = None

    @property
    def n_nodes(self) -> int:
        return int(self.material_state.shape[0])


# --- shared triangulation helpers (reused by operators.py) -------------------

def triangle_areas(coords: np.ndarray, elements: np.ndarray) -> np.ndarray:
    p0, p1, p2 = coords[elements[:, 0]], coords[elements[:, 1]], coords[elements[:, 2]]
    return 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def nodal_volumes(coords: np.ndarray, elements: np.ndarray,
                  areas: Optional[np.ndarray] = None) -> np.ndarray:
    if areas is None:
        areas = triangle_areas(coords, elements)
    V = np.zeros(coords.shape[0])
    third = areas / 3.0
    for k in range(3):
        np.add.at(V, elements[:, k], third)
    return V


# --- hex lattice -------------------------------------------------------------

def _hex_cells(rings: int) -> List[Tuple[int, int, int]]:
    """Axial (q, r) cells within `rings`, with hex-distance d. Returns (q, r, d)."""
    cells = []
    for q in range(-rings, rings + 1):
        for r in range(max(-rings, -q - rings), min(rings, -q + rings) + 1):
            d = (abs(q) + abs(r) + abs(q + r)) // 2
            cells.append((q, r, d))
    return cells


def _axial_to_xy(q: int, r: int, pitch: float) -> Tuple[float, float]:
    return pitch * (q + r / 2.0), pitch * (SQRT3 / 2.0) * r


def _hex_corners(cx: float, cy: float, Rc: float) -> np.ndarray:
    """6 corners of a flat-topped hexagon (edges face the 6 lattice neighbours)."""
    ang = np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.stack([cx + Rc * np.cos(ang), cy + Rc * np.sin(ang)], axis=1)


def _point_in_hex(p: np.ndarray, center: np.ndarray, Rc: float) -> bool:
    corners = _hex_corners(center[0], center[1], Rc)
    x, y = p
    inside = False
    for i in range(6):
        xi, yi = corners[i]
        xj, yj = corners[(i + 5) % 6]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
    return inside


# --- role + control assignment ----------------------------------------------

def _assign_roles(cells, hx: HexCoreConfig, enrichment_boundary: int,
                  reflector_rings: int, shield_rings: int):
    """material id per assembly by radial ring; None outside the core."""
    roles = {}
    fuel_edge = hx.fuel_rings
    refl_edge = fuel_edge + reflector_rings
    shld_edge = refl_edge + shield_rings
    for (q, r, d) in cells:
        if d < fuel_edge:
            roles[(q, r)] = (MATERIAL_IDS["fuel_inner"] if d <= enrichment_boundary
                             else MATERIAL_IDS["fuel_outer"])
        elif d < refl_edge:
            roles[(q, r)] = MATERIAL_IDS["reflector"]
        elif d < shld_edge:
            roles[(q, r)] = MATERIAL_IDS["shield"]
    return roles


def _pick_control(roles, hx: HexCoreConfig):
    """Deterministically choose 9 primary + 4 secondary control positions among fuel
    assemblies (innermost -> secondary, next spread -> primary). Returns dict
    (q,r)->material id, overwriting those fuel roles."""
    fuel = [(q, r) for (q, r), m in roles.items()
            if m in (MATERIAL_IDS["fuel_inner"], MATERIAL_IDS["fuel_outer"])
            and (q, r) != (0, 0)]
    # sort by (distance, angle) for a stable, roughly symmetric spread
    def key(c):
        q, r = c
        d = (abs(q) + abs(r) + abs(q + r)) // 2
        return (d, np.arctan2(r, q + r / 2.0))
    fuel.sort(key=key)
    n = hx.n_primary_control + hx.n_secondary_control
    if not fuel or n == 0:
        return {}
    idx = np.linspace(0, len(fuel) - 1, num=min(n, len(fuel))).round().astype(int)
    picked = [fuel[i] for i in dict.fromkeys(idx.tolist())]  # unique, ordered
    picked.sort(key=key)
    ctrl = {}
    for i, c in enumerate(picked):
        ctrl[c] = (MATERIAL_IDS["secondary_control"] if i < hx.n_secondary_control
                   else MATERIAL_IDS["primary_control"])
    return ctrl


# --- per-hex structured submesh ----------------------------------------------

def _hex_submesh(center: np.ndarray, Rc: float, assembly_mat: int):
    """Identical structured submesh for one hexagon.
    Returns local node coords [13,2], node materials [13], triangles [18,3] (local),
    and the 6 outer-corner local indices. Layout: center(0), inner ring 1..6 @
    Ri (assembly material), outer ring 7..12 @ Rc (duct)."""
    Ri = Rc * (1.0 - DUCT_RING_FRAC)
    inner = _hex_corners(center[0], center[1], Ri)
    outer = _hex_corners(center[0], center[1], Rc)
    coords = np.concatenate([center[None, :], inner, outer], axis=0)  # [13,2]
    mats = np.array([assembly_mat] + [assembly_mat] * 6 + [MATERIAL_IDS["duct"]] * 6,
                    dtype=np.int64)
    tris = []
    for k in range(6):
        a_in, b_in = 1 + k, 1 + (k + 1) % 6      # inner ring
        a_out, b_out = 7 + k, 7 + (k + 1) % 6     # outer ring
        tris.append([0, a_in, b_in])              # interior fan (assembly)
        tris.append([a_in, a_out, b_out])         # duct annulus
        tris.append([a_in, b_out, b_in])
    outer_idx = list(range(7, 13))
    return coords, mats, np.array(tris, dtype=np.int64), outer_idx


# --- core assembly -----------------------------------------------------------

def make_core(hx: HexCoreConfig, *, layout_name: str = "default",
              reflector_rings: Optional[int] = None,
              shield_rings: Optional[int] = None,
              enrichment_boundary: Optional[int] = None,
              insert_fraction: Optional[float] = None,
              rng: Optional[np.random.Generator] = None) -> CoreGeometry:
    """Build one hex-lattice core. Ring counts / enrichment boundary / control
    insertion can be overridden per sample for dataset variability."""
    rng = rng or np.random.default_rng()
    reflector_rings = hx.reflector_rings if reflector_rings is None else reflector_rings
    shield_rings = hx.shield_rings if shield_rings is None else shield_rings
    enrichment_boundary = (hx.enrichment_boundary_ring if enrichment_boundary is None
                           else enrichment_boundary)
    total_rings = hx.fuel_rings + reflector_rings + shield_rings - 1

    Rc = (hx.pitch_cm - hx.gap_cm) / SQRT3
    cells = _hex_cells(total_rings)
    roles = _assign_roles(cells, hx, enrichment_boundary, reflector_rings, shield_rings)
    ctrl = _pick_control(roles, hx)
    roles.update(ctrl)

    # per-rod control insertion DEPTH (gray-rod, in [0,1]): each control assembly is
    # drawn independently over the configured range (or pinned when insert_fraction
    # is given). depth=0 -> withdrawn (sodium follower), depth=1 -> full B4C absorber,
    # between -> a partially-inserted rod carried as an axially-averaged gray absorber
    # (materials.xs_for). Generalizes the old binary insert/withdraw and densely fills
    # the reactivity axis; real cores hold rods at intermediate heights to trim k_eff.
    ctrl_cells = list(ctrl.keys())
    if insert_fraction is None:
        depths = rng.uniform(*hx.control_insert_fraction, size=len(ctrl_cells))
    else:
        depths = np.full(len(ctrl_cells), float(insert_fraction))
    depth_by_cell = {tuple(c): float(d) for c, d in zip(ctrl_cells, depths)}

    # build per-hex submeshes
    coords_parts, mat_parts, elem_parts = [], [], []
    node_hex, node_offset = [], 0
    hex_centers, hex_outer_nodes, hex_meta = [], [], []
    for (q, r) in sorted(roles.keys()):
        cx, cy = _axial_to_xy(q, r, hx.pitch_cm)
        center = np.array([cx, cy])
        c, m, tris, outer_idx = _hex_submesh(center, Rc, roles[(q, r)])
        coords_parts.append(c)
        mat_parts.append(m)
        elem_parts.append(tris + node_offset)
        node_hex.append(np.full(c.shape[0], len(hex_centers), dtype=np.int64))
        hex_outer_nodes.append([i + node_offset for i in outer_idx])
        hex_centers.append(center)
        hex_meta.append({"qr": (q, r), "mat": int(roles[(q, r)]),
                         "depth": depth_by_cell.get((q, r), 0.0)})
        node_offset += c.shape[0]

    coords = np.concatenate(coords_parts, axis=0)
    material_state = np.concatenate(mat_parts)
    node_hex = np.concatenate(node_hex)
    elements = np.concatenate(elem_parts, axis=0)
    hex_centers = np.array(hex_centers)

    # --- sodium gaps: explicit coolant nodes + Delaunay over hex-boundary nodes ---
    gap_nodes = []
    from scipy.spatial import cKDTree
    ctree = cKDTree(hex_centers)
    for i in range(len(hex_centers)):
        neigh = ctree.query_ball_point(hex_centers[i], hx.pitch_cm * 1.1)
        for j in neigh:
            if j > i:
                gap_nodes.append((hex_centers[i] + hex_centers[j]) / 2.0)
    gap_nodes = np.array(gap_nodes) if gap_nodes else np.zeros((0, 2))
    if gap_nodes.size:
        base = coords.shape[0]
        coords = np.concatenate([coords, gap_nodes], axis=0)
        material_state = np.concatenate(
            [material_state, np.full(gap_nodes.shape[0], MATERIAL_IDS["coolant"])])
        node_hex = np.concatenate([node_hex, np.full(gap_nodes.shape[0], -1)])

    # Delaunay over all outer-ring duct nodes + the gap coolant nodes
    outer_all = np.concatenate([np.array(h) for h in hex_outer_nodes])
    gap_ids = np.arange(base, coords.shape[0]) if gap_nodes.size else np.zeros(0, int)
    stitch_ids = np.unique(np.concatenate([outer_all, gap_ids]))
    if stitch_ids.size >= 3:
        dt = Delaunay(coords[stitch_ids])
        cent = coords[stitch_ids][dt.simplices].mean(axis=1)
        # keep triangles whose centroid is NOT inside any hexagon (i.e. in a gap)
        dists, nearest = ctree.query(cent)
        keep = []
        for t, c in enumerate(cent):
            if not _point_in_hex(c, hex_centers[nearest[t]], Rc):
                tri = stitch_ids[dt.simplices[t]]
                # reject long spurious triangles that jump across the core
                p = coords[tri]
                emax = max(np.hypot(*(p[0] - p[1])), np.hypot(*(p[1] - p[2])),
                           np.hypot(*(p[2] - p[0])))
                if emax < hx.pitch_cm:
                    keep.append(tri)
        if keep:
            elements = np.concatenate([elements, np.array(keep, dtype=np.int64)], axis=0)

    # drop degenerate triangles
    areas = triangle_areas(coords, elements)
    keepA = areas > 1e-9
    elements, areas = elements[keepA], areas[keepA]

    # --- per-node cross sections (control insertion + per-assembly burnup/temperature) ---
    hex_depth = np.array([m["depth"] for m in hex_meta] + [1.0])  # -1 (gap) -> unused
    nA = len(hex_meta) + 1
    fuel_ids = (MATERIAL_IDS["fuel_inner"], MATERIAL_IDS["fuel_outer"])
    # core-average control insertion (over CONTROL assemblies only): the spectrum
    # shift that inserted absorbers impose on every other material in the core. This
    # is what the rod-out / rod-in branch pair measures.
    core_rod_frac = float(depths.mean()) if len(depths) else 0.0

    # OpenMC branch table (the only source of cross sections): draw a physical burnup
    # [MWd/kgHM] and temperature [K] per assembly and look the constants up.
    # Depletion and Doppler are computed, not assumed.
    table = branch()          # raises MissingBranchTable if no OpenMC constants exist
    check_axis_coverage(table, "burnup", *hx.burnup_mwd_kg_range, label="hex")
    check_axis_coverage(table, "temperature", *hx.temperature_k_range, label="hex")
    burn_mwd = rng.uniform(*hx.burnup_mwd_kg_range, size=nA)
    temp_k = rng.uniform(*hx.temperature_k_range, size=nA)
    burn_mwd[-1] = 0.0                                # gap pseudo-assembly

    G = len(xs_for_id(0).D)
    ncol = n_xs_cols(G)
    cross_sections = np.zeros((coords.shape[0], ncol))
    for n in range(coords.shape[0]):
        h = node_hex[n]
        depth = float(hex_depth[h]) if h >= 0 else 1.0   # gap = coolant, depth ignored
        m = int(material_state[n])
        bu = float(burn_mwd[h]) if m in fuel_ids else 0.0
        cross_sections[n] = np.array(
            xs_for_id(m, insert_frac=depth, burnup_mwd_kg=bu,
                      temperature_k=float(temp_k[h]),
                      core_rod_frac=core_rod_frac).as_row())

    # --- boundary = outer hex edges with no neighbouring assembly ---
    # For each hexagon edge, the neighbour (if any) sits at 2*mid - center. If no
    # assembly is there, that edge is on the domain perimeter (Marshak vacuum BC).
    be = []
    for i in range(len(hex_centers)):
        onodes = hex_outer_nodes[i]                 # corners 0..5, order = _hex_corners
        for k in range(6):
            a, b = onodes[k], onodes[(k + 1) % 6]
            mid = 0.5 * (coords[a] + coords[b])
            guess = 2.0 * mid - hex_centers[i]      # where a neighbour center would be
            if ctree.query(guess)[0] > hx.pitch_cm * 0.3:   # no assembly -> perimeter
                be.append((int(a), int(b)))
    boundary_edges = np.asarray(be, dtype=np.int64) if be else np.zeros((0, 2), np.int64)
    boundary_mask = np.zeros(coords.shape[0], dtype=bool)
    if boundary_edges.size:
        boundary_mask[np.unique(boundary_edges)] = True

    nodal_volume = nodal_volumes(coords, elements, areas)
    rod_cells = np.where(np.isin(material_state,
                                 [MATERIAL_IDS["primary_control"],
                                  MATERIAL_IDS["secondary_control"]]))[0].astype(np.int64)

    counts = {ID_TO_MATERIAL[i]: int((material_state == i).sum())
              for i in range(len(MATERIAL_IDS))}
    assembly_metadata = {
        "geometry_model": "hex-lattice / P1-FEM (structured submesh + Delaunay gaps)",
        "n_nodes": int(coords.shape[0]),
        "n_elements": int(elements.shape[0]),
        "n_assemblies": len(hex_centers),
        "pitch_cm": hx.pitch_cm,
        "total_rings": total_rings,
        "reflector_rings": reflector_rings,
        "shield_rings": shield_rings,
        "enrichment_boundary_ring": enrichment_boundary,
        "n_primary_control": sum(1 for m in hex_meta if m["mat"] == MATERIAL_IDS["primary_control"]),
        "n_secondary_control": sum(1 for m in hex_meta if m["mat"] == MATERIAL_IDS["secondary_control"]),
        "control_insert_range": [float(hx.control_insert_fraction[0]),
                                 float(hx.control_insert_fraction[1])],
        "mean_control_depth": float(depths.mean()) if len(depths) else 0.0,
        "control_depths": [float(d) for d in depths],
        "n_control_inserted": int((depths > 0.0).sum()),
        "core_area_cm2": float(nodal_volume.sum()),
        "material_counts": counts,
    }

    return CoreGeometry(
        material_state=material_state,
        coordinates=coords,
        boundary_mask=boundary_mask,
        cross_sections=cross_sections,
        control_rod_cells=rod_cells,
        elements=elements,
        boundary_edges=boundary_edges,
        nodal_volume=nodal_volume,
        mesh=None,
        layout_name=layout_name,
        assembly_metadata=assembly_metadata,
    )
