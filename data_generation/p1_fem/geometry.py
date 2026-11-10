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
  - Control insertion toggles absorber vs sodium-follower cross sections; burnup
    and temperature are drawn per assembly and looked up on the OpenMC branch
    table (no ad-hoc XS perturbation).

Domain boundary (for the Marshak vacuum BC) is the set of outer hex edges with no
neighbouring assembly -- the true (non-convex) lattice perimeter, found per edge by
probing for a neighbour center across the edge midpoint.

One NODE = one mesh vertex. N varies per sample. Both groups are FAST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

from datagen_config import HexCoreConfig, robin_alpha_from_albedo
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
    cross_sections: np.ndarray   # [N, n_xs_cols(G)]; at G=2: (D1,D2,Sr1,Sr2,Ss12,nuSf1,nuSf2)
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
    # Optional PER-BOUNDARY-EDGE Marshak coefficient [B], alpha = (1-beta)/(2(1+beta))
    # for albedo beta. None -> PhysicsConfig.vacuum_robin_alpha (0.5, pure vacuum)
    # on every edge, which is the original behaviour. This is the axis of the
    # boundary-condition transfer study.
    boundary_alpha: Optional[np.ndarray] = None

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

def _hex_ring_points(center: np.ndarray, radius: float, k: int) -> np.ndarray:
    """The 6k points of hexagonal ring `k` at circumradius `radius`.

    Walks corner 0 -> corner 1 -> ... -> corner 5 and places k evenly spaced
    points along each side (the corner itself plus k-1 interior points), so ring
    k has 6k nodes and index c*k is corner c. That indexing is what makes the
    strip triangulation and the per-side boundary walk below index-arithmetic
    rather than a geometric search.
    """
    corners = _hex_corners(center[0], center[1], radius)
    pts = np.empty((6 * k, 2))
    for c in range(6):
        a, b = corners[c], corners[(c + 1) % 6]
        for j in range(k):
            pts[c * k + j] = a + (b - a) * (j / k)
    return pts


def _hex_submesh(center: np.ndarray, Rc: float, assembly_mat: int,
                 subdiv: int = 0):
    """Identical structured submesh for one hexagon.

    Returns (coords [n,2], materials [n], triangles [t,3] local, outer_idx,
    side_nodes) where `outer_idx` lists the outermost ring's local indices (used
    to stitch the sodium gaps) and `side_nodes` gives, for each of the 6 hexagon
    sides, the ordered local indices along that side including both corners.

    RESOLUTION. `subdiv` is the mesh-refinement level, and it is the ONLY thing
    that changes between the levels of the resolution-transfer study -- the
    physical geometry, the materials and every cross section are untouched.

      subdiv = 0  the original 13-node / 18-triangle mesh, reproduced exactly.
                  Kept as a distinct branch (rather than a special case of the
                  general scheme) so previously generated datasets regenerate
                  bit-for-bit.

      subdiv = s  concentric-ring refinement with R = 2s rings, ring k carrying
                  6k nodes: 1 + 3R(R+1) nodes and 6R^2 triangles per assembly.
                  s = 1, 2, 3 gives 19 / 61 / 127 nodes and 24 / 96 / 216
                  triangles.

    Rings 1..s fill the assembly region out to Ri and rings s+1..2s fill the duct
    annulus out to Rc, so the assembly/duct material interface lands EXACTLY on
    ring s at every level. That is why refinement is by concentric rings rather
    than by midpoint subdivision of the existing triangles: a midpoint between an
    assembly node and a duct node has no defensible material, and assigning it
    one would move a material interface as a side effect of refining the mesh.
    """
    Ri = Rc * (1.0 - DUCT_RING_FRAC)

    if subdiv <= 0:
        inner = _hex_corners(center[0], center[1], Ri)
        outer = _hex_corners(center[0], center[1], Rc)
        coords = np.concatenate([center[None, :], inner, outer], axis=0)  # [13,2]
        mats = np.array([assembly_mat] + [assembly_mat] * 6
                        + [MATERIAL_IDS["duct"]] * 6, dtype=np.int64)
        tris = []
        for k in range(6):
            a_in, b_in = 1 + k, 1 + (k + 1) % 6      # inner ring
            a_out, b_out = 7 + k, 7 + (k + 1) % 6     # outer ring
            tris.append([0, a_in, b_in])              # interior fan (assembly)
            tris.append([a_in, a_out, b_out])         # duct annulus
            tris.append([a_in, b_out, b_in])
        outer_idx = list(range(7, 13))
        side_nodes = [[7 + c, 7 + (c + 1) % 6] for c in range(6)]
        return coords, mats, np.array(tris, dtype=np.int64), outer_idx, side_nodes

    s = int(subdiv)
    R = 2 * s
    radius = lambda k: (Ri * k / s) if k <= s else (Ri + (Rc - Ri) * (k - s) / s)

    coords = [center[None, :]]
    mats = [np.array([assembly_mat], dtype=np.int64)]
    ring_off = [0]                       # local index of ring k's first node
    for k in range(1, R + 1):
        coords.append(_hex_ring_points(center, radius(k), k))
        mats.append(np.full(6 * k, assembly_mat if k <= s
                            else MATERIAL_IDS["duct"], dtype=np.int64))
        ring_off.append(ring_off[-1] + (1 if k == 1 else 6 * (k - 1)))
    coords = np.concatenate(coords, axis=0)
    mats = np.concatenate(mats)

    def node(k: int, i: int) -> int:
        """local index of node i (mod 6k) on ring k; ring 0 is the centre."""
        return 0 if k == 0 else ring_off[k] + (i % (6 * k))

    tris = []
    for c in range(6):                                   # ring 1: fan from centre
        tris.append([0, node(1, c), node(1, c + 1)])
    for k in range(2, R + 1):                            # strip between k-1 and k
        for c in range(6):
            A = [node(k - 1, c * (k - 1) + j) for j in range(k)]      # k points
            B = [node(k, c * k + j) for j in range(k + 1)]            # k+1 points
            for j in range(k):
                tris.append([A[j], B[j], B[j + 1]])
            for j in range(k - 1):
                tris.append([A[j], B[j + 1], A[j + 1]])

    outer_idx = [node(R, i) for i in range(6 * R)]
    side_nodes = [[node(R, c * R + j) for j in range(R + 1)] for c in range(6)]
    return coords, mats, np.array(tris, dtype=np.int64), outer_idx, side_nodes


def _subdiv_resolver(spec):
    """Turn a hex_subdiv spec into a callable (q, r, ring_distance) -> level.

    Accepted forms:
      int                       uniform refinement, every assembly at that level
      (s_inner, s_outer, d)     NON-UNIFORM: assemblies within ring distance d get
                                s_inner, the rest get s_outer
      callable                  arbitrary, called as f(q, r, d)

    The non-uniform form exists for the discretization probe. Uniform refinement
    changes node density everywhere at once, which a message-passing model can
    partly absorb by rescaling; a MIXED mesh makes density vary WITHIN a single
    graph, which is where an unweighted scatter-add and a volume-weighted
    quadrature genuinely diverge -- the aggregate magnitude of the former tracks
    local node count, and the core no longer has one local node count.
    """
    if callable(spec):
        return spec
    if isinstance(spec, (tuple, list)):
        if len(spec) != 3:
            raise ValueError(
                "non-uniform hex_subdiv must be (s_inner, s_outer, ring_boundary), "
                f"got {spec!r}")
        s_in, s_out, d_bnd = int(spec[0]), int(spec[1]), int(spec[2])
        return lambda q, r, d: (s_in if d <= d_bnd else s_out)
    s = int(spec)
    return lambda q, r, d: s


# --- core assembly -----------------------------------------------------------

def make_core(hx: HexCoreConfig, *, layout_name: str = "default",
              reflector_rings: Optional[int] = None,
              shield_rings: Optional[int] = None,
              enrichment_boundary: Optional[int] = None,
              insert_fraction: Optional[float] = None,
              hex_subdiv: Optional[int] = None,
              boundary_albedo: Optional[float] = None,
              rng: Optional[np.random.Generator] = None) -> CoreGeometry:
    """Build one hex-lattice core. Ring counts / enrichment boundary / control
    insertion can be overridden per sample for dataset variability.

    `hex_subdiv` is the MESH RESOLUTION level (see _hex_submesh). It changes only
    the discretization: hold the rng and every other argument fixed and you get
    the same physical core at a different mesh density, which is what the
    resolution-transfer study requires. Defaults to HexCoreConfig.hex_subdiv."""
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
    subdiv_spec = getattr(hx, "hex_subdiv", 0) if hex_subdiv is None else hex_subdiv
    subdiv_of = _subdiv_resolver(subdiv_spec)
    coords_parts, mat_parts, elem_parts = [], [], []
    node_hex, node_offset = [], 0
    hex_centers, hex_outer_nodes, hex_meta = [], [], []
    hex_side_nodes = []
    subdiv_used = []
    for (q, r) in sorted(roles.keys()):
        cx, cy = _axial_to_xy(q, r, hx.pitch_cm)
        center = np.array([cx, cy])
        d_ring = (abs(q) + abs(r) + abs(q + r)) // 2
        subdiv = subdiv_of(q, r, d_ring)
        subdiv_used.append(subdiv)
        c, m, tris, outer_idx, side_nodes = _hex_submesh(
            center, Rc, roles[(q, r)], subdiv)
        coords_parts.append(c)
        mat_parts.append(m)
        elem_parts.append(tris + node_offset)
        node_hex.append(np.full(c.shape[0], len(hex_centers), dtype=np.int64))
        hex_outer_nodes.append([i + node_offset for i in outer_idx])
        hex_side_nodes.append([[i + node_offset for i in side]
                               for side in side_nodes])
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
    # Under refinement a hexagon side carries several sub-edges, so the
    # neighbour probe is done once PER SIDE using the side's two corners. It
    # cannot be done per sub-edge: reflecting the centre through a sub-edge
    # midpoint does not land on the neighbouring assembly centre, only
    # reflecting through the FULL side midpoint does.
    be = []
    for i in range(len(hex_centers)):
        for side in hex_side_nodes[i]:
            a0, a1 = side[0], side[-1]              # the side's two corners
            mid = 0.5 * (coords[a0] + coords[a1])
            guess = 2.0 * mid - hex_centers[i]      # where a neighbour center would be
            if ctree.query(guess)[0] > hx.pitch_cm * 0.3:   # no assembly -> perimeter
                for a, b in zip(side[:-1], side[1:]):
                    be.append((int(a), int(b)))
    boundary_edges = np.asarray(be, dtype=np.int64) if be else np.zeros((0, 2), np.int64)
    boundary_mask = np.zeros(coords.shape[0], dtype=bool)
    if boundary_edges.size:
        boundary_mask[np.unique(boundary_edges)] = True

    # Per-edge Marshak coefficient. None -> pure vacuum, left as None so
    # assemble_AF takes the original scalar path and existing data is unchanged.
    boundary_alpha = None
    if boundary_albedo is not None and boundary_edges.size:
        boundary_alpha = np.full(boundary_edges.shape[0],
                                 float(robin_alpha_from_albedo(boundary_albedo)))

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
        # mesh refinement level: the resolution-study axis. Recorded per sample so
        # a level can never be inferred from N alone (N also moves with
        # reflector_rings). For a non-uniform mesh the spec is recorded along with
        # the distinct levels actually used, since a single number cannot
        # describe it.
        "hex_subdiv": (list(subdiv_spec) if isinstance(subdiv_spec, (tuple, list))
                       else subdiv_spec),
        "hex_subdiv_levels": sorted(set(int(s) for s in subdiv_used)),
        "hex_subdiv_uniform": len(set(subdiv_used)) == 1,
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
        "boundary_albedo": (None if boundary_albedo is None
                            else float(boundary_albedo)),
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
        boundary_alpha=boundary_alpha,
    )
