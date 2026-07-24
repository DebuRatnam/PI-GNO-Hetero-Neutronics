"""Physics, mesh, and units configuration for the 2D two-group neutron-diffusion
data generator (P1 finite elements on an irregular Delaunay mesh).

Everything that defines the discretization and physical conventions lives here so
it can be serialized into each sample's `geometry_metadata`. Do not hard-code these
values elsewhere; import from this module.

Conventions (recorded in metadata, never changed silently):
  - Group ordering: BOTH groups are FAST (sodium fast reactor). index 0 =
    high-fast (group 1, ~0.8-10 MeV), index 1 = slow-fast (group 2,
    ~1 keV-0.8 MeV). Both sit in the fast spectrum (no moderated lower group).
  - Mesh: IRREGULAR. An adaptive point cloud (dense near material interfaces,
    coarse in bulk) is triangulated with a Delaunay triangulation; the PDE is
    discretized with linear (P1) finite elements. One NODE = one mesh vertex; node
    count N varies per sample. Node ordering is the point-cloud ordering (see
    geometry.py), NOT a row-major grid index.
  - Global DOF ordering for A, F, flux vectors: group-major block layout
    [ all group-1 nodes (N) , all group-2 nodes (N) ]  -> length 2N.
  - chi (fission spectrum) is fixed nuclear data and lives in F, not node features.
  - Boundary condition: vacuum, implemented as a Marshak partial-current Robin
    term on boundary edges (see operators.py / VACUUM_ROBIN_ALPHA).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple


# Fission spectrum across the two FAST groups. Watt/fission spectrum peaks ~2 MeV
# -> most neutrons born in the high-energy fast group (g1), a few percent below
# the g1/g2 cut land in the slow-fast group (g2). NOT a moderated/slowed split.
CHI = (0.95, 0.05)

# Legacy FDM vacuum extrapolation factor (Milne). UNUSED by the FEM assembly; kept
# for provenance / metadata continuity only.
VACUUM_EXTRAP_FACTOR = 0.7104

# FEM vacuum BC: Marshak partial-current Robin term added to the stiffness diagonal
# on each boundary edge, +alpha * L/2 per endpoint (L = boundary edge length).
# alpha = 0.5 is the standard Marshak coefficient (outgoing partial current J = phi/2).
# Boundary flux is intentionally NONZERO -> the zero-flux L_BC stays off in training
# (src lambda_bc = 0). Stored in metadata.
VACUUM_ROBIN_ALPHA = 0.5

# Power normalization: power_density = kappa * (kappa_f1*Sigma_f1*phi1 + kappa_f2*Sigma_f2*phi2)
# We fold kappa (energy/fission) and use nuSigma_f as a proxy when Sigma_f is not
# tracked separately; documented in power.py. Stored in metadata.
ENERGY_PER_FISSION_J = 3.2e-11  # ~200 MeV in joules


@dataclass(frozen=True)
class MeshConfig:
    # Defines the physical DOMAIN box the point sampler fills. With the FEM mesh the
    # grid itself is no longer solved on: width = nx*dx, height = ny*dy set the
    # extent; dx/dy are a reference length scale. Actual node count is variable and
    # comes from the sampled cloud (see MeshingConfig / geometry.py).
    nx: int = 32              # domain columns (extent only)
    ny: int = 32              # domain rows (extent only)
    dx: float = 5.0           # reference length [cm]
    dy: float = 5.0           # reference length [cm]

    @property
    def width(self) -> float:
        return self.nx * self.dx

    @property
    def height(self) -> float:
        return self.ny * self.dy

    @property
    def n_nodes(self) -> int:
        # NOMINAL only (kept for compatibility). The FEM node count varies per
        # sample; use geom.n_nodes, not this.
        return self.nx * self.ny


@dataclass(frozen=True)
class PhysicsConfig:
    # n_groups is AUTHORITATIVE: operators/solver/power assemble G = n_groups blocks
    # and expect a per-node XS width of n_xs_cols(G) (see xs_common). chi is the
    # length-G fission spectrum (down-scatter-only multigroup; defaults to the
    # two fast groups). For a multigroup run set BOTH n_groups and chi together.
    n_groups: int = 2
    chi: Tuple[float, ...] = CHI
    vacuum_extrap_factor: float = VACUUM_EXTRAP_FACTOR  # legacy FDM; unused by FEM
    vacuum_robin_alpha: float = VACUUM_ROBIN_ALPHA      # FEM Marshak Robin coeff
    energy_per_fission_j: float = ENERGY_PER_FISSION_J


@dataclass(frozen=True)
class SolverConfig:
    # Power iteration on A^{-1} F (dominant k_eff mode).
    max_outer: int = 5000
    tol_k: float = 1e-9          # relative k convergence
    tol_flux: float = 1e-8       # relative flux L2 convergence
    residual_tol: float = 1e-6   # accept solution if ||A phi - (1/k) F phi|| / ||F phi|| < tol


@dataclass(frozen=True)
class GraphConfig:
    # Message graph (NEURAL). Kept separate from the P1-FEM physics graph that
    # defines A/F. Default connectivity is a kNN graph on the mesh nodes: every
    # node connects to its `knn_k` nearest neighbors (symmetric closure). Fixed
    # degree gives clean batching + good GPU utilization on variable-N samples and
    # avoids the ragged-degree / radius-tuning sensitivity of a radius graph on the
    # structured hex mesh (dense interior vs. thin sodium gaps).
    graph_method: str = "knn"         # "knn" (default) or "radius"
    knn_k: int = 10                   # neighbors per node for the kNN message graph
    # Radius graph (legacy / FRNN path, used only when graph_method="radius").
    # Must exceed in-hex node spacing (~pitch/(2*hex_subdiv)) so nodes connect
    # across the thin sodium gap, yet stay small enough that degree stays bounded.
    frnn_radius: float = 7.0          # ~ 1.4 * in-hex node spacing (subdiv=2, pitch 18.7)
    edge_feature_dim: int = 8         # see edge_feature_order in metadata()
    include_diagonals: bool = True    # DEPRECATED: unused; kept for config back-compat


@dataclass(frozen=True)
class HexCoreConfig:
    """Natrium-inspired HEXAGONAL-duct core. All lengths in cm.

    The core is a hex lattice (rings 0..R). Each assembly is a hexagon with a fixed
    structured triangulation (identical per assembly -> spatial invariance); the
    outer triangle ring is homogenized HT9 duct + sodium gap; the thin inter-assembly
    gaps are Delaunay-stitched sodium. Radial roles: inner fuel -> outer fuel ->
    reflector rings -> shield rings, with 9 primary + 4 secondary control positions
    replacing selected fuel assemblies.
    """
    pitch_cm: float = 18.7            # assembly center-to-center
    duct_wall_cm: float = 0.37        # HT9 wall thickness (provenance/homogenization)
    gap_cm: float = 0.40              # inter-assembly sodium gap
    hex_subdiv: int = 2               # triangulation refinement per hex (1->6, 2->24 tris)

    # radial ring layout (from the center outward). Default = REDUCED dev core
    # (fuel_rings=4 -> R=6 -> 127 assemblies). Full Natrium-like: fuel_rings~7,
    # reflector_rings=2, shield_rings=2.
    fuel_rings: int = 4
    reflector_rings: int = 1
    shield_rings: int = 1
    enrichment_boundary_ring: int = 2   # rings <= this = inner enrichment, else outer

    # reactivity control (positions chosen deterministically from these counts)
    n_primary_control: int = 9
    n_secondary_control: int = 4
    control_insert_fraction: Tuple[float, float] = (0.0, 1.0)  # frac of rods inserted

    # cross-section perturbation (burnup/temperature proxy), fractional +/- range
    xs_perturb: float = 0.05


@dataclass(frozen=True)
class PebbleCoreConfig:
    """Kairos KP-FHR PEBBLE-BED core (thermal spectrum). All lengths in cm.

    ANNULAR core (gFHR-accurate), radial build out from the center:
        0          .. R_center_refl : central graphite reflector column (solid)
        R_center   .. R_fuel_in     : inner unfueled pebble zone (graphite pebbles)
        R_fuel_in  .. R_fuel_out    : FUELED pebble annulus (fuel + moderator pebbles)
        R_fuel_out .. R_bed         : outer unfueled pebble zone (graphite pebbles)
        R_bed      .. R_refl        : outer graphite reflector (houses control channels)
        R_refl     .. R_vessel      : steel vessel ring
    Each pebble center = one node (4 cm dia, r_peb=2.0). FLiBe fills the bed gaps.
    Control elements sit in the OUTER REFLECTOR (NRC: control inserts into the side
    graphite reflector); shutdown elements insert into the inner fueled bed (NRC).
    Both are rigid shapes in graphite-lined channels. Defaults = REDUCED dev core;
    scale the radii up for the full thousands-of-pebbles core. See
    geometry_pebble.make_pebble_core.
    """
    # annular radii (out from center)
    R_center_refl: float = 12.0       # central graphite reflector column radius
    R_fuel_in: float = 20.0           # inner edge of the fueled pebble annulus
    R_fuel_out: float = 56.0          # outer edge of the fueled pebble annulus
    R_bed: float = 62.0               # outer edge of the pebble bed (unfueled band)
    R_refl: float = 78.0              # outer graphite reflector outer radius
    R_vessel: float = 80.5            # structural vessel outer radius (thin steel shell)
    r_peb: float = 2.0                # pebble radius (4 cm dia, gFHR; node half-spacing)
    packing_fraction: float = 0.50    # target 2D RSA packing fraction (jamming ~0.55)
    # real pebble beds are ~60% pebbles / 40% FLiBe by volume (random sphere packing
    # ~0.60). Interstitial FLiBe node count = coolant_per_pebble * n_pebbles; 1.0 is
    # EMPIRICALLY CALIBRATED to ~60% pebble / 40% coolant nodal-volume in the bed.
    coolant_per_pebble: float = 1.0
    graphite_pebble_frac: float = 0.15  # fraction of moderator-only pebbles in the
                                        # fueled annulus (NRC: sets C/HM ratio)

    # control elements: rigid cylinders in the OUTER graphite reflector.
    # NOTE: exact Hermes control-element diameter is proprietary; r_ctrl is a
    # documented REPRESENTATIVE value anchored to the public gFHR rod (5.2 cm dia).
    r_ctrl: float = 2.6               # control-element radius (5.2 cm dia, gFHR proxy)
    n_control: int = 4                # 4 control elements (NRC: in the reflector)
    control_offset: float = 7.0       # control-ring radius = R_bed + this (into reflector)
    n_circle_nodes: int = 16          # rigid-circle boundary nodes per control cylinder
    channel_wall: float = 1.6         # graphite channel-lining thickness around elements

    # shutdown elements: rigid X-shapes in the inner fueled bed (NRC).
    n_shutdown: int = 3               # 3 X-shaped shutdown elements
    shutdown_ring_frac: float = 0.42  # shutdown-ring radius / R_fuel_out (inner bed)
    x_arm_len: float = 7.0            # X arm half-length
    x_arm_w: float = 3.0              # X bar width

    coolant_step_frac: float = 1.5    # interstitial coolant grid step / r_peb
    burnup_perturb: float = 0.08      # fractional +/- nuSf perturbation (burnup proxy)
    # per-sample variability drawn per split (see make_split_plans_fhr); ranges here
    # bound the fraction of control / shutdown inserted and the moderator-pebble
    # fraction (fuel:moderator ratio is a real KP-FHR reactivity lever).
    control_insert_fraction: Tuple[float, float] = (0.0, 1.0)
    shutdown_insert_fraction: Tuple[float, float] = (0.0, 1.0)
    graphite_pebble_fraction_range: Tuple[float, float] = (0.08, 0.22)


@dataclass(frozen=True)
class SamplingConfig:
    # Dataset scale. Defaults are the large-study TARGET; generation is never
    # auto-run, and counts are overridable on the generate.py CLI. Nothing here
    # forces 7000 solves -- it only sizes the split plans when you choose to run.
    train_samples: int = 5000
    val_samples: int = 1000
    test_samples: int = 1000


@dataclass(frozen=True)
class DataGenConfig:
    # reactor_type selects the geometry + material set: "hex" (Natrium hex-duct,
    # fast spectrum) or "fhr" (KP-FHR pebble bed, thermal spectrum). Physics/solver/
    # graph are shared; only the geometry builder + material module differ.
    reactor_type: str = "hex"
    mesh: MeshConfig = field(default_factory=MeshConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    hexcore: HexCoreConfig = field(default_factory=HexCoreConfig)
    pebblecore: PebbleCoreConfig = field(default_factory=PebbleCoreConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    units: Dict[str, str] = field(default_factory=lambda: {
        "length": "cm",
        "D": "cm",
        "Sigma": "1/cm",
        "nuSigma_f": "1/cm",
        "flux": "n/cm^2/s (relative, eigenvector-normalized)",
        "power_density": "W/cm^3 (relative)",
    })

    def metadata(self) -> dict:
        """Flat dict embedded into every sample for provenance.

        Reactor-aware: the material set, node-feature order, and group structure are
        derived from `reactor_type` + `physics.n_groups`, so both Natrium (hex, fast)
        and KP-FHR (pebble, thermal) samples record a self-describing schema.
        """
        from xs_common import xs_col_names   # local import avoids any load-order cycle
        G = self.physics.n_groups
        if self.reactor_type == "fhr":
            from materials_fhr import MATERIAL_ORDER
            core_cfg, core_key = asdict(self.pebblecore), "pebblecore"
            discretization = ("P1 finite elements on a KP-FHR pebble-bed core: "
                              "RSA-packed pebble nodes + FLiBe coolant, Delaunay "
                              "triangulated (N varies per sample)")
            spectrum = ("thermal (Kairos KP-FHR / graphite + FLiBe moderated); "
                        "g1 fast, g2 thermal (~0.625 eV boundary)")
            group_ordering = "0=fast, 1=thermal (thermal-spectrum pebble bed)"
            geometry_model = ("KP-FHR pebble bed: RSA-packed fuel/graphite pebbles "
                              "(1 node/pebble), FLiBe coolant, 4 outer B4C control "
                              "cylinders + 3 inner B4C X shutdown elements, graphite "
                              "reflector + steel vessel")
            node_index_order = "pebbles first, then coolant/structure/reflector/vessel fill; see elements[T,3]"
        else:
            from materials import MATERIAL_ORDER
            core_cfg, core_key = asdict(self.hexcore), "hexcore"
            discretization = ("P1 finite elements on a hexagonal-duct core: structured "
                              "triangulation per assembly + Delaunay-stitched sodium gaps "
                              "(N varies per sample)")
            spectrum = "fast (sodium fast reactor / Natrium-inspired); high-fast + slow-fast groups"
            group_ordering = "0=high-fast(g1), 1=slow-fast(g2); both groups fast spectrum"
            geometry_model = ("Natrium-inspired hexagonal-duct lattice: identical structured "
                              "submesh per assembly, homogenized HT9 duct+gap ring, "
                              "9 primary + 4 secondary control positions")
            node_index_order = "hex-mesh order (per-assembly submeshes then gaps); see elements[T,3]"

        node_feature_order = (["x", "y"]
                              + [f"mat_{m}" for m in MATERIAL_ORDER]
                              + xs_col_names(G)
                              + ["boundary_flag"])

        return {
            "reactor_type": self.reactor_type,
            "n_groups": G,
            "n_materials": len(MATERIAL_ORDER),
            "mesh": asdict(self.mesh),
            "physics": asdict(self.physics),
            "solver": asdict(self.solver),
            "graph": asdict(self.graph),
            core_key: core_cfg,
            "sampling": asdict(self.sampling),
            "units": self.units,
            "discretization": discretization,
            "dof_ordering": "group-major [g0(N), ..., g{G-1}(N)] -> GN",
            "node_index_order": node_index_order,
            "group_ordering": group_ordering,
            "spectrum": spectrum,
            "boundary_condition": "vacuum (Marshak partial-current Robin term on boundary edges, alpha=0.5)",
            "material_encoding": "one-hot [" + ", ".join(MATERIAL_ORDER) + "]",
            "graph_construction": (f"{self.graph.graph_method} message graph on mesh "
                                   "nodes; no hardcoded neighbors"),
            "geometry_model": geometry_model,
            # Canonical edge feature order (8 dims). Keep in sync with
            # graph_build.build_message_graph and src/config.ModelConfig.edge_in_dim.
            "edge_feature_order": [
                "distance", "dx", "dy", "interface_flag",
                "harmonic_D1", "dD1", "dSigma_r1", "dSigma_s12",
            ],
            # Canonical node feature order: [x,y] + one-hot material + G-group XS +
            # boundary_flag. Derived from reactor_type + n_groups (self-describing).
            "node_feature_order": node_feature_order,
        }


DEFAULT = DataGenConfig()
