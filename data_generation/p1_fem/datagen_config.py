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
    # Transverse (axial) leakage buckling Bz^2 [1/cm^2] for the 2D radial model:
    # removal gains D_g*Bz^2, approximating axial leakage a 2D slice otherwise ignores
    # (without it k_eff is biased high). 0 = pure 2D. Set per-reactor in dataset.py
    # from the material module's AXIAL_BUCKLING_CM2 (fast core leaks more per unit
    # height than the taller pebble bed).
    axial_buckling: float = 0.0
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
    # defines A/F. Connectivity is a kNN graph on the mesh nodes: every node connects
    # to its `knn_k` nearest neighbors (symmetric closure). Fixed degree gives clean
    # batching + good GPU utilization on variable-N samples and avoids the
    # ragged-degree / radius-tuning sensitivity of a radius graph on the structured
    # hex mesh (dense interior vs. thin sodium gaps).
    knn_k: int = 10                   # neighbors per node for the kNN message graph
    edge_feature_dim: int = 8         # see edge_feature_order in metadata()


@dataclass(frozen=True)
class HexCoreConfig:
    """Natrium-inspired HEXAGONAL-duct core. All lengths in cm.

    The core is a hex lattice (rings 0..R). Each assembly is a hexagon with a fixed
    structured triangulation (identical per assembly -> spatial invariance); the
    outer triangle ring is homogenized HT9 duct + sodium gap; the thin inter-assembly
    gaps are Delaunay-stitched sodium. Radial roles: inner fuel -> outer fuel ->
    reflector rings -> shield rings, with 9 primary + 4 secondary control positions
    replacing selected fuel assemblies.

    WHAT IS ACTUALLY SOURCED. TerraPower's Natrium construction permit application
    (Kemmerer Unit 1, 2024) and the associated NRC safety evaluations are public, but
    the core map is not: the public record fixes only that Natrium is an 840 MWth
    pool-type SFR; that Type 1 fuel is U-10wt%Zr metal, sodium-bonded, HT9-clad, with
    enrichment varying by core position and a PEAK below 20 wt% U-235; that reactivity
    control is NINE primary + FOUR secondary control assemblies using B4C absorber
    (NRC ML24220A154 / ML24103A212, Fuel and Control Assembly topical report SE); and
    that five assembly types are modelled (fuel, primary control, secondary control,
    reflector, shield), all hexagonal ducts with an inlet nozzle, load pads and a
    handling socket (ML24088A085, Core Design and Thermal Hydraulic Technical Report).

    Everything dimensional below -- pitch, duct wall, gap, ring counts, the pin
    lattices in openmc_models -- is REPRESENTATIVE of a HALEU metal-fuel SFR of this
    class, not a vendor number, because those values are not in the public docket.
    This is why the reactor is labelled "Natrium-inspired" everywhere and why
    xs_provenance in every sample carries the transport/data-library trail instead.
    """
    pitch_cm: float = 18.7            # assembly center-to-center (representative)
    duct_wall_cm: float = 0.37        # HT9 wall thickness (provenance/homogenization)
    gap_cm: float = 0.40              # inter-assembly sodium gap
    hex_subdiv: int = 2               # triangulation refinement per hex (1->6, 2->24 tris)

    # radial ring layout (from the center outward). Default = REDUCED dev core
    # (fuel_rings=4 -> R=6 -> 127 assemblies). Full Natrium-like: fuel_rings~7,
    # reflector_rings=2, shield_rings=2.
    fuel_rings: int = 4
    reflector_rings: int = 1
    shield_rings: int = 1
    # rings <= this = INNER enrichment zone, else OUTER. SFR radial zoning puts the
    # HIGHER enrichment in the OUTER zone: the periphery leaks hard (long fast
    # mean free path, steep flux gradient), so the outer zone needs more fissile to
    # flatten the radial power profile. See materials.LIBRARY / openmc_models.
    enrichment_boundary_ring: int = 2

    # Reactivity control: 9 primary + 4 secondary control assemblies, B4C absorber.
    # This IS a docketed Natrium number (NRC Fuel and Control Assembly TR SE).
    n_primary_control: int = 9
    n_secondary_control: int = 4
    control_insert_fraction: Tuple[float, float] = (0.0, 1.0)  # frac of rods inserted

    # Per-assembly burnup + temperature spread.
    #
    # PRIMARY PATH (OpenMC branch table present): a burnup [MWd/kgHM] and a
    # temperature [K] are drawn per assembly from these ranges and looked up in the
    # branch table, so depletion and Doppler are computed quantities.
    #
    # These ranges MUST stay inside the tabulated branch points, or the interpolator
    # clamps and every state past the last point silently collapses onto it. The
    # burnup ceiling tracks the depletion table (depletion_natrium.json reaches
    # 60 MWd/kgHM); raise both together, never this alone.
    # xs_branch.check_axis_coverage warns at generation time if they drift apart.
    #
    # The floor is 2, not 0, on purpose: Xe-135 equilibrates in ~3 days (~0.14
    # MWd/kgHM here) and Sm-149 in ~1-2 months (~1.4), so a truly fresh endpoint is
    # the only state in the whole range with no saturating fission-product poison.
    # Interpolating from it smears that step across the entire axis. An operating
    # core has no xenon-free fuel anyway. Fast spectrum, so the effect is small here
    # -- Xe-135's 2.6 Mbarn absorption is a thermal resonance -- but the FHR floor
    # exists for the same reason and there it matters.
    burnup_mwd_kg_range: Tuple[float, float] = (2.0, 60.0)
    # Natrium runs sodium at roughly 360 C in / 510 C out (633 / 783 K) with metal
    # fuel a few hundred K above the local coolant, so the Doppler axis spans core
    # inlet to peak fuel rather than sitting entirely above the outlet temperature.
    temperature_k_range: Tuple[float, float] = (630.0, 1000.0)
    #
    # FALLBACK PATH (no table): the original ad-hoc perturbation. xs_perturb is a
    # small SYMMETRIC +/- temperature/density wiggle on nuSf; burnup is DIRECTIONAL
    # (fuel only): fissile depletion lowers nuSf by up to burnup_max, and
    # fission-product poison raises removal by burnup_poison_coeff*burnup. Kept so
    # the generator still runs without OpenMC -- NOT publication-grade.
    xs_perturb: float = 0.03          # symmetric +/- temperature/density wiggle (nuSf)
    burnup_max: float = 0.12          # max burnup fraction (directional depletion)
    burnup_poison_coeff: float = 0.30 # removal rise per unit burnup (fission products)


@dataclass(frozen=True)
class PebbleCoreConfig:
    """Kairos KP-FHR PEBBLE-BED core (thermal spectrum). All lengths in cm.

    Dimensions are the PUBLISHED gFHR benchmark -- Kairos Power's own non-proprietary
    KP-FHR surrogate (Satvat et al., Nucl. Eng. Des. 384 (2021) 111461; INL Virtual
    Test Bed "Description of the generic FHR"; Duchnowski et al. 2023 Table 1). That
    is the only KP-FHR-family core with an open dimensional specification, so it is
    what the geometry is anchored to. Radial build out from the center:

        0          .. R_bed     = 120  : pebble bed, SOLID cylinder
        R_bed      .. R_refl    = 180  : graphite side reflector (60 cm thick),
                                         holds the control-element channels
        R_refl     .. R_barrel  = 182  : SS316H core barrel (2 cm)
        R_barrel   .. R_downcmr = 187  : FLiBe downcomer (5 cm)
        R_downcmr  .. R_vessel  = 191  : SS316H reactor vessel (4 cm)

    NOT ANNULAR: the gFHR/KP-FHR bed is a full-diameter cylinder with no central
    reflector column (pebbles are buoyant and float up through the whole bed). The
    R_center_refl / R_fuel_in / R_fuel_out knobs are kept so an inner column or an
    unfueled radial band CAN be modelled, but they default to "off" (0 / full bed).

    Active height is 309.47 cm and enters this 2D radial model only through
    materials_fhr.AXIAL_BUCKLING_CM2, never as an in-plane dimension.

    Barrel + downcomer + vessel are carried as ONE homogenized `vessel` FEM ring;
    openmc_models tallies the three separately and homogenizes them by area.

    Reactivity control follows HERMES AS LICENSED (NRC KP-FHR Core Design and
    Analysis Methodology topical report KP-TR-024-NP Rev 0, ML24095A258, April 2024
    -- earlier revisions ML21272A383 / ML23195A130; element counts from the Hermes
    PSAR): the reactivity CONTROL system inserts 4 control elements into engineered
    channels in the side graphite reflector, and the reactivity SHUTDOWN system
    inserts 3 shutdown elements DIRECTLY into the packed bed. The gFHR surrogate instead carries 10 reflector rods and no shutdown
    elements, so the element COUNTS here are Hermes' while the element GEOMETRY
    (2.6 cm radius B4C, 7.9 cm from bed edge to rod centre) is the published gFHR rod.
    Each pebble center = one node (4 cm dia, r_peb=2.0). FLiBe fills the bed gaps.

    For fast iteration scale the radii down together (a Hermes-sized 2 m^3 core is
    roughly R_bed ~ 60 cm); the solver/graph code is size-agnostic.
    """
    # radial build (out from center); gFHR benchmark values
    R_center_refl: float = 0.0        # central graphite column (gFHR/KP-FHR: none)
    R_fuel_in: float = 0.0            # inner edge of the FUELED zone (0 = from center)
    R_fuel_out: float = 120.0         # outer edge of the FUELED zone (= R_bed)
    R_bed: float = 120.0              # pebble bed radius (gFHR: 1.2 m)
    R_refl: float = 180.0             # graphite side reflector outer radius (60 cm thick)
    R_barrel: float = 182.0           # SS316H core barrel outer radius (2 cm)
    R_downcomer: float = 187.0        # FLiBe downcomer outer radius (5 cm)
    R_vessel: float = 191.0           # SS316H reactor vessel outer radius (4 cm)
    r_peb: float = 2.0                # pebble radius (4 cm dia, gFHR; node half-spacing)
    packing_fraction: float = 0.50    # target 2D RSA packing fraction (jamming ~0.55)
    # real pebble beds are ~60% pebbles / 40% FLiBe by volume (random sphere packing
    # ~0.60). Interstitial FLiBe node count = coolant_per_pebble * n_pebbles; 1.0 is
    # EMPIRICALLY CALIBRATED to ~60% pebble / 40% coolant nodal-volume in the bed.
    coolant_per_pebble: float = 1.0
    # Moderator-only ("graphite") pebbles are a real KP-FHR feature: NRC Hermes
    # docket material states that a portion of the pebbles in the core are moderator
    # pebbles, and that they contribute moderation alongside the fuel-pebble graphite,
    # the FLiBe and the reflector blocks. The FRACTION is not public (the gFHR
    # surrogate publishes a single equilibrium pebble type), so this is a documented
    # representative value, swept over graphite_pebble_fraction_range per sample.
    graphite_pebble_frac: float = 0.15

    # control elements: rigid cylinders in the OUTER graphite reflector.
    # NOTE: exact Hermes control-element diameter is proprietary; r_ctrl and the
    # radial offset are the PUBLISHED gFHR rod (2.6 cm radius; 7.9 cm from the bed
    # edge to the rod centre) used as the documented representative geometry.
    r_ctrl: float = 2.6               # control-element radius (gFHR rod, 5.2 cm dia)
    # 10 equally-spaced reflector rods, matching the gFHR reactivity-control system
    # whose bed radius this core uses. Hermes as licensed has FOUR control elements,
    # but in a ~2 m^3 core roughly a sixteenth of this bed's cross section -- putting
    # 4 rods around a 120 cm bed would make the bank nearly worthless. If you rescale
    # the radii to Hermes, set n_control=4 with it.
    n_control: int = 10
    control_offset: float = 7.9       # control-ring radius = R_bed + this (gFHR: 7.9 cm)
    n_circle_nodes: int = 16          # rigid-circle boundary nodes per control cylinder
    channel_wall: float = 1.6         # graphite channel-lining thickness around elements

    # shutdown elements: rigid X-shapes inserted DIRECTLY into the packed bed (NRC
    # KP-TR-024-NP: the RSS inserts shutdown elements directly into the pebble bed,
    # while the RCS inserts control elements into side-reflector channels).
    n_shutdown: int = 3               # 3 X-shaped shutdown elements (Hermes)
    shutdown_ring_frac: float = 0.42  # shutdown-ring radius / R_fuel_out (inner bed)
    # Shutdown-element geometry is NOT public (gFHR carries no shutdown elements at
    # all). These are sized to keep the blade footprint a fixed ~12% of the bed
    # radius, so the bank stays a meaningful reactivity lever at the gFHR bed size
    # instead of a token absorber; scale them with R_bed if you rescale the core.
    x_arm_len: float = 15.0           # X arm half-length
    x_arm_w: float = 6.0              # X bar width

    coolant_step_frac: float = 1.5    # interstitial coolant grid step / r_peb
    # Grid step for the SOLID structure rings, as a multiple of r_peb. The 60 cm
    # graphite reflector has no sub-pebble features, so it is meshed coarser than the
    # bed; the thin barrel/downcomer/vessel rings keep the fine (coolant) step so each
    # 2-5 cm layer is still resolved.
    structure_step_frac: float = 3.0
    # Per-pebble burnup + temperature spread. In a pebble bed this is real physical
    # variability: recirculation means pebbles of every burnup coexist at every
    # radius, which is exactly the state this dataset is meant to cover.
    #
    # PRIMARY PATH (OpenMC branch table present): burnup [MWd/kgHM] and temperature
    # [K] are drawn per pebble from these ranges and looked up in the branch table,
    # so depletion (including Xe/Sm poisoning) and Doppler are computed.
    # Floor is 2, not 0: past Xe-135 (~0.09 MWd/kgHM) and Sm-149 (~1.8) equilibrium.
    # This is a THERMAL core, so those poisons are first-order -- a fresh endpoint
    # would put a saturating step change at one end of a linearly interpolated axis.
    # Ceiling tracks the published gFHR discharge burnup: 17.6% FIMA average / ~20%
    # FIMA peak, i.e. ~170 / ~190 MWd/kgHM at ~9.6 MWd/kgHM per % FIMA. With 8-pass
    # recirculation every burnup between the floor and the discharge limit coexists in
    # the bed at once, which is exactly the spread this axis samples.
    burnup_mwd_kg_range: Tuple[float, float] = (2.0, 190.0)
    # gFHR coolant runs 823.15 K in / 923 K out; fuel and graphite sit above the local
    # salt temperature, so the axis spans core inlet to peak fuel.
    temperature_k_range: Tuple[float, float] = (823.15, 1100.0)
    #
    # SPATIAL CORRELATION of the burnup field, in [0, 1]. 0 (default) = burnup drawn
    # i.i.d. per pebble, uncorrelated in space. >0 blends in a linear radial ramp, so
    # a fraction `w` of the burnup range is explained by radius (bed centre -> edge)
    # and (1-w) stays random.
    #
    # The licensed methodology (KP-TR-024-NP) does not do either: it runs DEM pebble
    # flow through ZONER to build spectral zones, so burnup is correlated along pebble
    # FLOW PATHS. Most of that correlation is axial and a 2D radial slice cannot see
    # it. The radial part is real -- pebbles near the reflector wall flow slower, so
    # they reside longer and burn deeper -- but its magnitude is proprietary. Hence
    # opt-in rather than a default: turning it on is a documented modelling choice,
    # and the value used is recorded in each sample's modelling_assumptions.
    burnup_radial_weight: float = 0.0
    #
    # FALLBACK PATH (no table): the original ad-hoc perturbation. burnup_perturb is
    # the max burnup FRACTION: fissile depletion lowers nuSf, and fission-product
    # poison (Xe/Sm) raises the THERMAL removal Sr2 by burnup_poison_coeff*burnup.
    # temp_perturb is a symmetric +/- wiggle on nuSf. NOT publication-grade.
    burnup_perturb: float = 0.10      # max burnup fraction (directional depletion)
    burnup_poison_coeff: float = 0.60 # thermal-removal rise per unit burnup (poison)
    temp_perturb: float = 0.03        # symmetric +/- temperature/density wiggle (nuSf)
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
            from materials_fhr import MATERIAL_ORDER, branch_metadata
            core_cfg, core_key = asdict(self.pebblecore), "pebblecore"
            discretization = ("P1 finite elements on a KP-FHR pebble-bed core: "
                              "RSA-packed pebble nodes + FLiBe coolant, Delaunay "
                              "triangulated (N varies per sample)")
            spectrum = ("thermal (Kairos KP-FHR / graphite + FLiBe moderated); "
                        "g1 fast, g2 thermal (~0.625 eV boundary)")
            group_ordering = "0=fast, 1=thermal (thermal-spectrum pebble bed)"
            geometry_model = ("KP-FHR pebble bed at published gFHR dimensions: "
                              "cylindrical (non-annular) RSA-packed fuel/graphite "
                              "pebble bed R=120 cm (1 node/pebble) in FLiBe, 60 cm "
                              "graphite side reflector holding 4 B4C control "
                              "cylinders (Hermes/NRC), 3 B4C X shutdown elements "
                              "inserted directly into the bed, SS316H barrel + FLiBe "
                              "downcomer + SS316H vessel as one homogenized ring")
            node_index_order = "pebbles first, then coolant/structure/reflector/vessel fill; see elements[T,3]"
        else:
            from materials import MATERIAL_ORDER, branch_metadata
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

        # Where the cross sections came from. This is the provenance a reviewer needs
        # to judge the labels: transport code + version, evaluated nuclear data
        # library, weighting spectrum, branch grid, per-branch k_eff, Monte Carlo
        # uncertainty. Absent table -> say so plainly rather than implying rigor.
        xs_prov = branch_metadata() or {
            "source": "hand-tuned committed library (no OpenMC branch table present)",
            "weighting": "none -- representative order-of-magnitude constants",
            "warning": ("these constants are NOT traceable to an evaluated nuclear "
                        "data library; run xs_openmc.py before publishing results"),
        }

        return {
            "xs_provenance": xs_prov,
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
            "graph_construction": "kNN message graph on mesh nodes; no hardcoded neighbors",
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
