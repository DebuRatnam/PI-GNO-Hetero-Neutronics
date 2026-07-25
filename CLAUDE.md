# PI-GNO: Heterogeneous Neutronics Implementation Guide

## Scope

Build a Physics-Informed Graph Neural Operator (PI-GNO) for a 2D **multigroup neutron-diffusion eigenvalue problem**, discretized with **P1 finite elements**. Group count is **configurable G** (`PhysicsConfig.n_groups`; default G=2, byte-compatible with the original two-group data). The generator supports **two reactor types** (`reactor_type`), both flowing through the same solver / graph / model:

- **`hex`** — a **Natrium-inspired sodium fast reactor**: a hexagonal-duct lattice (identical structured submesh per assembly + Delaunay-stitched sodium gaps). Material set (8, one-hot): fuel_inner, fuel_outer, primary_control, secondary_control, reflector, shield, **duct** (HT9 wall + sodium gap, homogenized), coolant. **Fast** spectrum (g1 high-fast, g2 slow-fast).
- **`fhr`** — a **Kairos KP-FHR pebble bed** (annular, gFHR/NRC-faithful): a central graphite reflector column, a fueled pebble annulus (RSA-packed **fuel + graphite moderator pebbles**) in FLiBe, an outer graphite reflector, and a steel vessel. **4 B4C control elements sit in the OUTER REFLECTOR** (NRC ML21272A383: KP-FHR control inserts into the side graphite reflector — NOT the bed); **3 X-shaped shutdown elements insert into the inner bed**; both are rigid shapes in graphite-lined channels. Material set (7, one-hot): fuel_pebble, graphite_pebble, control_element, shutdown_element, reflector, coolant (FLiBe), vessel. **Thermal** spectrum (g1 fast, g2 thermal, ~0.625 eV cut). Each pebble = one homogenized node; recirculation/burnup + fuel:moderator ratio enter as dataset variability (frozen 2D states). Exact Hermes control-element diameter is proprietary → `r_ctrl` is a documented representative value (gFHR 5.2 cm rod proxy).

The XS row layout, node-feature width, and model dims are **schema-driven** from each sample's metadata (`n_groups`, `n_materials`, `node_feature_order`) — never hardcode them. Never say "thermal" for the `hex`/Natrium reactor (fast spectrum); the `fhr` reactor is genuinely thermal and named accordingly.

Given a core configuration and local nuclear data, predict:

- node-wise group fluxes `phi_1(x, y)` and `phi_2(x, y)`;
- graph-wise effective multiplication factor `k_eff`; and
- node-wise power density, computed from the predicted fluxes.

Do not silently change the physics model, group ordering, mesh convention, or boundary-condition convention. Record any such choice in dataset metadata.

## Dataset Contract

One sample represents one meshed core state and must include:

```text
material_state     [N]       material id per mesh node (8-way)
coordinates        [N, 2]    physical node coordinates
cross_sections     [N, C]    local multigroup nuclear data
elements           [T, 3]    P1 triangles -> node ids (FEM physics graph)
boundary_edges     [B, 2]    perimeter mesh edges (Marshak vacuum BC)
nodal_volume       [N]       lumped nodal volume (tiles the core area)
edge_index         [2, E]    directed message-passing edges
edge_features      [E, 8]    distance, dx, dy, interface_flag, harmonic_D1,
                             dD1, dSigma_r1, dSigma_s12
A                  sparse    loss/removal/leakage operator [GN, GN]
F                  sparse    fission-production operator   [GN, GN]
boundary_mask      [N]       Boolean boundary-node mask
k_eff              scalar    reference eigenvalue
flux               [N, G]    reference group fluxes (group-major)
power_density      [N]       reference derived field
geometry_metadata             reactor_type, n_groups, n_materials, layout,
                             control/shutdown insertion, node/edge feature order
```

The node feature order is **schema-driven** (recorded in `geometry_metadata["node_feature_order"]`): `[x, y] + one-hot material (n_materials) + XS block (n_xs_cols(G)) + boundary_flag`. Material is ONE-HOT encoded, not an ordinal id, to avoid a spurious ordering between materials. The XS block order is `D(G), Sigma_r(G), down-scatter(G(G-1)/2), nuSigma_f(G)` (down-scatter-only). For the Natrium `hex` reactor at G=2 this is exactly the original 18-dim layout:

```text
[x, y,
 mat_fuel_inner, mat_fuel_outer, mat_primary_control, mat_secondary_control,
 mat_reflector, mat_shield, mat_duct, mat_coolant,
 D1, D2, Sigma_r1, Sigma_r2, Sigma_s12,
 nuSigma_f1, nuSigma_f2, boundary_flag]
```

The KP-FHR `fhr` reactor at G=2 is 17-dim (7 materials). Read the widths from metadata; never hardcode. `material_state` is stored separately as integer ids; only the node feature vector uses the one-hot block. N varies per sample.

`chi` is fixed nuclear data: incorporate it in `F`, not in node features. Preserve physical units and document normalization factors. Normalize only with statistics fitted on the training split; apply exactly the same transform to validation/test data and retain inverse transforms for reporting.

## Data Generation and Validation with the P1-FEM Hex-Core Solver

For every configuration: build the hex lattice + per-assembly submesh; assign cross sections (control insertion toggles absorber vs sodium-follower XS); assemble sparse `A` and `F` with P1 finite elements (stiffness + lumped mass + Marshak Robin vacuum); solve

```text
A * phi = (1 / k_eff) * F * phi
```

with a validated sparse eigensolver (SciPy `splu` power iteration); store fluxes and `k_eff`; compute power density; then create graph tensors.

Keep the **physics graph** separate from the **message graph**:

- The physics graph is the FEM triangulation (`elements`). It defines `A`, `F`, and the PDE residual.
- The message graph is a **kNN graph** on the mesh nodes (`knn_k` neighbors, fixed degree → clean batching / GPU utilization). It supports neural communication and may be richer than the FEM adjacency. There is no FRNN/radius path.

Before training, verify shapes, finite values, symmetry/consistency properties appropriate to the discretization, nonempty boundary masks, sparse-matrix dimensions `[GN, GN]` (group-major), and a small residual for every reference solution. Never reconstruct `A` or `F` from a lossy neural graph if the assembled operators are available.

## Model Architecture

Use separate lifting networks for node features and edge features. Map both into a latent space, then apply stacked kernel-integration/message-passing layers with residual updates:

```text
h_i <- h_i + sum_j message(h_i, h_j, edge_ij)
```

Messages must depend on node states and the edge/interface physics: distance, interface diffusion coefficient, material transition, and available geometric information. Apply LayerNorm or GraphNorm for stability. Avoid using only material IDs when cross sections are available.

Use projection heads for:

- `flux_hat [N, 2]`, with a documented group ordering;
- `k_hat`, formed from global mean, sum, or attention pooling followed by an MLP; and
- `power_hat`, computed from `flux_hat` using the same documented physical relation used for labels (not an unrelated unconstrained head unless explicitly studied).

## Physics-Informed Objective

Train with

```text
L = L_flux + lambda_k L_k + lambda_PDE L_PDE + lambda_BC L_BC
R = A phi_hat - (1 / k_hat) F phi_hat
L_PDE = mean(R^2)
L_BC = mean(phi_hat[boundary_mask]^2)
```

`L_flux` is flux MSE and `L_k` is `k_eff` MSE. The initial benchmark uses vacuum boundaries. Compute PDE residuals with the physics operators and compatible flux-vector ordering; do not use the message graph as a substitute. Tune and log all loss weights, normalization choices, solver tolerances, split seeds, and model hyperparameters.

## Reporting Minimums

Report flux error per energy group, `k_eff` error, power-density error, PDE residual, boundary residual, and performance by material/region. Keep train/validation/test geometry configurations disjoint enough to measure generalization, especially across control-rod layouts and reflector placements.
