# custom_frnn — USER-IMPLEMENTED (pure C++/CUDA)

This directory is **left for you to implement**. The PI-GNO build does not provide
the FRNN; it only defines the interface the rest of the pipeline depends on.

## What it must do

Fixed-Radius Nearest Neighbors over 2D cell-center coordinates, building the
**message graph** (the neural communication graph, distinct from the physics
stencil that defines `A`/`F`). Diagonal connectivity is expected in addition to
fixed-radius neighbors.

## Output contract (must match the Python fallback it replaces)

Produce a directed edge list:

- `edge_index`: int64, shape `[2, E]`, rows `[src; dst]`, **no self-loops**.
- Node ids are row-major grid indices: `node = iy * nx + ix`.
- Document whether the output is symmetric (both `i->j` and `j->i`) so edge
  features stay consistent.

## Wiring it in (one place)

1. Build this extension (e.g. `python setup.py build_ext --inplace`).
2. Edit `data generation/2D_fdm/frnn_interface.py`:
   - import your compiled module,
   - implement `frnn_query(...)` to call it and return `edge_index [2, E]` int64,
   - set `FRNN_AVAILABLE = True`.

Nothing else changes: `graph_build.py` already routes through `frnn_interface`
and only uses the Python fixed-radius fallback while `FRNN_AVAILABLE` is `False`.

## Reference for the fused-scatter extension

See `../message_passing/` for an example of the torch/pybind + CUDA + `setup.py`
layout you can mirror here.
