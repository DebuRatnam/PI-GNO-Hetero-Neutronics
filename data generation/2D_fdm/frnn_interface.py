"""Python-side interface contract for the user's custom FRNN (C++/CUDA).

The user implements FRNN (fixed-radius nearest neighbors) in csrc/custom_frnn as
a compiled extension. This module is the ONLY place data generation touches it,
so wiring it in later means: (1) build the extension, (2) import it here, (3) set
FRNN_AVAILABLE = True. Nothing else in the pipeline changes.

Expected extension surface (suggested; adapt the import to your build):

    import custom_frnn  # compiled torch/pybind extension
    edge_index = custom_frnn.query(points_xy, radius)   # -> LongTensor [2, E]

REQUIRED OUTPUT CONTRACT (must match the Python fallback in graph_build.py):
    - RADIUS graph: connect every pair of cell centers within `radius` (cm).
    - edge_index: int64 array/tensor of shape [2, E], directed src->dst.
    - No self-loops.
    - Symmetric closure expected (both directions), matching the KD-tree fallback.
    - Indices are row-major node ids (node = iy*nx + ix).
    - No hardcoded diagonal / 8-neighbor connectivity: edges come from the radius.
"""

from __future__ import annotations

import numpy as np


# Flip to True once the CUDA FRNN extension is importable and wired below.
FRNN_AVAILABLE = False


def frnn_query(coords: np.ndarray, radius: float, *, mesh=None) -> np.ndarray:
    """Adapter to the user's FRNN radius graph. Returns edge_index [2, E] (int64).

    Replace the body with a call into the compiled extension, e.g.:

        import torch, custom_frnn
        pts = torch.as_tensor(coords, dtype=torch.float32, device="cuda")
        ei = custom_frnn.query(pts, float(radius))   # [2, E] long, radius graph
        return ei.cpu().numpy().astype(np.int64)
    """
    raise NotImplementedError(
        "FRNN not wired in. Build csrc/custom_frnn, import it here, and set "
        "FRNN_AVAILABLE=True. Until then graph_build.py uses its KD-tree fallback."
    )
