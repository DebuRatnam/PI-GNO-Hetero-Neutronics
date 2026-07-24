"""Scatter-add aggregation for message passing.

This is the ONE primitive we push to a custom CUDA kernel (csrc/message_passing):
aggregating per-edge messages into destination nodes is memory-bound and runs
every forward/backward of every layer, so a fused scatter-add with a hand-written
backward avoids materializing large intermediate gradients.

This module exposes `scatter_add_messages(messages, dst, n_nodes)`:
  - If the compiled extension `pigno_mp` is importable and CUDA is used, it calls
    the fused kernel via a custom autograd Function.
  - Otherwise it falls back to a pure-PyTorch index_add (correct, differentiable,
    runs on CPU). The model is fully functional without compiling anything.

The message MLP itself stays in PyTorch (cuBLAS); only the scatter is custom.
"""

from __future__ import annotations

import os
import sys

import torch

# Make the in-place-built extension importable regardless of cwd: the build
# (`python setup.py build_ext --inplace`) drops pigno_mp*.so in
# csrc/message_passing, which is NOT on sys.path when training is launched from
# src/. Add it explicitly so a compiled kernel is actually used (otherwise every
# run silently falls back to pure PyTorch). `pip install -e csrc/message_passing`
# also works and makes this path injection redundant.
_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "csrc", "message_passing")
)
if os.path.isdir(_EXT_DIR) and _EXT_DIR not in sys.path:
    sys.path.append(_EXT_DIR)

try:
    import pigno_mp  # compiled CUDA extension from csrc/message_passing
    _HAS_EXT = True
except Exception:
    pigno_mp = None
    _HAS_EXT = False


class _ScatterAddCUDA(torch.autograd.Function):
    """Forward: out[dst[e]] += messages[e]. Backward: grad_messages = grad_out[dst]."""

    @staticmethod
    def forward(ctx, messages: torch.Tensor, dst: torch.Tensor, n_nodes: int):
        ctx.save_for_backward(dst)
        ctx.n_nodes = n_nodes
        return pigno_mp.scatter_add_forward(messages, dst, n_nodes)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (dst,) = ctx.saved_tensors
        grad_messages = pigno_mp.scatter_add_backward(grad_out, dst)
        return grad_messages, None, None


def scatter_add_messages(messages: torch.Tensor, dst: torch.Tensor,
                         n_nodes: int, *, use_cuda_scatter: bool = True) -> torch.Tensor:
    """messages [E, F], dst [E] (long) -> aggregated [n_nodes, F]."""
    if use_cuda_scatter and _HAS_EXT and messages.is_cuda:
        return _ScatterAddCUDA.apply(messages, dst, n_nodes)
    # pure-PyTorch fallback (CPU or no extension): fully differentiable
    out = messages.new_zeros((n_nodes, messages.shape[1]))
    out.index_add_(0, dst, messages)
    return out


def extension_available() -> bool:
    return _HAS_EXT
