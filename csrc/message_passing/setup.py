"""Build the PI-GNO message-passing CUDA extension `pigno_mp`.

    cd csrc/message_passing
    python setup.py build_ext --inplace      # or: pip install -e .

After building, src/scatter.py auto-imports `pigno_mp` and routes the scatter-add
through the CUDA kernel on GPU tensors. If the build is skipped, the model still
runs via the pure-PyTorch fallback in src/scatter.py.

NOTE: not run here (no GPU). Requires a CUDA toolkit matching the installed
PyTorch build.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pigno_mp",
    ext_modules=[
        CUDAExtension(
            name="pigno_mp",
            sources=["scatter.cpp", "scatter_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
