"""sys.path wiring for the benchmark suite.

src/ modules import each other flat (`from lifting import mlp`, `from physics
import ...`), and src/dataio.py in turn injects data_generation/p1_fem for the
npz loader. Import this module first from anything under benchmarks/ so those
flat imports resolve without turning src/ into a package and touching every
existing import statement.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(_ROOT, "src")
DATAGEN = os.path.join(_ROOT, "data_generation", "p1_fem")
BENCH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = _ROOT

# BENCH is included so `models.pigno_wrap` resolves whether harness.py is run as
# a script (sys.path[0] is already benchmarks/) or imported from elsewhere.
for _p in (SRC, DATAGEN, BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)
