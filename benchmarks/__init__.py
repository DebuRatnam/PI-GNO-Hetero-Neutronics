"""Benchmark suite: PI-GNO vs FNO / DeepONet / MeshGraphNet.

Everything here shares ONE training loop, ONE normalization, ONE metric function
and ONE parameter budget, so differences between models are attributable to the
models. See the plan for the experiment matrix (E1..E5).
"""
