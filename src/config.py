"""Model and training configuration for PI-GNO.

All loss weights, normalization choices, solver tolerances, seeds, and
hyperparameters are logged from here (CLAUDE reporting requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    # node_in_dim, n_groups, n_materials are SCHEMA-DRIVEN: set them from a dataset's
    # geometry_metadata via ModelConfig.from_metadata so the same model handles
    # Natrium hex (8 materials, G=2 -> 18 dims) and KP-FHR pebble (7 materials, G=2
    # -> 17 dims) and any group count. Node COUNT N is variable per sample; the model
    # is N-agnostic (per-graph pooling + scatter). Defaults = Natrium hex.
    node_in_dim: int = 18      # x,y + one-hot material + G-group XS + boundary flag
    edge_in_dim: int = 8       # [distance, dx, dy, interface_flag, harmonic_D1,
                               #  dD1, dSigma_r1, dSigma_s12] (message-graph edges)
    n_materials: int = 8       # one-hot material block width (8 hex / 7 fhr)
    latent_dim: int = 64
    n_mp_layers: int = 6
    message_hidden: int = 128
    norm: str = "layernorm"    # "layernorm" | "graphnorm" | "none"
    n_groups: int = 2
    k_pool: str = "mean"       # "mean" | "sum" | "attention"
    activation: str = "silu"   # message-passing nonlinearity (required: SiLU)
    # Message aggregation: "sum" is unweighted scatter-add (a GNN aggregation);
    # "volume" weights each message by the SOURCE node's nodal_volume and
    # normalizes, making it a quadrature estimate of the kernel integral -- the
    # GNO form. "volume_raw" is the unnormalized Nystrom sum, kept for the
    # ablation that shows why normalization is needed under a fixed-k graph.
    # Default stays "sum" so existing behaviour is unchanged unless asked for.
    aggregation: str = "sum"   # "sum" | "volume" | "volume_raw"

    @classmethod
    def from_metadata(cls, meta: dict, **overrides) -> "ModelConfig":
        """Build a ModelConfig whose dims match a generated dataset. Reads
        n_groups + n_materials from geometry_metadata and derives node_in_dim."""
        from features import NodeLayout
        layout = NodeLayout.from_metadata(meta)
        base = dict(node_in_dim=layout.total_dim, n_materials=layout.n_materials,
                    n_groups=layout.n_groups)
        base.update(overrides)
        return cls(**base)


@dataclass
class LossConfig:
    lambda_k: float = 1.0
    lambda_pde: float = 0.1
    # lambda_bc DEFAULT 0.0: the data generator imposes vacuum as extrapolated-
    # length leakage baked into A (operators._diffusion_block), so boundary
    # CELL-CENTER flux is deliberately NONZERO and the vacuum BC is already
    # enforced through L_PDE. The zero-flux L_BC = mean(phi[boundary]^2) term
    # would penalize the reference solution, so it is OFF by default. Enable it
    # (lambda_bc > 0) ONLY with a true Dirichlet-zero boundary discretization.
    lambda_bc: float = 0.0
    # PDE residual computed in PHYSICAL (un-normalized) flux units; see losses.py.


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 500
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"       # falls back to cpu in train.py if unavailable
    use_cuda_scatter: bool = True   # use csrc kernel if compiled, else Python ref


@dataclass
class PIGNOConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "loss": asdict(self.loss),
                "train": asdict(self.train)}


DEFAULT = PIGNOConfig()
