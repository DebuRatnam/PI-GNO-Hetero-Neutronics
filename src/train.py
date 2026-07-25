"""Training loop scaffold for PI-GNO. NOT auto-run (requires GPU + a generated
dataset). Provided so the full pipeline is wired end-to-end and ready to launch:

    python train.py --data ../datasets/run01

Steps: load train split -> fit normalization on TRAIN ONLY -> build model ->
optimize L = L_flux + λk L_k + λpde L_PDE + λbc L_BC -> log metrics. Normalization,
loss weights, tolerances, seeds, and hyperparameters are all logged (CLAUDE).
"""

from __future__ import annotations

import argparse
import json

import torch

from dataclasses import replace

from config import DEFAULT, ModelConfig
from dataio import load_torch_sample, list_split
from dataset import load_sample
from features import fit_normalization, NodeLayout
from model import PIGNO
from losses import compute_loss
from metrics import sample_metrics
from scatter import extension_available


def fit_norm_on_train(train_paths, device, node_passthrough=None):
    nf, ef, fl, ks = [], [], [], []
    for p in train_paths:
        s = load_torch_sample(p, device="cpu")
        nf.append(s.node_feats); ef.append(s.edge_feats)
        fl.append(s.flux); ks.append(float(s.k_eff))
    return fit_normalization(torch.cat(nf), torch.cat(ef), torch.cat(fl), ks,
                             node_passthrough=node_passthrough)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULT.train.epochs)
    args = ap.parse_args()

    cfg = DEFAULT
    torch.manual_seed(cfg.train.seed)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    use_cuda_scatter = cfg.train.use_cuda_scatter and device == "cuda"
    print(f"device={device}  cuda_scatter_ext={extension_available()}  "
          f"use_cuda_scatter={use_cuda_scatter}")

    train_paths = list_split(args.data, "train")
    val_paths = list_split(args.data, "val")
    assert train_paths, f"no train samples in {args.data}/train"

    # schema-driven model dims: read the dataset's material/group counts from
    # metadata so the SAME script trains on Natrium (hex) or KP-FHR (pebble) data.
    meta = load_sample(train_paths[0])["geometry_metadata"]
    layout = NodeLayout.from_metadata(meta)
    model_cfg = ModelConfig.from_metadata(
        meta, latent_dim=cfg.model.latent_dim, n_mp_layers=cfg.model.n_mp_layers,
        message_hidden=cfg.model.message_hidden, norm=cfg.model.norm,
        k_pool=cfg.model.k_pool, activation=cfg.model.activation)
    cfg = replace(cfg, model=model_cfg)
    print(f"reactor={meta.get('reactor_type','hex')}  node_in_dim={model_cfg.node_in_dim}  "
          f"n_groups={model_cfg.n_groups}  n_materials={model_cfg.n_materials}")

    norm = fit_norm_on_train(train_paths, device,
                             node_passthrough=layout.passthrough_cols)
    energy_pf = 3.2e-11  # keep consistent with data_generation/config.py
    model = PIGNO(cfg.model, energy_pf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    print("config:", json.dumps(cfg.to_dict()))
    print("normalization:", json.dumps(norm.to_dict())[:200], "...")

    for epoch in range(args.epochs):
        model.train()
        running = {}
        for p in train_paths:
            s = load_torch_sample(p, device=device)
            out = model(
                node_feats_norm=norm.node.transform(s.node_feats),
                edge_feats_norm=norm.edge.transform(s.edge_feats),
                raw_node_feats=s.node_feats,
                edge_index=s.edge_index,
                flux_scaler=norm.flux, k_mean=norm.k_mean, k_std=norm.k_std,
                use_cuda_scatter=use_cuda_scatter,
            )
            terms = compute_loss(
                flux_hat_norm=out.flux_norm,
                flux_ref_norm=norm.flux.transform(s.flux),
                k_hat_norm=out.k_norm,
                k_ref_norm=(s.k_eff - norm.k_mean) / norm.k_std,
                flux_hat_phys=out.flux_phys, k_hat_phys=out.k_phys,
                A=s.A, F=s.F, boundary_mask=s.boundary_mask, cfg=cfg.loss,
            )
            opt.zero_grad()
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            for k, v in terms.item_dict().items():
                running[k] = running.get(k, 0.0) + v
        if epoch % 10 == 0:
            avg = {k: v / len(train_paths) for k, v in running.items()}
            print(f"epoch {epoch:4d}  " + "  ".join(f"{k}={v:.3e}" for k, v in avg.items()))

    # validation report
    model.eval()
    with torch.no_grad():
        agg = {}
        for p in val_paths:
            s = load_torch_sample(p, device=device)
            out = model(
                node_feats_norm=norm.node.transform(s.node_feats),
                edge_feats_norm=norm.edge.transform(s.edge_feats),
                raw_node_feats=s.node_feats, edge_index=s.edge_index,
                flux_scaler=norm.flux, k_mean=norm.k_mean, k_std=norm.k_std,
                use_cuda_scatter=use_cuda_scatter,
            )
            for k, v in sample_metrics(
                    out, s, material_onehot_cols=layout.material_onehot_cols).items():
                agg[k] = agg.get(k, 0.0) + v
        print("VAL:", {k: v / max(len(val_paths), 1) for k, v in agg.items()})


if __name__ == "__main__":
    main()
