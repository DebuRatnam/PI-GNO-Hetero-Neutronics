# Discretization probe (30 paired configs)

Invariance is measured against mesh **L3**; lower is more discretization-invariant.

`reference` is the FEM solution's OWN mesh-dependence -- the number a model must beat to be more discretization-invariant than the discretization it was trained on.

| mesh | reference g1 | model g1 | reference g2 | model g2 |
|---|---|---|---|---|
| L0 | 0.0963 | nan | 0.0584 | nan |
| L1 | 0.0655 | nan | 0.0667 | nan |
| Lmixed3_1_3 | 0.0863 | nan | 0.0570 | nan |
| L2 | 0.0158 | nan | 0.0172 | nan |

## Accuracy vs truth mesh (L6)

`reference` here is the DISCRETIZATION FLOOR: how far that mesh's own FEM solution sits from the fine truth. A model cannot be faulted for error below its floor.

| mesh | floor g1 | model g1 | model k err (pcm) |
|---|---|---|---|
| L0 | 0.1131 | nan | - |
| L1 | 0.0815 | nan | - |
| Lmixed3_1_3 | 0.0977 | nan | - |
| L2 | 0.0315 | nan | - |
| L3 | 0.0155 | nan | - |
