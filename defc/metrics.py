"""
metrics.py — Anisotropic distance metric for DEFC.

The core idea: not all feature dimensions are equally relevant for
separating classes. We weight each dimension by its discriminative
power (Fisher ratio), so the field is stronger along axes that
actually separate the classes.

This is used inside the Yukawa kernel to compute d_W(x, c_j).
"""

import torch
from torch import Tensor


def fisher_weights(
    X: Tensor,
    y: Tensor,
    alpha: float = 1e-6,
) -> Tensor:
    """
    Compute per-feature anisotropy weights via Fisher criterion.

    For each feature m:
        v_m = |mu_m^+ - mu_m^-| / (sigma_m^+ + sigma_m^- + alpha)

    High v_m  → feature strongly discriminates classes → large metric weight.
    Low  v_m  → feature is noise-like               → small metric weight.

    Args:
        X:     (N, d) feature matrix (already normalised).
        y:     (N,)  binary labels in {-1, +1}.
        alpha: small constant to avoid division by zero.

    Returns:
        v: (d,) non-negative weight vector, NOT normalised to sum=1
           (absolute scale is absorbed by gamma in the kernel).
    """
    pos_mask = (y == 1)
    neg_mask = (y == -1)

    X_pos = X[pos_mask]   # (N+, d)
    X_neg = X[neg_mask]   # (N-, d)

    mu_pos = X_pos.mean(dim=0)    # (d,)
    mu_neg = X_neg.mean(dim=0)    # (d,)
    std_pos = X_pos.std(dim=0)    # (d,)
    std_neg = X_neg.std(dim=0)    # (d,)

    v = (mu_pos - mu_neg).abs() / (std_pos + std_neg + alpha)

    return v  # (d,)


def anisotropic_distance(
    X: Tensor,
    C: Tensor,
    v: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute weighted Euclidean distances between all (x, c) pairs.

        d_W(x, c) = sqrt( sum_m  v_m * (x_m - c_m)^2 )

    Vectorised over the full (N, K) cross-product — no Python loops.

    Args:
        X:   (N, d) data points in current (possibly deformed) space.
        C:   (K, d) prototype positions.
        v:   (d,)  anisotropy weights from fisher_weights().
        eps: floor added before sqrt to avoid NaN gradients at r=0.

    Returns:
        D: (N, K) pairwise distance matrix.
    """
    # (N, 1, d) - (1, K, d)  →  (N, K, d)
    diff = X.unsqueeze(1) - C.unsqueeze(0)

    # Weighted squared distances: (N, K)
    D_sq = (diff ** 2 * v.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    return torch.sqrt(D_sq + eps)   # (N, K)