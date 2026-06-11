"""
prototypes.py — Prototype initialisation for each cascade layer.

Prototypes are NOT just class representatives. They are field sources,
and their placement directly shapes the deformation dynamics:

  C_in       (k-means centers)   — stabilise the "mass" of each class.
                                   Pulled from the interior, updated
                                   every T layers.

  C_boundary (cross-class kNN)   — mark the conflict zone where classes
                                   meet. Re-computed at every layer
                                   because the boundary shifts after
                                   each deformation step.

Together they give the field two distinct regimes:
  - global gravitational pull (interior)
  - sharp local pressure at decision boundaries

Reference: DEFC §3.
"""

import torch
from torch import Tensor
from sklearn.cluster import MiniBatchKMeans
import numpy as np


# ---------------------------------------------------------------------------
# Interior prototypes — k-means
# ---------------------------------------------------------------------------

def kmeans_prototypes(
    X: Tensor,
    y: Tensor,
    n_prototypes: int,
    random_state: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Place interior prototypes at k-means centers inside each class.

    We use sklearn's MiniBatchKMeans for speed; the result is converted
    back to a torch Tensor on the same device as X.

    Args:
        X:             (N, d) data in current (possibly deformed) space.
        y:             (N,)   binary labels {-1, +1}.
        n_prototypes:  number of centers *per class*.
        random_state:  reproducibility seed.

    Returns:
        C:  (2*n_prototypes, d)  prototype positions.
        Q:  (2*n_prototypes,)    prototype charges (+1 or -1, sign only).
    """
    device = X.device
    X_np   = X.detach().cpu().numpy()
    y_np   = y.detach().cpu().numpy()

    centers_list  = []
    charges_list  = []

    for label in [-1, 1]:
        mask   = (y_np == label)
        X_cls  = X_np[mask]

        k = min(n_prototypes, len(X_cls))
        km = MiniBatchKMeans(
            n_clusters    = k,
            random_state  = random_state,
            n_init        = "auto",
        ).fit(X_cls)

        centers_list.append(km.cluster_centers_)
        charges_list.append(np.full(k, label, dtype=np.float32))

    C = torch.tensor(
        np.concatenate(centers_list, axis=0),
        dtype=torch.float32, device=device,
    )
    Q = torch.tensor(
        np.concatenate(charges_list, axis=0),
        dtype=torch.float32, device=device,
    )
    return C, Q


# ---------------------------------------------------------------------------
# Boundary prototypes — cross-class kNN
# ---------------------------------------------------------------------------

def boundary_prototypes(
    X: Tensor,
    y: Tensor,
    n_prototypes: int,
    k_neighbors: int = 5,
) -> tuple[Tensor, Tensor]:
    """
    Select boundary prototypes: points from each class whose nearest
    neighbors include points from the *opposite* class.

    Concretely, for each point x in class c we compute its k nearest
    neighbors across the full dataset. If any of those k neighbors
    belongs to class -c, x is a boundary candidate.

    From all boundary candidates we keep the n_prototypes points with
    the *smallest* distance to the opposite class (highest conflict).

    Args:
        X:             (N, d) current data positions.
        y:             (N,)   labels {-1, +1}.
        n_prototypes:  max number of boundary prototypes *per class*.
        k_neighbors:   neighbourhood size for conflict detection.

    Returns:
        C:  (M, d)  boundary prototype positions (M ≤ 2*n_prototypes).
        Q:  (M,)    prototype charges (sign = class of the point).
    """
    device = X.device
    N      = X.shape[0]

    # pairwise Euclidean distance matrix  (N, N)
    # For large N this should be replaced by approximate kNN (FAISS/HNSW).
    with torch.no_grad():
        # chunked to avoid OOM on moderate N
        D = _pairwise_euclidean(X)   # (N, N)

    # set self-distance to inf so a point never picks itself
    D.fill_diagonal_(float("inf"))

    # indices of k nearest neighbors for every point  (N, k)
    _, knn_idx = torch.topk(D, k=k_neighbors, dim=1, largest=False)

    centers_list = []
    charges_list = []

    for label in [-1, 1]:
        own_mask  = (y == label)                              # (N,)
        own_idx   = own_mask.nonzero(as_tuple=True)[0]        # indices of this class

        # for each point in this class, check if any kNN is from opposite class
        knn_labels = y[knn_idx[own_idx]]                      # (N_cls, k)
        has_opp    = (knn_labels != label).any(dim=1)         # (N_cls,)

        candidate_idx = own_idx[has_opp]                      # conflict zone

        if len(candidate_idx) == 0:
            continue  # classes already well separated

        # rank by minimum distance to opposite-class neighbor
        opp_mask     = (y != label)
        D_to_opp     = D[candidate_idx][:, opp_mask]          # (n_cand, N_opp)
        min_d_to_opp = D_to_opp.mean(dim=1)            # (n_cand,)

        # keep the n_prototypes points closest to the boundary
        k_keep = min(n_prototypes, len(candidate_idx))
        _, top_k = torch.topk(min_d_to_opp, k=k_keep, largest=False)

        selected = candidate_idx[top_k]
        centers_list.append(X[selected])
        charges_list.append(y[selected].float())

    if not centers_list:
        # fallback: return empty tensors (fully separable data)
        return (
            torch.empty(0, X.shape[1], device=device),
            torch.empty(0, device=device),
        )

    C = torch.cat(centers_list, dim=0)   # (M, d)
    Q = torch.cat(charges_list, dim=0)   # (M,)
    return C, Q


# ---------------------------------------------------------------------------
# Combined initialisation
# ---------------------------------------------------------------------------

def build_prototypes(
    X: Tensor,
    y: Tensor,
    n_in: int,
    n_boundary: int,
    k_neighbors: int = 5,
    random_state: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Build the full prototype set for one cascade layer.

        C = C_in ∪ C_boundary
        Q = Q_in ∪ Q_boundary

    Args:
        X:            (N, d) current data.
        y:            (N,)   labels.
        n_in:         k-means prototypes per class.
        n_boundary:   boundary prototypes per class.
        k_neighbors:  kNN neighborhood for boundary detection.
        random_state: seed for k-means.

    Returns:
        C: (K, d)  all prototype positions.
        Q: (K,)    prototype charges (sign only; learnable strength s_j
                   is initialised outside and multiplied in the cascade).
    """
    C_in,       Q_in       = kmeans_prototypes(X, y, n_in, random_state)
    C_boundary, Q_boundary = boundary_prototypes(X, y, n_boundary, k_neighbors)

    C = torch.cat([C_in, C_boundary], dim=0)   # (K, d)
    Q = torch.cat([Q_in, Q_boundary], dim=0)   # (K,)
    return C, Q


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_euclidean(X: Tensor) -> Tensor:
    """
    Compute full pairwise Euclidean distance matrix in O(N^2 d).

    Uses the expansion ‖a-b‖^2 = ‖a‖^2 + ‖b‖^2 - 2 a·b to stay
    fully on-device without Python loops.

    Note: for N > ~20k this becomes a bottleneck. At that scale,
    replace with an approximate kNN index (e.g. faiss.IndexFlatL2).
    """
    sq = (X ** 2).sum(dim=1)                          # (N,)
    D_sq = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * X @ X.T
    D_sq = D_sq.clamp(min=0.0)                        # numerical safety
    return torch.sqrt(D_sq)
