"""
stopping.py — Cascade stopping criteria for DEFC.

Two independent guards prevent over-deformation:

  A) Structural stability (KNN graph overlap)
     ─────────────────────────────────────────
     We build a KNN graph G_0 on the original data X_0 *once*.
     After each deformation step we compute G_k on X_k and measure
     what fraction of the original edges survived:

         S_k = |E(G_0) ∩ E(G_k)| / |E(G_0)|

     If S_k drops below threshold τ the deformation is destroying
     the intrinsic neighbourhood structure → halve η.
     If η < η_min → stop.

  B) Margin utility (linear SVM margin)
     ────────────────────────────────────
     We train a linear SVM on X_k after each layer and record the
     margin m_k. If the margin improvement falls below δ the layer
     added no discriminative value → stop.

Reference: DEFC §9–10.
"""

import torch
from torch import Tensor
from sklearn.svm import SVC
import numpy as np


# ---------------------------------------------------------------------------
# KNN graph
# ---------------------------------------------------------------------------

def build_knn_graph(X: Tensor, k: int) -> set[tuple[int, int]]:
    """
    Build a KNN graph as a set of undirected edges.

    Each point keeps its k nearest neighbors. Edges are stored as
    sorted (i, j) tuples with i < j so the set is symmetric.

    Args:
        X: (N, d) data points.
        k: number of neighbors.

    Returns:
        edges: set of (i, j) pairs representing the graph E(G).
    """
    N = X.shape[0]
    k = min(k, N - 1)

    # pairwise distance  (reuse the same trick as in prototypes.py)
    sq = (X ** 2).sum(dim=1)
    D  = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * X @ X.T
    D  = D.clamp(min=0.0)
    D.fill_diagonal_(float("inf"))

    _, knn_idx = torch.topk(D, k=k, dim=1, largest=False)   # (N, k)
    knn_idx    = knn_idx.cpu().numpy()

    edges = set()
    for i in range(N):
        for j in knn_idx[i]:
            edges.add((min(i, int(j)), max(i, int(j))))

    return edges


def structural_overlap(edges_0: set, edges_k: set) -> float:
    """
    Compute the structural preservation score S_k.

        S_k = |E(G_0) ∩ E(G_k)| / |E(G_0)|

    S_k = 1.0 → topology perfectly preserved.
    S_k → 0   → neighbourhood structure destroyed.

    Args:
        edges_0: edge set of the original graph G_0.
        edges_k: edge set of the current graph G_k.

    Returns:
        S_k in [0, 1].
    """
    if len(edges_0) == 0:
        return 1.0
    return len(edges_0 & edges_k) / len(edges_0)


# ---------------------------------------------------------------------------
# Margin utility
# ---------------------------------------------------------------------------

def rbf_svm_margin(X: Tensor, y: Tensor, verbose: bool = False) -> float:
    """
    Fit a RBF SVM and return accuracy (not margin).
    
    Key insight from debugging: we should stop when RBF SVM accuracy is already high.
    This prevents over-deformation that damages good geometry.
    
    On circles: Layer 1 gives 98% accuracy. Layer 2+ makes it worse.
    Solution: stop when accuracy exceeds threshold (~0.97).

    Args:
        X: (N, d) current data.
        y: (N,)   labels {-1, +1}.
        verbose: if True, print margin information.

    Returns:
        margin: float in (0, 1], based on RBF SVM accuracy.
    """
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    try:
        clf = SVC(kernel="rbf", C=2.0, gamma="scale")
        clf.fit(X_np, y_np)
        
        # Return accuracy directly
        accuracy = clf.score(X_np, y_np)
        
        if verbose:
            num_sv = len(clf.support_)
            print(f"  [acc={accuracy:.4f}, sv={num_sv}/{len(X_np)}]", end="")
            
        return float(accuracy)
    except Exception:
        if verbose:
            print(f"  [margin=ERROR]", end="")
        return 0.5


# ---------------------------------------------------------------------------
# Stopping controller (stateful)
# ---------------------------------------------------------------------------

class StoppingController:
    """
    Stateful stopping controller for the DEFC cascade.

    Holds G_0 (built once at the start) and tracks margin history.
    Called once per cascade layer to determine whether to continue,
    reduce η, or stop entirely.

    Uses two criteria:
      A) Structural stability: if KNN graph is too corrupted, halve η
      B) Margin improvement: if no improvement for N consecutive layers, stop

    Usage:
        ctrl = StoppingController(X_0, k=10, tau=0.7, delta=1e-3,
                                  eta_min=1e-4)
        for each layer k:
            stop, eta_k = ctrl.step(X_k, y, eta_k)
            if stop:
                break
    """

    def __init__(
        self,
        X_0: Tensor,
        k: int     = 10,
        tau: float = 0.7,
        delta: float = 1e-3,
        eta_min: float = 1e-4,
    ):
        """
        Args:
            X_0:     (N, d) original (pre-cascade) data.
            k:       neighbourhood size for KNN graph.
            tau:     minimum acceptable structural overlap S_k.
            delta:   relative improvement threshold (e.g., 0.01 = 1%).
            eta_min: if η drops below this, stop regardless.
        """
        self.edges_0  = build_knn_graph(X_0, k)
        self.k        = k
        self.tau      = tau
        self.delta    = delta
        self.eta_min  = eta_min
        self.margin_prev = -0.1
        self.no_improve_count = 0

    def step(
        self,
        X_k: Tensor,
        y:   Tensor,
        eta: float,
        layer_num: int = -1,
    ) -> tuple[bool, float]:
        """
        Stopping when RBF SVM accuracy is already good.
        
        Key insight: on easy datasets (circles, moons, etc.) 1-2 deformation layers
        often produce excellent RBF SVM accuracy. Further deformation can degrade this.
        
        Stopping rules:
        1. If accuracy > 0.97: we've done well, stop
        2. If structural overlap S_k < tau: deformation damaged topology, halve eta
        3. If eta becomes too small: give up

        Args:
            X_k: (N, d)  data after deformation at layer k.
            y:   (N,)    labels.
            eta: current step size.
            layer_num: layer number for debugging.

        Returns:
            stop:    True if cascade should terminate.
            eta_new: (possibly halved) step size for the next layer.
        """
        # --- criterion A: structural stability ---
        edges_k = build_knn_graph(X_k, self.k)
        S_k     = structural_overlap(self.edges_0, edges_k)

        if S_k < self.tau:
            eta = eta / 2.0
            if eta < self.eta_min:
                return True, eta

        # --- criterion B: accuracy-based stopping ---
        accuracy = rbf_svm_margin(X_k, y)
        
        # Stop if accuracy is already high (>0.95) - prevent over-deformation
        if accuracy > 0.95:
            return True, eta
        
        # Otherwise, stop if improvement is very small (<0.001 = 0.1%)
        improvement = accuracy - self.margin_prev
        if improvement < 0.001:
            self.no_improve_count += 1
        else:
            self.no_improve_count = 0
        
        # Stop after 4 consecutive non-improving steps
        if self.no_improve_count >= 4:
            return True, eta

        self.margin_prev = accuracy
        return False, eta
