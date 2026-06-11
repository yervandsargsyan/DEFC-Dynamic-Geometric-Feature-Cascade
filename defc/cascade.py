"""
cascade.py — A single DEFC cascade layer.

One layer does three things:
  1. Build (or refresh) prototypes in the current space X_k.
  2. Compute the vector field F_k(X_k).
  3. Apply the deformation step: X_{k+1} = X_k + η_k * F_k(X_k).

Prototype weights w_j are solved analytically (ridge regression)
using the potential matrix H:

    H[i,j] = q_j * φ̃(x_i, c_j)
    W       = (H^T H + λI)^{-1} H^T y

This is the "learning" step of each layer — no gradient descent needed.

Reference: DEFC §8.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .field      import compute_field
from .metrics    import fisher_weights, anisotropic_distance
from .prototypes import build_prototypes


class DEFCLayer(nn.Module):
    """
    One layer of the Dynamic Electromagnetic Field Cascade.

    Holds the prototype positions, charges, and learned weights for
    layer k. The forward pass deforms the input space X_k → X_{k+1}.

    Args:
        n_in:         k-means prototypes per class.
        n_boundary:   boundary prototypes per class.
        gamma:        Yukawa decay rate.
        lam:          ridge regularisation for weight learning.
        k_neighbors:  neighbourhood size for boundary detection.
        random_state: seed for k-means init.
    """

    def __init__(
        self,
        n_in:         int   = 5,
        n_boundary:   int   = 5,
        gamma:        float = 1.0,
        lam:          float = 1e-3,
        k_neighbors:  int   = 5,
        random_state: int   = 0,
        use_radial:   bool  = False,
        lambda_rad:   float = 0.1,
    ):
        super().__init__()
        self.n_in         = n_in
        self.n_boundary   = n_boundary
        self.gamma        = gamma
        self.lam          = lam
        self.k_neighbors  = k_neighbors
        self.random_state = random_state
        self.use_radial   = use_radial
        self.lambda_rad   = lambda_rad

        # set at fit() time
        self.C:       Tensor | None = None  # (K, d) prototype positions
        self.Q_sign:  Tensor | None = None  # (K,)   fixed charge signs
        self.s:       Tensor | None = None  # (K,)   learnable strengths ∈ [0,1]
        self.W:       Tensor | None = None  # (K,)   ridge-regression weights
        self.v:       Tensor | None = None  # (d,)   Fisher anisotropy weights

    # ------------------------------------------------------------------
    # Fitting (analytic, no gradient descent)
    # ------------------------------------------------------------------

    def fit(self, X: Tensor, y: Tensor) -> "DEFCLayer":
        """
        Fit this layer to the current (possibly deformed) space X.

          1. Compute Fisher anisotropy weights v.
          2. Build prototypes C = C_in ∪ C_boundary.
          3. Initialise learnable strengths s_j = 1.
          4. Solve for ridge weights W via the potential matrix H.

        Args:
            X: (N, d) data in the current space.
            y: (N,)   labels {-1, +1}.

        Returns:
            self (for chaining).
        """
        device = X.device

        # 1. anisotropy weights
        self.v = fisher_weights(X, y)                        # (d,)

        # 2. prototypes
        self.C, self.Q_sign = build_prototypes(
            X, y,
            n_in         = self.n_in,
            n_boundary   = self.n_boundary,
            k_neighbors  = self.k_neighbors,
            random_state = self.random_state,
        )
        K = self.C.shape[0]

        # 3. learnable strengths (start uniform; can be fine-tuned later)
        self.s = torch.ones(K, device=device)

        # 4. effective charges q_j = sign * strength
        charges = self.Q_sign * self.s                       # (K,)

        # 5. potential matrix H[i,j] = q_j * φ̃(x_i, c_j)
        H = self._potential_matrix(X, charges)               # (N, K)

        # 6. ridge regression:  W = (H^T H + λI)^{-1} H^T y
        self.W = self._ridge(H, y)                           # (K,)

        return self

    # ------------------------------------------------------------------
    # Forward pass — deformation step
    # ------------------------------------------------------------------

    def forward(self, X: Tensor, eta: float) -> Tensor:
        """
        Apply one deformation step:

            X_{k+1} = X_k + η * F_k(X_k)

        where F_k is either:
          - standard vector field (use_radial=False)
          - hybrid vector + radial field (use_radial=True)

        Args:
            X:   (N, d) input data positions.
            eta: step size for this layer.

        Returns:
            X_new: (N, d) deformed data positions.
        """
        assert self.C is not None, "Call .fit() before .forward()"

        charges = self.Q_sign * self.s   # (K,)

        if self.use_radial:
            from .field import compute_hybrid_field
            F = compute_hybrid_field(
                X       = X,
                C       = self.C,
                charges = charges,
                weights = self.W,
                v       = self.v,
                gamma   = self.gamma,
                lambda_rad = self.lambda_rad,
            )
        else:
            from .field import compute_field
            F = compute_field(
                X       = X,
                C       = self.C,
                charges = charges,
                weights = self.W,
                v       = self.v,
                gamma   = self.gamma,
            )

        return X + eta * F

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _potential_matrix(self, X: Tensor, charges: Tensor) -> Tensor:
        """
        Build H[i,j] = q_j * φ̃(x_i, c_j) using the same normalised
        Yukawa kernel as the forward field.
        """
        from .field import yukawa_kernel, normalise_kernel

        D        = anisotropic_distance(X, self.C, self.v)   # (N, K)
        phi_raw  = yukawa_kernel(D, gamma=self.gamma)         # (N, K)
        phi_norm = normalise_kernel(phi_raw)                  # (N, K)
        H        = phi_norm * charges.unsqueeze(0)            # (N, K)
        return H

    def _ridge(self, H: Tensor, y: Tensor) -> Tensor:
        """
        Solve W = (H^T H + λI)^{-1} H^T y via torch.linalg.solve.

        This is the analytic (closed-form) optimum — no iterations.
        """
        K   = H.shape[1]
        HtH = H.T @ H                                          # (K, K)
        Hty = H.T @ y.float()                                  # (K,)
        A   = HtH + self.lam * torch.eye(K, device=H.device)  # (K, K)
        W   = torch.linalg.solve(A, Hty)                       # (K,)
        return W
