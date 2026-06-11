"""
field.py — Yukawa potential kernel and prototype vector field.

This is the physical heart of DEFC.

Each prototype c_j acts as a charged source that exerts a force on
every data point x. The force has:
  - intensity  : screened Yukawa potential  φ(r) = exp(-γ r) / (r + ε)
  - direction  : unit vector from c_j toward x  (attractive if same class,
                 repulsive if opposite — controlled by charge q_j)

The total field at x sums contributions from all prototypes:

    F(x) = Σ_j  w_j * q_j * φ̃(x, c_j) * (x - c_j) / ‖x - c_j‖

where φ̃ is the row-normalised kernel (softmax-style over prototypes)
to prevent scale explosion.

After summation the field vector is clipped via:

    F(x) ← tanh(‖F(x)‖) * F(x) / ‖F(x)‖

so the maximum displacement per step is bounded to 1 in every direction.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .metrics import anisotropic_distance


# ---------------------------------------------------------------------------
# Yukawa kernel
# ---------------------------------------------------------------------------

def yukawa_kernel(
    D: Tensor,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Screened Coulomb (Yukawa) potential.

        φ(r) = exp(-γ r) / (r + ε)

    Properties:
      - Decays exponentially at long range  → far prototypes contribute little.
      - Regularised at r=0 by ε            → no singularity when x == c_j.
      - γ controls the interaction radius  → larger γ = more local field.

    Args:
        D:     (N, K) pairwise distance matrix from anisotropic_distance().
        gamma: decay rate (hyperparameter).
        eps:   numerical regulariser.

    Returns:
        phi: (N, K) raw kernel values.
    """
    return torch.exp(-gamma * D) / (D + eps)


def normalise_kernel(phi: Tensor, eps: float = 1e-8, tau: float = 0.5) -> Tensor:
    """
    Stable sharp kernel normalization (soft attention version).

    Instead of forcing probabilities that sum to 1 (which destroys
    contrast), we use a temperature-controlled softmax over negative
    distances to preserve local dominance structure.

    This keeps:
      - strong local interactions
      - competition between prototypes
      - numerical stability

    Args:
        phi: (N, K) raw kernel values OR distance-derived similarities.
        eps: numerical stability term (unused but kept for API compatibility).
        tau: temperature (lower = sharper decision boundaries).

    Returns:
        phi_norm: (N, K) row-normalised but contrast-preserving weights.
    """

    # convert to logit-like scores
    logits = phi / (phi.max(dim=1, keepdim=True).values + eps)
    
    # sharpened softmax (key for XOR-type structures)
    phi_norm = torch.softmax(logits / tau, dim=1)

    return phi_norm


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

def compute_field(
    X: Tensor,
    C: Tensor,
    charges: Tensor,
    weights: Tensor,
    v: Tensor,
    gamma: float = 1.0,
    kernel_eps: float = 1e-6,
    dir_eps: float = 1e-8,
) -> Tensor:
    """
    Compute the prototype-induced vector field at every data point.

    Step 1 — distances:
        D[i,j] = d_W(x_i, c_j)                              (N, K)

    Step 2 — normalised kernel intensity:
        φ̃[i,j] = φ(D[i,j]) / Σ_j φ(D[i,j])                (N, K)

    Step 3 — direction vectors (unit, from prototype toward point):
        u[i,j] = (x_i - c_j) / ‖x_i - c_j‖                (N, K, d)

    Step 4 — weighted field sum:
        F[i] = Σ_j  w_j * q_j * φ̃[i,j] * u[i,j]           (N, d)

    Step 5 — bounded clipping:
        F[i] ← tanh(‖F[i]‖) * F[i] / ‖F[i]‖

    Args:
        X:       (N, d)  current data positions.
        C:       (K, d)  prototype positions.
        charges: (K,)    signed strengths q_j = sign(class) * s_j.
        weights: (K,)    non-negative prototype weights w_j.
        v:       (d,)    anisotropy weights (Fisher).
        gamma:   Yukawa decay rate.
        kernel_eps: regulariser inside φ.
        dir_eps:    regulariser inside direction normalisation.

    Returns:
        F: (N, d)  clipped displacement field. Each entry is in (-1, 1)^d
           after the tanh clip, so the actual step is η * F.
    """
    N, d = X.shape
    K    = C.shape[0]

    # --- distances -----------------------------------------------------------
    D = anisotropic_distance(X, C, v, eps=kernel_eps)       # (N, K)

    # --- kernel intensity (normalised) ---------------------------------------
    phi_raw  = yukawa_kernel(D, gamma=gamma, eps=kernel_eps) # (N, K)
    phi_norm = normalise_kernel(phi_raw)                      # (N, K)

    # --- direction vectors ---------------------------------------------------
    # diff[i,j] = x_i - c_j,  shape (N, K, d)
    diff = X.unsqueeze(1) - C.unsqueeze(0)                   # (N, K, d)

    # Euclidean norm for direction (plain, not weighted)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=dir_eps)  # (N,K,1)
    unit = diff / diff_norm                                   # (N, K, d)

    # --- signed scalar field per prototype -----------------------------------
    # scalar[i,j] = w_j * q_j * φ̃[i,j]
    wq = weights * charges                                    # (K,)
    scalar = phi_norm * wq.unsqueeze(0)                       # (N, K)

    # --- sum over prototypes → (N, d) ----------------------------------------
    # scalar[:, :, None] * unit  →  (N, K, d),  then sum over K
    F = (scalar.unsqueeze(-1) * unit).sum(dim=1)              # (N, d)

    # --- bounded clipping ----------------------------------------------------
    F = clip_field(F, eps=dir_eps)

    return F


def clip_field(F: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Bound the field magnitude via tanh, preserving direction.

        F ← tanh(‖F‖) * F / ‖F‖

    This maps any magnitude to (0, 1), so the maximum displacement
    at any single cascade step is bounded by the step size η.

    Args:
        F:   (N, d) raw field vectors.
        eps: floor for the norm to avoid 0/0.

    Returns:
        F_clipped: (N, d) unit-direction field scaled by tanh(‖F‖).
    """
    norms = F.norm(dim=-1, keepdim=True).clamp(min=eps)   # (N, 1)
    return torch.tanh(norms) * (F / norms)


# ---------------------------------------------------------------------------
# Scalar potential field (radial stabilizer)
# ---------------------------------------------------------------------------

def compute_scalar_potential(
    X: Tensor,
    C: Tensor,
    charges: Tensor,
    weights: Tensor,
    v: Tensor,
    gamma: float = 1.0,
    kernel_eps: float = 1e-6,
) -> Tensor:
    """
    Compute scalar potential energy field E(x) for radial structure support.

    Prototypes define an energy landscape:
        E(x) = Σ_j  w_j * q_j * ψ_j(x)

    where ψ_j(x) = φ(d_W(x, c_j)) is the normalized kernel intensity.

    This energy field helps stabilize radial decision boundaries (e.g., circles)
    by providing an additional radial restoring force: F_rad = -∇E.

    Args:
        X:       (N, d)  current data positions.
        C:       (K, d)  prototype positions.
        charges: (K,)    signed strengths.
        weights: (K,)    prototype weights.
        v:       (d,)    anisotropy weights (Fisher).
        gamma:   Yukawa decay rate.
        kernel_eps: regulariser.

    Returns:
        E: (N,)  scalar potential energy at each point.
    """
    D = anisotropic_distance(X, C, v, eps=kernel_eps)      # (N, K)
    phi_raw = yukawa_kernel(D, gamma=gamma, eps=kernel_eps)  # (N, K)
    phi_norm = normalise_kernel(phi_raw)                     # (N, K)
    
    # E[i] = Σ_j  w_j * q_j * φ̃[i,j]
    wq = weights * charges                                   # (K,)
    E = (phi_norm * wq.unsqueeze(0)).sum(dim=1)             # (N,)
    
    return E


def compute_radial_stabilizer(
    X: Tensor,
    C: Tensor,
    charges: Tensor,
    weights: Tensor,
    v: Tensor,
    gamma: float = 1.0,
    kernel_eps: float = 1e-6,
    dir_eps: float = 1e-8,
) -> Tensor:
    """
    Compute radial stabilizer field from scalar potential gradient.

    F_rad = -∇_x E(x)

    Approximates the gradient analytically using:
        ∇_x φ(d(x,c)) ≈ φ'(d) * ∇_x d(x,c)
    
    where φ'(d) = -γ*φ(d) - φ(d)/(d+ε) is the kernel derivative.

    Args:
        X:       (N, d)  current data positions.
        C:       (K, d)  prototype positions.
        charges: (K,)    signed strengths.
        weights: (K,)    prototype weights.
        v:       (d,)    anisotropy weights.
        gamma:   Yukawa decay rate.
        kernel_eps: regulariser.
        dir_eps: direction regulariser.

    Returns:
        F_rad: (N, d)  radial stabilizer field.
    """
    N, d = X.shape
    K = C.shape[0]
    
    # Distances and kernel
    D = anisotropic_distance(X, C, v, eps=kernel_eps)         # (N, K)
    phi_raw = yukawa_kernel(D, gamma=gamma, eps=kernel_eps)   # (N, K)
    phi_norm = normalise_kernel(phi_raw)                       # (N, K)
    
    # Kernel derivative: φ'(d) = -γ*φ(d) - φ(d)/(d+ε)
    # d(φ)/d(d) = -γ*exp(-γ*d)/(d+ε) - exp(-γ*d)/(d+ε)²
    phi_prime = -gamma * phi_raw - phi_raw / (D + kernel_eps)**2  # (N, K)
    
    # Direction vectors (unit, from prototype to point)
    diff = X.unsqueeze(1) - C.unsqueeze(0)                    # (N, K, d)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=dir_eps)  # (N, K, 1)
    unit = diff / diff_norm                                    # (N, K, d)
    
    # ∇_x φ(d) ≈ φ'(d) * unit  for each (i,j) pair
    # Energy gradient: ∇E = Σ_j w_j * q_j * ∇_x φ(d)
    wq = weights * charges                                      # (K,)
    
    # Shape: (N, K) -> (N, K, 1) -> (N, K, d)
    grad_contrib = phi_prime.unsqueeze(-1) * unit              # (N, K, d)
    
    # Weighted sum: (N, K, d) * (K,) -> (N, d)
    F_rad = (grad_contrib * wq.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (N, d)
    
    # Negate to get -∇E (attraction toward energy minima)
    F_rad = -F_rad
    
    # Clip to prevent explosion
    F_rad = clip_field(F_rad, eps=dir_eps)
    
    return F_rad


def compute_hybrid_field(
    X: Tensor,
    C: Tensor,
    charges: Tensor,
    weights: Tensor,
    v: Tensor,
    gamma: float = 1.0,
    lambda_rad: float = 0.1,
    kernel_eps: float = 1e-6,
    dir_eps: float = 1e-8,
) -> Tensor:
    """
    Combine vector field and radial stabilizer for improved circular separation.

    F_total = F_vec + λ_rad * F_rad

    where:
      - F_vec is the standard DEFC prototype-attraction field
      - F_rad is the gradient of scalar potential energy
      - λ_rad ∈ [0.1, 0.5] weights the radial component

    This hybrid approach preserves the interpretable prototype dynamics
    while adding energy-based radial structure support.

    Args:
        X, C, charges, weights, v, gamma, kernel_eps, dir_eps: as before.
        lambda_rad: mixing coefficient for radial stabilizer (default 0.1).

    Returns:
        F_hybrid: (N, d) combined field = F_vec + λ_rad * F_rad.
    """
    F_vec = compute_field(X, C, charges, weights, v, gamma, kernel_eps, dir_eps)
    F_rad = compute_radial_stabilizer(X, C, charges, weights, v, gamma, kernel_eps, dir_eps)
    
    F_hybrid = F_vec + lambda_rad * F_rad
    F_hybrid = clip_field(F_hybrid, eps=dir_eps)
    
    return F_hybrid
