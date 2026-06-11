"""
early_fusion.py — DEFCEarlyFusion: early fusion of geometric coordinate spaces.

Early fusion approach:
  1. Transform data in all three geometries: Cartesian, Polar, Learnable
  2. Concatenate representations: [X_cart | X_polar | X_learnable]
  3. Run DEFC cascade on fused representation (6D → 6D)
  4. Train single SVM on final deformed space
  
This allows the cascade to learn cross-geometric interactions.

Reference: Multi-view learning, early fusion.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from .cascade import DEFCLayer
from .stopping import StoppingController
from .metrics import fisher_weights


class DEFCEarlyFusion(BaseEstimator, ClassifierMixin):
    """
    Early fusion of DEFC across multiple geometric coordinate systems.
    
    Combines Cartesian, Polar, and Learnable coordinate representations
    before cascade, allowing the deformation layers to exploit
    cross-geometric dependencies.
    
    Parameters
    ----------
    max_layers : int
        Hard upper bound on cascade depth.
    eta : float
        Initial deformation step size.
    n_in : int
        Interior prototypes per class per layer.
    n_boundary : int
        Boundary prototypes per class per layer.
    gamma : float
        Yukawa decay rate.
    lam : float
        Ridge regularisation.
    k_neighbors : int
        Neighbourhood size for KNN graph.
    tau : float
        Minimum structural overlap before halving η.
    delta : float
        Minimum margin improvement to add another layer.
    eta_min : float
        Minimum step size before termination.
    update_in_every : int
        Re-compute interior prototypes every T layers.
    device : str
        'cpu', 'cuda', or 'mps'.
    random_state : int
        Global reproducibility seed.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        max_layers:      int   = 20,
        eta:             float = 0.5,
        n_in:            int   = 5,
        n_boundary:      int   = 5,
        gamma:           float = 1.0,
        lam:             float = 1e-3,
        k_neighbors:     int   = 10,
        tau:             float = 0.7,
        delta:           float = 1e-3,
        eta_min:         float = 1e-4,
        update_in_every: int   = 3,
        device:          str   = "cpu",
        random_state:    int   = 0,
        verbose:         int   = 1,
    ):
        self.max_layers      = max_layers
        self.eta             = eta
        self.n_in            = n_in
        self.n_boundary      = n_boundary
        self.gamma           = gamma
        self.lam             = lam
        self.k_neighbors     = k_neighbors
        self.tau             = tau
        self.delta           = delta
        self.eta_min         = eta_min
        self.update_in_every = update_in_every
        self.device          = device
        self.random_state    = random_state
        self.verbose         = verbose

    # ------------------------------------------------------------------
    # Coordinate transformations
    # ------------------------------------------------------------------

    def _to_polar(self, X: np.ndarray) -> np.ndarray:
        """Convert 2D Cartesian to polar (r, θ)."""
        if X.shape[1] != 2:
            X = PCA(2).fit_transform(X)
        center = X.mean(axis=0)
        X_centered = X - center
        r = np.linalg.norm(X_centered, axis=1, keepdims=True)
        theta = np.arctan2(X_centered[:, 1], X_centered[:, 0]).reshape(-1, 1)
        return np.hstack([r, theta])

    def _learn_coordinates(self, X: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        """Learn Fisher-optimized linear transformation."""
        dev = torch.device(self.device)
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        
        d = X.shape[1]
        A = torch.eye(d, device=dev, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([A], lr=0.2)
        
        y_np = y.cpu().numpy()
        class_0_mask = y_np == -1
        class_1_mask = y_np == 1
        
        best_fisher = -np.inf
        best_A = A.clone().detach()
        
        for _ in range(100):
            X_proj = X_t @ A
            class_0 = X_proj[class_0_mask]
            class_1 = X_proj[class_1_mask]
            
            if len(class_0) == 0 or len(class_1) == 0:
                break
            
            mean_0 = class_0.mean(dim=0)
            mean_1 = class_1.mean(dim=0)
            
            between = torch.sum((mean_0 - mean_1) ** 2)
            var_0 = torch.sum(class_0.var(dim=0))
            var_1 = torch.sum(class_1.var(dim=0))
            within = (var_0 + var_1) / 2
            
            fisher = between / (within + 1e-8)
            fisher_val = fisher.item()
            
            if fisher_val > best_fisher:
                best_fisher = fisher_val
                best_A = A.clone().detach()
            
            loss = -fisher
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return best_A

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DEFCEarlyFusion":
        """
        Fit early fusion DEFC.
        
        Steps:
          1. Encode labels to {-1, +1}
          2. Learn learnable coordinates (A matrix)
          3. Normalize and transform to all three geometries
          4. Concatenate: X_fused = [X_cart | X_polar | X_learn]
          5. Run cascade on X_fused (6D → 6D)
          6. Train RBF SVM on final representation
        """
        torch.manual_seed(self.random_state)
        dev = torch.device(self.device)

        # --- label encoding ---
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "DEFC is binary."
        y_enc = np.where(y == self.classes_[0], -1.0, 1.0)

        # --- normalize input ---
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

        # --- learn coordinates (on scaled data) ---
        X_t = torch.tensor(X_scaled, device=dev)
        y_t = torch.tensor(y_enc, dtype=torch.float32, device=dev)
        self.A_ = self._learn_coordinates(X_scaled, y_t)

        # --- compute all three representations ---
        X_cart = X_scaled
        X_polar = self._to_polar(X_scaled)
        X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
        
        X_learn = (X_scaled @ self.A_.cpu().numpy()).astype(np.float32)
        X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)

        # --- early fusion: concatenate all representations ---
        X_fused = np.hstack([X_cart, X_polar_scaled, X_learn_scaled]).astype(np.float32)

        if self.verbose >= 1:
            print(f"[DEFCEarlyFusion] Fused representation shape: {X_fused.shape}")
            print(f"  Cartesian:  {X_cart.shape}")
            print(f"  Polar:      {X_polar_scaled.shape}")
            print(f"  Learnable:  {X_learn_scaled.shape}")

        # --- cascade on fused representation ---
        X_t_fused = torch.tensor(X_fused, device=dev)
        y_t = torch.tensor(y_enc, dtype=torch.float32, device=dev)

        ctrl = StoppingController(
            X_0     = X_t_fused,
            k       = self.k_neighbors,
            tau     = self.tau,
            delta   = self.delta,
            eta_min = self.eta_min,
        )

        self.layers_: list[DEFCLayer] = []
        self.etas_: list[float] = []

        X_k = X_t_fused
        eta_k = self.eta

        for k in range(self.max_layers):
            refresh_in = (k % self.update_in_every == 0)

            layer = DEFCLayer(
                n_in         = self.n_in if refresh_in else 1,
                n_boundary   = self.n_boundary,
                gamma        = self.gamma,
                lam          = self.lam,
                k_neighbors  = self.k_neighbors,
                random_state = self.random_state + k,
            )
            layer.fit(X_k, y_t)

            X_next = layer(X_k, eta=eta_k)
            stop, eta_k = ctrl.step(X_next, y_t, eta_k)

            self.layers_.append(layer)
            self.etas_.append(eta_k)

            X_k = X_next

            if stop:
                break

        self.n_layers_ = len(self.layers_)

        # --- final rbf SVM ---
        X_L_np = X_k.detach().cpu().numpy()
        self.svm_ = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale"
        )
        self.svm_.fit(X_L_np, y_enc)

        if self.verbose >= 1:
            print(f"[DEFCEarlyFusion] Trained {self.n_layers_} layers")

        return self

    # ------------------------------------------------------------------
    # transform / predict / score
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned transformations and cascade."""
        dev = torch.device(self.device)
        
        X_scaled = self.scaler_.transform(X).astype(np.float32)

        X_cart = X_scaled
        X_polar = self._to_polar(X_scaled)
        X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
        
        X_learn = (X_scaled @ self.A_.cpu().numpy()).astype(np.float32)
        X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)

        X_fused = np.hstack([X_cart, X_polar_scaled, X_learn_scaled]).astype(np.float32)
        X_t = torch.tensor(X_fused, device=dev)

        with torch.no_grad():
            for layer, eta in zip(self.layers_, self.etas_):
                X_t = layer(X_t, eta=eta)

        return X_t.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_L = self.transform(X)
        y_enc = self.svm_.predict(X_L)
        return np.where(y_enc == -1, self.classes_[0], self.classes_[1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return float((self.predict(X) == y).mean())

    def cascade_summary(self) -> dict:
        """Return cascade summary."""
        return {
            "n_layers": self.n_layers_,
            "n_protos": [
                (l.C.shape[0] if l.C is not None else 0)
                for l in self.layers_
            ],
            "final_eta": self.etas_[-1] if self.etas_ else self.eta,
        }
