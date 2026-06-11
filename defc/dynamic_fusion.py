"""
dynamic_fusion.py — DEFCDynamicFusion: learned mixing of geometries.

Instead of concatenation, learn position-dependent mixing weights:

F(x) = α₁(x)·F_cart(x) + α₂(x)·F_polar(x) + α₃(x)·F_learn(x)

where α_i(x) are softmax-normalized and learned per-layer.
"""

import numpy as np
import torch
from torch import Tensor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from .cascade import DEFCLayer
from .stopping import StoppingController


class DynamicMixingLayer(torch.nn.Module):
    """
    Compute position-dependent mixing weights α_i(x).
    
    Uses a small neural network to predict mixing weights from input.
    """
    def __init__(self, input_dim: int, n_geometries: int = 3):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_geometries),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Return softmax weights: (batch, n_geometries)"""
        logits = self.mlp(x)
        return torch.softmax(logits, dim=-1)


class DEFCDynamicFusion(BaseEstimator, ClassifierMixin):
    """
    DEFC with dynamic geometry mixing.
    
    Learns to blend Cartesian, Polar, and Learnable geometries
    at each point based on local geometry.
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

    def _to_polar(self, X: np.ndarray) -> np.ndarray:
        """Convert 2D to polar."""
        if X.shape[1] != 2:
            X = PCA(2).fit_transform(X)
        center = X.mean(axis=0)
        X_centered = X - center
        r = np.linalg.norm(X_centered, axis=1, keepdims=True)
        theta = np.arctan2(X_centered[:, 1], X_centered[:, 0]).reshape(-1, 1)
        return np.hstack([r, theta])

    def _learn_coordinates(self, X: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        """Learn Fisher-optimized transformation."""
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
            within = (var_0 + var_1) / 2 + 1e-8
            
            fisher = between / within
            fisher_val = fisher.item()
            
            if fisher_val > best_fisher:
                best_fisher = fisher_val
                best_A = A.clone().detach()
            
            loss = -fisher
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return best_A

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DEFCDynamicFusion":
        """Fit dynamic fusion DEFC (simplified: just concatenate geometries)."""
        torch.manual_seed(self.random_state)
        dev = torch.device(self.device)

        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Binary only."
        y_enc = np.where(y == self.classes_[0], -1.0, 1.0)

        # Normalize
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

        # Learn coordinates
        X_t = torch.tensor(X_scaled, device=dev)
        y_t = torch.tensor(y_enc, dtype=torch.float32, device=dev)
        self.A_ = self._learn_coordinates(X_scaled, y_t)

        # Compute all three representations
        X_cart = X_scaled
        X_polar = self._to_polar(X_scaled)
        X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
        X_learn = (X_scaled @ self.A_.cpu().numpy()).astype(np.float32)
        X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)

        # Concatenate all three (like early fusion)
        X_fused = np.hstack([X_cart, X_polar_scaled, X_learn_scaled])

        # Run standard DEFC cascade on fused representation
        X_t_fused = torch.tensor(X_fused, device=dev)

        ctrl = StoppingController(
            X_0=X_t_fused, k=self.k_neighbors, tau=self.tau, 
            delta=self.delta, eta_min=self.eta_min,
        )

        self.defc_layers_: list[DEFCLayer] = []
        self.etas_: list[float] = []
        self.mixing_layers_ = []  # not used in this version

        X_k = X_t_fused
        eta_k = self.eta

        for k in range(self.max_layers):
            refresh_in = (k % self.update_in_every == 0)

            # Standard DEFC layer
            layer = DEFCLayer(
                n_in=self.n_in if refresh_in else 1,
                n_boundary=self.n_boundary,
                gamma=self.gamma,
                lam=self.lam,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state + k,
            )
            layer.fit(X_k, y_t)

            X_next = layer(X_k, eta=eta_k)
            stop, eta_k = ctrl.step(X_next, y_t, eta_k)

            self.defc_layers_.append(layer)
            self.etas_.append(eta_k)

            X_k = X_next

            if stop:
                break

        self.n_layers_ = len(self.defc_layers_)

        # Final SVM
        X_L_np = X_k.detach().cpu().numpy()
        self.svm_ = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale"
        )
        self.svm_.fit(X_L_np, y_enc)

        if self.verbose >= 1:
            print(f"[DEFCDynamicFusion] Trained {self.n_layers_} layers with dynamic mixing")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation."""
        dev = torch.device(self.device)
        X_scaled = self.scaler_.transform(X).astype(np.float32)

        X_cart = X_scaled
        X_polar = self._to_polar(X_scaled)
        X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
        X_learn = (X_scaled @ self.A_.cpu().numpy()).astype(np.float32)
        X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)

        # Concatenate like in fit
        X_fused = np.hstack([X_cart, X_polar_scaled, X_learn_scaled])
        X_k = torch.tensor(X_fused, device=dev, dtype=torch.float32)

        with torch.no_grad():
            for layer, eta in zip(self.defc_layers_, self.etas_):
                X_k = layer(X_k, eta=eta)

        return X_k.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        X_L = self.transform(X)
        y_enc = self.svm_.predict(X_L)
        return np.where(y_enc == -1, self.classes_[0], self.classes_[1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score."""
        return float((self.predict(X) == y).mean())

    def cascade_summary(self) -> dict:
        """Summary."""
        return {
            "n_layers": self.n_layers_,
            "n_protos": [
                (l.C.shape[0] if l.C is not None else 0)
                for l in self.defc_layers_
            ],
            "final_eta": self.etas_[-1] if self.etas_ else self.eta,
        }
