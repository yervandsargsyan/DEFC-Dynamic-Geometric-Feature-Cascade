"""
model.py — DEFCClassifier: top-level sklearn-compatible estimator.

This is the entry point for all experiments. It wraps the full cascade
in a fit / predict / score interface so it can be dropped into any
sklearn pipeline or benchmark without modification.

    clf = DEFCClassifier(max_layers=10, eta=0.5, n_in=5, n_boundary=5)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

Internally:
  - fit() runs the cascade: build layer → deform → check stopping.
  - transform() applies the learned deformation sequence to new data.
  - predict() transforms then classifies with the trained linear SVM.

Reference: DEFC §11–12.
"""

import torch
import numpy as np
from torch import Tensor
from sklearn.base   import BaseEstimator, ClassifierMixin
from sklearn.svm    import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .cascade  import DEFCLayer
from .stopping import StoppingController


class DEFCClassifier(BaseEstimator, ClassifierMixin):
    """
    Dynamic Electromagnetic Field Cascade Classifier.

    Parameters
    ----------
    max_layers : int
        Hard upper bound on cascade depth (safety cap).
    eta : float
        Initial deformation step size η_0.
    n_in : int
        k-means (interior) prototypes per class per layer.
    n_boundary : int
        Boundary prototypes per class per layer.
    gamma : float
        Yukawa decay rate (interaction radius).
    lam : float
        Ridge regularisation for analytic weight solving.
    k_neighbors : int
        Neighbourhood size for boundary sampling and KNN graph.
    tau : float
        Minimum structural overlap S_k before halving η.
    delta : float
        Minimum margin improvement to add another layer.
    eta_min : float
        Minimum step size before cascade terminates.
    update_in_every : int
        Re-compute interior (k-means) prototypes every T layers.
        Boundary prototypes are always re-computed.
    device : str
        'cpu', 'cuda', or 'mps'.
    random_state : int
        Global reproducibility seed.
    use_polar : bool
        If True, convert 2D data to polar coordinates before cascade.
    learn_coords : bool
        If True, learn optimal Fisher-criterion linear transformation.
    use_latent_state : bool
        If True, evolve latent geometry state Z independently from data X.
        Z starts as copy of X but evolves in cascade with memory inertia.
    memory_inertia : float
        Inertia coefficient β ∈ [0, 1] controlling latent state evolution.
        Z_{k+1} = (1-β)Z_k + β(Z_k + ηF_k). Default 0.7 (stable evolution).
    use_radial : bool
        If True, add scalar potential radial stabilizer field for circular separation.
        Helps with radially symmetric datasets like circles. Default False.
    lambda_rad : float
        Weight coefficient for radial stabilizer (0.1-0.5). Default 0.1.
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
        use_polar:       bool  = False,
        learn_coords:    bool  = False,
        use_latent_state: bool = False,
        memory_inertia:  float = 0.7,
        use_radial:      bool  = False,
        lambda_rad:      float = 0.1,
    ):
        self.max_layers       = max_layers
        self.eta              = eta
        self.n_in             = n_in
        self.n_boundary       = n_boundary
        self.gamma            = gamma
        self.lam              = lam
        self.k_neighbors      = k_neighbors
        self.tau              = tau
        self.delta            = delta
        self.eta_min          = eta_min
        self.update_in_every  = update_in_every
        self.device           = device
        self.random_state     = random_state
        self.use_polar        = use_polar
        self.learn_coords     = learn_coords
        self.use_latent_state = use_latent_state
        self.memory_inertia   = memory_inertia
        self.use_radial       = use_radial
        self.lambda_rad       = lambda_rad

    # ------------------------------------------------------------------
    # Learnable coordinate transformation
    # ------------------------------------------------------------------

    def _learn_coordinates(self, X: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        """
        Learn an optimal linear transformation A such that X @ A maximizes
        Fisher discriminant criterion. Uses PyTorch optimization.
        
        Returns learned transformation matrix as torch tensor.
        """
        dev = torch.device(self.device)
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        
        # Initialize A as identity + small random perturbation
        d = X.shape[1]
        A = torch.eye(d, device=dev, dtype=torch.float32, requires_grad=True)
        
        optimizer = torch.optim.Adam([A], lr=0.2)
        
        y_np = y.cpu().numpy()
        class_0_mask = y_np == -1
        class_1_mask = y_np == 1
        
        best_fisher = -np.inf
        best_A = A.clone().detach()
        
        for iteration in range(100):
            X_proj = X_t @ A
            
            class_0 = X_proj[class_0_mask]
            class_1 = X_proj[class_1_mask]
            
            if len(class_0) == 0 or len(class_1) == 0:
                break
            
            mean_0 = class_0.mean(dim=0)
            mean_1 = class_1.mean(dim=0)
            
            # Between-class variance (sum over dimensions)
            between = torch.sum((mean_0 - mean_1) ** 2)
            
            # Within-class variance (pooled, per dimension)
            var_0 = torch.sum(class_0.var(dim=0))
            var_1 = torch.sum(class_1.var(dim=0))
            within = (var_0 + var_1) / 2
            
            # Fisher criterion: maximize between / within
            fisher = between / (within + 1e-8)
            
            # Track best solution
            fisher_val = fisher.item()
            if fisher_val > best_fisher:
                best_fisher = fisher_val
                best_A = A.clone().detach()
            
            loss = -fisher  # minimize negative fisher
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return best_A

    # ------------------------------------------------------------------
    # Polar coordinate helpers
    # ------------------------------------------------------------------

    def _to_polar(self, X: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian to polar coordinates (r, θ).
        Center is computed as mean of X. Works for 2D data.
        """
        if X.shape[1] != 2:
            X = PCA(2).fit_transform(X)
        
        center = X.mean(axis=0)
        X_centered = X - center
        r = np.linalg.norm(X_centered, axis=1, keepdims=True)
        theta = np.arctan2(X_centered[:, 1], X_centered[:, 0]).reshape(-1, 1)
        return np.hstack([r, theta])

    def _from_polar(self, X_polar: np.ndarray) -> np.ndarray:
        """
        Convert polar (r, θ) back to Cartesian coordinates.
        """
        r = X_polar[:, 0]
        theta = X_polar[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        X_cartesian = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
        return X_cartesian + self.center_

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DEFCClassifier":
        """
        Run the full DEFC cascade and train the final linear SVM.

        Steps:
          1. Normalise X → X_0.
          2. Build G_0 (KNN graph, once only).
          3. For each layer k:
               a. Fit DEFCLayer on X_k (prototypes + ridge weights).
               b. Deform: X_{k+1} = X_k + η_k * F_k(X_k).
               c. Check stopping criteria; break if triggered.
          4. Train RBF SVM on the final X_L.

        Args:
            X: (N, d) feature matrix.
            y: (N,)   binary labels (any two distinct values).

        Returns:
            self.
        """
        torch.manual_seed(self.random_state)
        dev = torch.device(self.device)

        # --- label encoding: map to {-1, +1} --------------------------------
        self.classes_  = np.unique(y)
        assert len(self.classes_) == 2, "DEFC is a binary classifier."
        y_enc = np.where(y == self.classes_[0], -1.0, 1.0)

        # --- polar coordinates (if enabled) ----------------------------------
        if self.use_polar:
            self.center_ = X.mean(axis=0)
            X = self._to_polar(X)

        # --- normalisation ---------------------------------------------------
        self.scaler_ = StandardScaler()
        X_np = self.scaler_.fit_transform(X).astype(np.float32)

        X_t = torch.tensor(X_np, device=dev)
        y_t = torch.tensor(y_enc, dtype=torch.float32, device=dev)

        # --- learn coordinate transformation (if enabled) --------------------
        if self.learn_coords:
            self.A_ = self._learn_coordinates(X_np, y_t)
            X_t = X_t @ self.A_
        else:
            self.A_ = None

        # --- stopping controller (builds G_0 here) --------------------------
        ctrl = StoppingController(
            X_0     = X_t,
            k       = self.k_neighbors,
            tau     = self.tau,
            delta   = self.delta,
            eta_min = self.eta_min,
        )

        # --- cascade ---------------------------------------------------------
        self.layers_: list[DEFCLayer] = []
        self.etas_:   list[float]     = []

        X_k  = X_t
        eta_k = self.eta

        for k in range(self.max_layers):

            # whether to refresh interior prototypes this layer
            refresh_in = (k % self.update_in_every == 0)

            layer = DEFCLayer(
                n_in         = self.n_in if refresh_in else 1,  # min 1 to avoid empty k-means
                n_boundary   = self.n_boundary,
                gamma        = self.gamma,
                lam          = self.lam,
                k_neighbors  = self.k_neighbors,
                random_state = self.random_state + k,
                use_radial   = self.use_radial,
                lambda_rad   = self.lambda_rad,
            )
            
            layer.fit(X_k, y_t)
            
            X_next_full = layer(X_k, eta=eta_k)
            
            if self.use_latent_state:
                # Apply memory inertia: dampen the update
                # X_{k+1} = X_k + β*(X_next - X_k) = (1-β)*X_k + β*X_next
                X_next = X_k + self.memory_inertia * (X_next_full - X_k)
                state_for_stopping = X_next
            else:
                # Standard: full update
                X_next = X_next_full
                state_for_stopping = X_next

            # stopping check
            stop, eta_k = ctrl.step(state_for_stopping, y_t, eta_k)

            self.layers_.append(layer)
            self.etas_.append(eta_k)

            X_k = X_next

            if stop:
                break

        self.n_layers_ = len(self.layers_)

        # --- final RBF SVM ------------------------------------------------
        X_L = X_k
        X_L_np = X_L.detach().cpu().numpy()
        self.svm_ = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale"
        )
        self.svm_.fit(X_L_np, y_enc)

        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned cascade deformation to new data.

        Each layer uses the prototypes and weights learned during fit().
        No refitting occurs — this is a pure forward pass.

        Args:
            X: (M, d) new data points (raw, not normalised).

        Returns:
            Z_L: (M, d) deformed data in the final space.
        """
        dev = torch.device(self.device)
        
        if self.use_polar:
            X = self._to_polar(X)
        
        X_np = self.scaler_.transform(X).astype(np.float32)
        X_t  = torch.tensor(X_np, device=dev)
        
        if self.A_ is not None:
            X_t = X_t @ self.A_

        with torch.no_grad():
            if self.use_latent_state:
                # Apply inertia-damped cascade
                state = X_t.clone()
                for layer, eta in zip(self.layers_, self.etas_):
                    next_full = layer(state, eta=eta)
                    state = state + self.memory_inertia * (next_full - state)
                return state.cpu().numpy()
            else:
                # Standard cascade
                state = X_t
                for layer, eta in zip(self.layers_, self.etas_):
                    state = layer(state, eta=eta)
                return state.cpu().numpy()

    # ------------------------------------------------------------------
    # predict / predict_proba / score
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.

        Pipeline: X → normalise → cascade deform → linear SVM.
        """
        X_L   = self.transform(X)
        y_enc = self.svm_.predict(X_L)                    # {-1, +1}
        # map back to original labels
        return np.where(y_enc == -1, self.classes_[0], self.classes_[1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy on (X, y).  Inherited from ClassifierMixin
        but explicitly overridden here for clarity.
        """
        return float((self.predict(X) == y).mean())

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def cascade_summary(self) -> dict:
        """
        Return a summary dict useful for paper tables / logging.

        Keys: n_layers, n_prototypes_per_layer, final_eta.
        """
        return {
            "n_layers":   self.n_layers_,
            "n_protos":   [
                (l.C.shape[0] if l.C is not None else 0)
                for l in self.layers_
            ],
            "final_eta":  self.etas_[-1] if self.etas_ else self.eta,
        }
