"""
ensemble.py — DEFCEnsemble: ensemble of geometric coordinate spaces.

Trains three DEFC variants in parallel:
  1. Cartesian (baseline)
  2. Polar (radial coordinates)
  3. Learnable (Fisher-optimized linear transformation)

Selects the variant with maximum SVM margin after deformation.

Reference: DEFC geometry selection via margin maximization.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .model import DEFCClassifier





class DEFCEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of DEFC classifiers with different geometric coordinate systems.
    
    Trains three variants and selects the one with maximum SVM margin.
    
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
        Verbosity level (0=silent, 1=show selections).
    """

    def __init__(
        self,
        max_layers: int = 20,
        eta: float = 0.5,
        n_in: int = 5,
        n_boundary: int = 5,
        gamma: float = 1.0,
        lam: float = 1e-3,
        k_neighbors: int = 10,
        tau: float = 0.7,
        delta: float = 1e-3,
        eta_min: float = 1e-4,
        update_in_every: int = 3,
        device: str = "cpu",
        random_state: int = 0,
        verbose: int = 1,
    ):
        self.max_layers = max_layers
        self.eta = eta
        self.n_in = n_in
        self.n_boundary = n_boundary
        self.gamma = gamma
        self.lam = lam
        self.k_neighbors = k_neighbors
        self.tau = tau
        self.delta = delta
        self.eta_min = eta_min
        self.update_in_every = update_in_every
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # -----------------------------
    # safe scoring helpers
    # -----------------------------
    def _margin_proxy(self, clf, X, y):
        """
        Stable proxy for "decision strength".
        Works for ANY classifier (SVM / RBF / etc).
        """
        # distance to decision boundary via predict_proba or decision_function
        if hasattr(clf.svm_, "decision_function"):
            scores = clf.svm_.decision_function(clf.transform(X))
            return np.mean(np.abs(scores))
        else:
            # fallback: confidence via predictions
            preds = clf.predict(X)
            return float(np.mean(preds == y))

    # -----------------------------
    # fit
    # -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):

        clf_cartesian = DEFCClassifier(
            max_layers=self.max_layers,
            eta=self.eta,
            n_in=self.n_in,
            n_boundary=self.n_boundary,
            gamma=self.gamma,
            lam=self.lam,
            k_neighbors=self.k_neighbors,
            tau=self.tau,
            delta=self.delta,
            eta_min=self.eta_min,
            update_in_every=self.update_in_every,
            device=self.device,
            random_state=self.random_state,
            use_polar=False,
            learn_coords=False,
        ).fit(X, y)

        clf_polar = DEFCClassifier(
            max_layers=self.max_layers,
            eta=self.eta,
            n_in=self.n_in,
            n_boundary=self.n_boundary,
            gamma=self.gamma,
            lam=self.lam,
            k_neighbors=self.k_neighbors,
            tau=self.tau,
            delta=self.delta,
            eta_min=self.eta_min,
            update_in_every=self.update_in_every,
            device=self.device,
            random_state=self.random_state,
            use_polar=True,
            learn_coords=False,
        ).fit(X, y)

        clf_learnable = DEFCClassifier(
            max_layers=self.max_layers,
            eta=self.eta,
            n_in=self.n_in,
            n_boundary=self.n_boundary,
            gamma=self.gamma,
            lam=self.lam,
            k_neighbors=self.k_neighbors,
            tau=self.tau,
            delta=self.delta,
            eta_min=self.eta_min,
            update_in_every=self.update_in_every,
            device=self.device,
            random_state=self.random_state,
            use_polar=False,
            learn_coords=True,
        ).fit(X, y)

        # -----------------------------
        # evaluate properly
        # -----------------------------
        accs = {
            "cartesian": clf_cartesian.score(X, y),
            "polar": clf_polar.score(X, y),
            "learnable": clf_learnable.score(X, y),
        }

        scores = {
            "cartesian": self._margin_proxy(clf_cartesian, X, y),
            "polar": self._margin_proxy(clf_polar, X, y),
            "learnable": self._margin_proxy(clf_learnable, X, y),
        }

        best_acc = max(accs.values())

        candidates = [
            k for k in accs
            if accs[k] >= best_acc - 1e-6
        ]

        self.selected_geometry_ = max(candidates, key=lambda k: scores[k])

        # store
        self.clf_cartesian_ = clf_cartesian
        self.clf_polar_ = clf_polar
        self.clf_learnable_ = clf_learnable

        self.clf_ = {
            "cartesian": clf_cartesian,
            "polar": clf_polar,
            "learnable": clf_learnable
        }[self.selected_geometry_]

        self.classes_ = self.clf_.classes_

        if self.verbose:
            print("\n[DEFCEnsemble]")
            for k in accs:
                print(
                    f"{k:12s} acc={accs[k]:.4f}  score={scores[k]:.4f}"
                    + (" ← SELECTED" if k == self.selected_geometry_ else "")
                )

        return self

    # -----------------------------
    # inference
    # -----------------------------
    def transform(self, X):
        return self.clf_.transform(X)

    def predict(self, X):
        return self.clf_.predict(X)

    def score(self, X, y):
        return self.clf_.score(X, y)

    def cascade_summary(self):
        return self.clf_.cascade_summary()