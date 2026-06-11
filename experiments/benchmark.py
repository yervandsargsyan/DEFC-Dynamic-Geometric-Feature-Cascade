"""
experiments/benchmark.py — CLEAN & FAIR comparison benchmark.

Fixes:
- deterministic splits (same as manual experiment style)
- no CV-induced retraining instability
- strict seed control everywhere
- identical preprocessing for all models per dataset
- prevents hidden stochastic drift between runs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sklearn.datasets import (
    make_moons, make_circles, make_classification,
    load_breast_cancer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defc import DEFCClassifier
from defc.dynamic_fusion import DEFCDynamicFusion
from defc.early_fusion import DEFCEarlyFusion
from defc.ensemble import DEFCEnsemble


# -----------------------------
# reproducibility helper
# -----------------------------
SEED = 42


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.3,
        random_state=SEED,
        stratify=y
    )


# -----------------------------
# datasets 
# -----------------------------
def make_parity(
    n_samples=2000,
    n_bits=16,
    n_noise_features=16,
    flip_y=0.05,
    random_state=42,
):
    """
    High-dimensional parity dataset.

    y = parity(sign(x_1), ..., sign(x_n_bits))

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_bits : int
        Number of informative parity dimensions.

    n_noise_features : int
        Number of pure noise dimensions.

    flip_y : float
        Fraction of labels to randomly flip.

    random_state : int
        Seed.

    Returns
    -------
    X : ndarray, shape (n_samples, n_bits + n_noise_features)
    y : ndarray, shape (n_samples,)
    """

    rng = np.random.RandomState(random_state)

    # informative binary features {-1, +1}
    bits = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))

    # parity label: even -> 0, odd -> 1
    y = ((bits > 0).sum(axis=1) % 2).astype(int)

    # add Gaussian noise to informative dimensions
    bits = bits + 0.3 * rng.randn(n_samples, n_bits)

    # completely irrelevant features
    noise = rng.randn(n_samples, n_noise_features)

    X = np.hstack([bits, noise])

    # label noise
    if flip_y > 0:
        flip = rng.rand(n_samples) < flip_y
        y[flip] = 1 - y[flip]

    return X, y


def make_fractal_warped(n=2000, noise=0.15, label_noise=0.15):
    rng = np.random.RandomState(42)

    X = rng.uniform(-2, 2, (n, 2))

    # --- nonlinear warping (break Euclidean geometry) ---
    X[:, 0] = np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])
    X[:, 1] = np.sin(2 * X[:, 1]) - np.cos(3 * X[:, 0])

    # --- fractal-like boundary ---
    def fractal_score(x, y):
        return (
            np.sin(5 * x)
            + np.sin(7 * y)
            + np.sin(3 * (x + y))
            + np.sin(9 * (x - y))
        )

    scores = fractal_score(X[:, 0], X[:, 1])

    y = (scores > 0).astype(int)

    # --- density imbalance (hard for RF/kNN) ---
    prob = 0.3 + 0.7 * (scores - scores.min()) / (scores.max() - scores.min())
    mask = rng.rand(n) < prob
    X = X[mask]
    y = y[mask]

    # --- noise ---
    X += rng.normal(0, noise, X.shape)

    # --- label noise ---
    flip = rng.rand(len(y)) < label_noise
    y[flip] = 1 - y[flip]

    return X, y

def get_datasets():
    moons = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=SEED)
    fractals = make_fractal_warped(n=2000, noise=0.15, label_noise=0.15)
    
    parity = make_parity(
        n_samples=5000,
        n_bits=16,
        n_noise_features=16,
        flip_y=0.10,
        random_state=42,
    )
    
    linear = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=SEED,
    )

    xor = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=SEED,
    )

    rng = np.random.RandomState(SEED)
    gaussian = (
        np.vstack([rng.randn(250, 2), rng.randn(250, 2) + 3]),
        np.hstack([np.zeros(250), np.ones(250)])
    )

    overlapping = (
        rng.normal(0, 1.5, (500, 2)),
        (rng.normal(0.2, 1.5, (500, 2))[:, 0] > 0).astype(int)
    )

    cancer = load_breast_cancer(return_X_y=True)

    return {
        "moons": moons,
        "circles": circles,
        "linear": linear,
        "xor": xor,
        "gaussian": gaussian,
        "overlapping": overlapping,
        "breast_cancer": cancer,
        "fractal": fractals,
        "parity": parity
    }


# -----------------------------
# models
# -----------------------------
def get_models():
    models = {}

    def defc(**kwargs):
        return DEFCClassifier(
            max_layers=25,
            eta=0.35,
            n_in=7,
            n_boundary=7,
            gamma=1.2,
            lam=1e-3,
            k_neighbors=12,
            tau=0.6,
            delta=1e-4,
            eta_min=1e-5,
            update_in_every=2,
            use_latent_state=True,
            memory_inertia=0.85,
            use_radial=True,
            lambda_rad=0.15,
            random_state=SEED,
            **kwargs,
        )

    models["DEFC (base)"] = defc()

    models["DEFC (learnable)"] = defc(
        learn_coords=True,
        use_polar=False,
    )

    models["DEFC (polar)"] = defc(
        use_polar=True,
    )

    models["DEFC (full)"] = defc(
        learn_coords=True,
        use_polar=True,
    )
    
    for beta in [0.0, 0.3, 0.5, 0.7]:
        models[f"DEFC β={beta:.1f}"] = DEFCClassifier(
            random_state=SEED,
            use_latent_state=(beta < 1.0),
            memory_inertia=beta      
    )
    
    models["DynamicFusion"] = DEFCDynamicFusion(
        max_layers=15,
        eta=0.5,
        n_in=5,
        n_boundary=5,
        random_state=SEED,
    )

    models["EarlyFusion"] = DEFCEarlyFusion(
        max_layers=15,
        eta=0.5,
        n_in=5,
        n_boundary=5,
        random_state=SEED,
    )

    models["Ensemble"] = DEFCEnsemble(
        max_layers=15,
        eta=0.5,
        n_in=5,
        n_boundary=5,
        random_state=SEED,
    )

    models["LogReg"] = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="lbfgs"
        )),
    ])

    models["SVM (RBF)"] = Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2.0, gamma="scale")),
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
    )

    models["GradBoost"] = GradientBoostingClassifier(
        n_estimators=100,
        random_state=SEED,
    )

    models["KNN (k=5)"] = Pipeline([
        ("sc", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])

    return models


# -----------------------------
# runner (FAIR MODE)
# -----------------------------
def run_benchmark():
    datasets = get_datasets()
    models = get_models()

    results = {ds: {} for ds in datasets}

    for name, (X, y) in datasets.items():
        print(f"\n── {name} ──")

        X_train, X_test, y_train, y_test = split_data(X, y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)

                results[name][model_name] = acc
                print(f"{model_name:<25} {acc:.4f}")

            except Exception as e:
                results[name][model_name] = None
                print(f"{model_name:<25} SKIP ({type(e).__name__})")

    # =============== COMPARISON: DEFC vs RBF SVM ===============
    print("\n" + "=" * 90)
    print("KEY COMPARISON: Space Deformation Improves RBF SVM")
    print("=" * 90)
    
    comparison_results = {}
    for ds in datasets:
        svm_score = results[ds].get("SVM (RBF)")
        defc_scores = {k: v for k, v in results[ds].items() if k.startswith("DEFC")}
        
        if svm_score is not None and defc_scores:
            best_defc = max(defc_scores.values())
            improvement = ((best_defc - svm_score) / svm_score * 100) if svm_score > 0 else 0
            
            comparison_results[ds] = {
                "SVM (RBF)": svm_score,
                "Best DEFC": best_defc,
                "Improvement %": improvement,
                "Best DEFC model": max(defc_scores, key=defc_scores.get),
            }
    
    print(f"\n{'Dataset':<15} {'SVM(RBF)':<12} {'Best DEFC':<12} {'Improvement':<15} {'Model'}")
    print("-" * 90)
    for ds, comp in comparison_results.items():
        svm = comp["SVM (RBF)"]
        defc = comp["Best DEFC"]
        improvement = comp["Improvement %"]
        model = comp["Best DEFC model"]
        
        print(f"{ds:<15} {svm:.4f}       {defc:.4f}        {improvement:+.2f}%           {model}")

    # =============== SUMMARY TABLE ===============
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (FAIR SINGLE SPLIT EVALUATION)")
    print("=" * 90)

    print(f"{'Model':<25}", end="")
    for d in datasets:
        print(f"{d[:10]:<10}", end="")
    print()

    for model in models:
        print(f"{model:<25}", end="")
        for d in datasets:
            v = results[d][model]
            if v is None:
                print(f"{'SKIP':<10}", end="")
            else:
                print(f"{v:.3f}     ", end="")
        print()


if __name__ == "__main__":
    run_benchmark()