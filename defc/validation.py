"""
validation.py — Rigorous validation and ablation studies.

Tests:
  1. PCA analysis: is linear separability after fusion just from concatenation?
  2. Ablation: contribution of each geometric component
  3. Cascade interaction: do layers actually use cross-geometry information?
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons, make_gaussian_quantiles, make_blobs
from defc import DEFCEarlyFusion, DEFCClassifier


def test_pca_analysis(name, X, y):
    """
    Test if the improvement is just from concatenation trick.
    
    If PCA on fused representation gives good separability,
    it suggests the cascade may not be doing much work.
    """
    y_binary = np.where(y == np.unique(y)[0], -1, 1)
    
    # Create fusion features manually
    from defc.early_fusion import DEFCEarlyFusion
    clf_ef = DEFCEarlyFusion(max_layers=1, eta=0.1, n_in=2, n_boundary=2, 
                              random_state=42, verbose=0)
    
    # Just fit to get the transformation, but use early/simple version
    import torch
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Get representations
    X_cart = X_scaled
    center = X.mean(axis=0)
    X_centered = X - center
    r = np.linalg.norm(X_centered, axis=1, keepdims=True)
    theta = np.arctan2(X_centered[:, 1], X_centered[:, 0]).reshape(-1, 1)
    X_polar = np.hstack([r, theta])
    X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
    
    # Learn transformation
    dev = torch.device("cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_binary, dtype=torch.float32, device=dev)
    
    d = X_scaled.shape[1]
    A = torch.eye(d, device=dev, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([A], lr=0.2)
    
    for _ in range(50):
        X_proj = X_t @ A
        class_0 = X_proj[y_t == -1]
        class_1 = X_proj[y_t == 1]
        if len(class_0) > 0 and len(class_1) > 0:
            mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
            between = torch.sum((mean_0 - mean_1) ** 2)
            within = (class_0.var(dim=0).sum() + class_1.var(dim=0).sum()) / 2 + 1e-8
            loss = -between / within
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    X_learn = (X_scaled @ A.detach().cpu().numpy()).astype(np.float32)
    X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)
    
    # Fused representation
    X_fused = np.hstack([X_cart, X_polar_scaled, X_learn_scaled])
    
    # Test 1: Direct SVM on fused (no cascade)
    svm_fused = SVC(kernel="rbf", C=2.0, gamma="scale")
    svm_fused.fit(X_fused, y_binary)
    acc_svm_fused = float((svm_fused.predict(X_fused) == y_binary).mean())
    
    # Test 2: PCA + SVM on fused (project to 2D)
    pca = PCA(n_components=2)
    X_fused_pca = pca.fit_transform(X_fused)
    svm_pca = SVC(kernel="rbf", C=2.0, gamma="scale")
    svm_pca.fit(X_fused_pca, y_binary)
    acc_pca_2d = float((svm_pca.predict(X_fused_pca) == y_binary).mean())
    
    # Test 3: Baseline single geometry
    svm_single = SVC(kernel="rbf", C=2.0, gamma="scale")
    svm_single.fit(X_scaled, y_binary)
    acc_single = float((svm_single.predict(X_scaled) == y_binary).mean())
    
    # Test 4: Early Fusion full
    clf_ef = DEFCEarlyFusion(max_layers=5, eta=0.3, n_in=3, n_boundary=3, 
                              random_state=42, verbose=0)
    clf_ef.fit(X, y_binary)
    acc_ef = clf_ef.score(X, y_binary)
    
    return {
        "name": name,
        "svm_single": acc_single,
        "svm_fused": acc_svm_fused,
        "svm_pca_2d": acc_pca_2d,
        "early_fusion": acc_ef,
        "pca_variance": pca.explained_variance_ratio_[:2].sum(),
    }


def test_ablation(name, X, y):
    """
    Ablation: test each geometry individually and in combinations.
    """
    y_binary = np.where(y == np.unique(y)[0], -1, 1)
    
    from sklearn.preprocessing import StandardScaler
    import torch
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Get all three representations
    X_cart = X_scaled
    
    center = X.mean(axis=0)
    X_centered = X - center
    r = np.linalg.norm(X_centered, axis=1, keepdims=True)
    theta = np.arctan2(X_centered[:, 1], X_centered[:, 0]).reshape(-1, 1)
    X_polar = np.hstack([r, theta])
    X_polar_scaled = (X_polar - X_polar.mean(axis=0)) / (X_polar.std(axis=0) + 1e-10)
    
    # Learn transformation
    dev = torch.device("cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_binary, dtype=torch.float32, device=dev)
    
    d = X_scaled.shape[1]
    A = torch.eye(d, device=dev, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([A], lr=0.2)
    
    for _ in range(50):
        X_proj = X_t @ A
        class_0 = X_proj[y_t == -1]
        class_1 = X_proj[y_t == 1]
        if len(class_0) > 0 and len(class_1) > 0:
            mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
            between = torch.sum((mean_0 - mean_1) ** 2)
            within = (class_0.var(dim=0).sum() + class_1.var(dim=0).sum()) / 2 + 1e-8
            loss = -between / within
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    X_learn = (X_scaled @ A.detach().cpu().numpy()).astype(np.float32)
    X_learn_scaled = (X_learn - X_learn.mean(axis=0)) / (X_learn.std(axis=0) + 1e-10)
    
    # Test combinations
    combinations = {
        "cartesian_only": X_cart,
        "polar_only": X_polar_scaled,
        "learned_only": X_learn_scaled,
        "cart+polar": np.hstack([X_cart, X_polar_scaled]),
        "cart+learned": np.hstack([X_cart, X_learn_scaled]),
        "polar+learned": np.hstack([X_polar_scaled, X_learn_scaled]),
        "all_three": np.hstack([X_cart, X_polar_scaled, X_learn_scaled]),
    }
    
    results = {}
    for combo_name, X_combo in combinations.items():
        svm = SVC(kernel="rbf", C=2.0, gamma="scale")
        svm.fit(X_combo, y_binary)
        acc = float((svm.predict(X_combo) == y_binary).mean())
        results[combo_name] = acc
    
    return {
        "name": name,
        "ablations": results,
    }


def print_pca_results(results_list):
    """Print PCA analysis results."""
    print("\n" + "="*100)
    print("PCA ANALYSIS: Is improvement from concatenation or cascade?")
    print("="*100)
    print(f"{'Dataset':<25} | {'SVM Single':>12} {'SVM Fused':>12} {'SVM+PCA(2D)':>12} {'EarlyFusion':>12} | PCA Var")
    print("-"*100)
    
    for res in results_list:
        name = res["name"]
        single = res["svm_single"]
        fused = res["svm_fused"]
        pca_2d = res["svm_pca_2d"]
        ef = res["early_fusion"]
        var = res["pca_variance"]
        
        print(f"{name:<25} | {single:>12.4f} {fused:>12.4f} {pca_2d:>12.4f} {ef:>12.4f} | {var:.3f}")
    
    avg_single = np.mean([r["svm_single"] for r in results_list])
    avg_fused = np.mean([r["svm_fused"] for r in results_list])
    avg_pca = np.mean([r["svm_pca_2d"] for r in results_list])
    avg_ef = np.mean([r["early_fusion"] for r in results_list])
    
    print("-"*100)
    print(f"{'AVERAGE':<25} | {avg_single:>12.4f} {avg_fused:>12.4f} {avg_pca:>12.4f} {avg_ef:>12.4f}")
    print("\n💡 Interpretation:")
    print(f"   SVM Fused vs SVM Single: +{(avg_fused - avg_single)*100:.1f}% (concatenation effect)")
    print(f"   SVM+PCA vs SVM Single:   +{(avg_pca - avg_single)*100:.1f}% (linear separability)")
    print(f"   EarlyFusion vs SVM:      +{(avg_ef - avg_fused)*100:.1f}% (cascade effect)")


def print_ablation_results(results_list):
    """Print ablation results."""
    print("\n" + "="*100)
    print("ABLATION ANALYSIS: Contribution of each geometry")
    print("="*100)
    
    for res in results_list:
        print(f"\n{res['name']}:")
        ablations = res["ablations"]
        for combo_name, acc in sorted(ablations.items(), key=lambda x: -x[1]):
            print(f"  {combo_name:20s}: {acc:.4f}")


if __name__ == "__main__":
    datasets = [
        ("Circles", make_circles(n_samples=200, noise=0.05, random_state=42)),
        ("Moons", make_moons(n_samples=200, noise=0.1, random_state=42)),
        ("Gaussian quantiles", make_gaussian_quantiles(n_samples=200, n_features=2, random_state=42)),
        ("Blobs (overlap)", make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=1.5, random_state=42)),
    ]
    
    pca_results = []
    ablation_results = []
    
    for name, (X, y) in datasets:
        pca_res = test_pca_analysis(name, X, y)
        pca_results.append(pca_res)
        
        abl_res = test_ablation(name, X, y)
        ablation_results.append(abl_res)
    
    print_pca_results(pca_results)
    print_ablation_results(ablation_results)
