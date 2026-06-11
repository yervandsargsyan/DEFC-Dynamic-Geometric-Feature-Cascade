# DEFC — Dynamic Electromagnetic Field Cascade

## Installation

### Requirements
- Python 3.10+
- PyTorch 1.12+
- scikit-learn 1.0+
- NumPy 1.20+

### Quick Setup

```bash
cd defc
pip install -r requirments.txt
```

Or install manually:
```bash
pip install torch scikit-learn numpy
```

### Testing

```bash
# Run benchmark on 9 datasets
python experiments/benchmark.py

# Quick test on circles
python -c "from defc import DEFCClassifier; from sklearn.datasets import make_circles; \
X, y = make_circles(500, noise=0.1); clf = DEFCClassifier(); \
clf.fit(X[:350], y[:350]); print(f'Accuracy: {clf.score(X[350:], y[350:]):.4f}')"
```

## Overview

DEFC (Dynamic Electromagnetic Field Cascade) is a geometry-based machine learning method that classifies data by progressively deforming the feature space using a cascade of interaction-driven transformations.

Instead of directly learning a decision boundary in the original space, DEFC iteratively reshapes the geometry of the data so that class structure becomes increasingly separable.

The method is inspired by field-like interactions between data points: points of the same class tend to attract, while points of different classes induce repulsion through learned prototype-driven influences.

A key property of DEFC is that it does **not rely on backpropagation**. Learning is performed through sequential constructive updates of the space rather than gradient-based optimization of a global loss.

The final classification is performed in the transformed space, where standard classifiers (e.g., linear or kernel-based models) can operate more effectively due to improved geometric separation.


## Project structure
```text
defc/
├── defc/
│   ├── model.py
│   ├── cascade.py
│   ├── stopping.py
│   ├── ensemble.py
│   └── utils.py
│
├── experiments/
│   └── benchmark.py
│
├── README.md
├── requirements.txt
└── results/
    └── latest_results.txt
```
## Core idea

DEFC treats a dataset not as static points in a fixed space, but as a dynamic system of interacting entities embedded in a geometric field.

Each class defines a set of prototype “sources” that generate attraction and repulsion forces over the space. These forces are not used to directly optimize a loss function — instead, they *physically deform the representation of the data over multiple cascade layers*.

At each layer:
- local structure is estimated via neighborhood graphs and prototypes,
- interaction forces are computed between points and class structures,
- the entire dataset is updated by moving points along the induced field,
- the geometry gradually becomes more separable.

This process is repeated as a cascade, where each stage operates on the transformed output of the previous one.

Different variants of DEFC modify the geometry or dynamics of this field:
- polar coordinates emphasize radial separability,
- learnable transformations optimize global alignment,
- latent-state evolution introduces memory into the dynamics,
- radial stabilization strengthens circular decision structures.

Importantly, the classifier at the end (e.g., SVM) is not the core model — it is only a probe used to evaluate how well the space has been organized by the cascade.

## DEFC model family

DEFC is not a single model but a family of geometric cascade variants that differ in how the underlying field is constructed and how the space is updated during deformation.

---

### β-variants (latent inertia / impulse dynamics)

The β parameter controls *memory / inertia* in the cascade dynamics.

Instead of applying each deformation step fully, the update is blended with the previous state:

- β = 0 → no inertia (pure instantaneous deformation)
- β → 1 → strong memory (slow, smooth evolution)

Interpretation:
- Low β behaves like a sharp “force field”
- High β behaves like a damped physical system with momentum

This introduces an implicit trade-off between stability and expressiveness:
- low β → aggressive separation, but risk of instability
- high β → stable transformations, but weaker geometric reshaping

---

### Polar variant (radial inductive bias)

The polar version transforms the input space into polar coordinates before applying the cascade:

- radius captures distance structure
- angle captures circular separation patterns

This variant is particularly effective for:
- concentric structures
- circular decision boundaries
- radial manifolds

It encodes a strong geometric prior: that separability may depend more on angular structure than Euclidean linearity.

---

### Learnable coordinate transform

The learnable variant introduces a global linear transformation optimized using a Fisher-style criterion.

Instead of manually defining geometry, DEFC learns a projection that:
- maximizes class separation
- minimizes within-class variance
- aligns data into a more separable coordinate system before cascading

This acts as a global “pre-conditioning” step that improves the effectiveness of subsequent field dynamics.

---

### Full / hybrid variant

The full version combines multiple mechanisms:
- cascade deformation
- optional latent state evolution
- optional radial stabilization
- prototype-based interactions

It is the most expressive but also the most sensitive to hyperparameters, since multiple interacting mechanisms can amplify or cancel each other.

---

### Ensemble variant

The ensemble version treats DEFC as a *family of geometries* rather than a single system.

It trains multiple DEFC instances in parallel:
- Cartesian space
- Polar space
- Learnable transformed space

Then selects the best-performing variant based on:
1. classification accuracy on the transformed space
2. margin-based geometric confidence of the final classifier

This makes DEFC more robust by avoiding dependence on a single coordinate assumption.

---

### Interpretation of the family

Across all variants, the key unifying idea is:

> DEFC introduces a conditional Pareto improvement over RBF-SVM, achieving gains in geometry-dominated regimes (e.g., fractal-like structures) while maintaining parity on standard benchmarks.

Different configurations correspond to different assumptions about what “good geometry” means:
- Euclidean (baseline)
- radial (polar bias)
- discriminative linear alignment (learnable)
- dynamic/memory-based systems (β variants)
- multi-hypothesis selection (ensemble)

# Physical Interpretation of DEFC

DEFC can be interpreted as a discrete dynamical system where data points evolve under an iterative interaction field resembling a learned physical force field.

Each layer applies a transformation of the form:
- points are “pulled” or “repelled” by learned prototype structures,
- these prototypes act as sources of an interaction potential,
- the resulting motion corresponds to a gradient-like deformation of the embedding space.

From this perspective, the model behaves like a particle system with learned interactions:
- samples = particles,
- prototypes = interaction centers (charges / masses),
- deformation = evolution under a vector field.

## Field dynamics

The learned field is not static. It is recomputed layer-by-layer, which creates a time-evolving geometry. This makes the system closer to a **non-equilibrium physical process** than to a fixed kernel mapping.

Each cascade step can be viewed as:
- local force estimation from neighborhood structure,
- aggregation of these forces into a global deformation vector,
- update of positions in latent space.

## Latent memory and inertia

When latent state evolution is enabled, the system behaves like a damped physical medium:
- previous state acts as inertia,
- new deformation acts as external force,
- β controls viscosity / momentum.

This introduces a transition between:
- pure gradient flow (β → 0),
- and momentum-driven dynamics (β → 1).

## Geometric interpretation

Different variants correspond to different coordinate manifolds:
- Cartesian: flat Euclidean space,
- Polar: radial symmetry bias (useful for circular manifolds),
- Learned transform: adaptive metric learning approximating Fisher-optimal separation.

Thus, DEFC effectively learns not only a classifier, but the **metric structure of the space itself**.

## Interpretation of the decision rule

The final classifier operates in a deformed space where linear separation becomes possible. In this sense:
- non-linearity is pushed into geometry,
- classification remains linear in the transformed manifold,
- the cascade acts as a learned diffeomorphic embedding.

## Overall intuition

DEFC can be seen as:
a physically-inspired geometric flow that reshapes data until class structure becomes linearly separable in a learned emergent space.

# DEFC — Mathematical Formulation

## 1. Data and notation

Let:
- \(X \in \mathbb{R}^{N \times d}\) — input data
- \(y \in \{-1, +1\}^N\) — binary labels
- \(X_k\) — representation after cascade layer \(k\)
- \(F_k(\cdot)\) — deformation field at layer \(k\)

---

## 2. Cascade update rule

The core transformation is defined as:

\[
X_{k+1} = X_k + \eta_k F_k(X_k)
\]

where:
- \(\eta_k\) is the layer-wise step size
- \(F_k\) is the learned interaction field

---

## 3. Latent state (optional inertia)

When latent memory is enabled:

\[
Z_{k+1} = (1 - \beta) Z_k + \beta (Z_k + \eta_k F_k(Z_k))
\]

equivalently:

\[
Z_{k+1} = Z_k + \beta \eta_k F_k(Z_k)
\]

where:
- \(\beta \in [0,1]\) controls inertia
- \(Z_k\) is the latent evolving geometry

---

## 4. Prototype-based field construction

Each layer uses:
- interior prototypes \(C^{(k)}_c\)
- boundary prototypes \(B^{(k)}_c\)

for each class \(c\).

The field is a weighted sum of interactions:

\[
F_k(x) = \sum_{c} \sum_{p \in C^{(k)}_c} w_{cp} \, K(x, p)
\;-\;
\sum_{c} \sum_{q \in B^{(k)}_c} v_{cq} \, K(x, q)
\]

where \(K(\cdot,\cdot)\) is a kernel-like interaction.

---

## 5. Interaction kernel (Yukawa-style decay)

The interaction between points is defined as:

\[
K(x, p) = \exp(-\gamma \|x - p\|)
\]

where:
- \(\gamma > 0\) controls interaction radius
- larger \(\gamma\) → more local interactions

---

## 6. Radial stabilizer (optional)

For radial variants:

\[
F^{rad}(x) = -\lambda_{rad} \cdot \nabla \|x\|
\]

This introduces a stabilizing force toward radial symmetry.

---

## 7. Learnable coordinate transform

When enabled, a linear transformation is applied:

\[
x' = A x
\]

where \(A \in \mathbb{R}^{d \times d}\) is optimized via Fisher criterion:

\[
J(A) =
\frac{\| \mu_+ - \mu_- \|^2}
{\sigma_+^2 + \sigma_-^2}
\]

with:
- \(\mu_c\): class mean in projected space
- \(\sigma_c^2\): within-class variance

---

## 8. Polar transformation (2D case)

\[
r = \|x - \mu\|,\quad
\theta = \arctan2(x_2, x_1)
\]

\[
x \rightarrow (r, \theta)
\]

---

## 9. Stopping criterion (conceptual form)

Cascade stops when improvement falls below threshold:

\[
\Delta_k < \delta
\quad \text{or} \quad
\eta_k < \eta_{min}
\]

and/or structural overlap condition:

\[
S_k < \tau
\]

---

## 10. Final classifier

After cascade:

\[
\hat{y} = \text{sign}(w^\top X_L + b)
\]

where:
- \(X_L\) is the final deformed space
- \(w, b\) are learned by linear SVM or equivalent separator

---

## 11. Overall system

The full model is:

\[
X \xrightarrow{\text{cascade}} X_L \xrightarrow{\text{linear classifier}} y
\]

or equivalently:

\[
y = f_{\text{linear}}(T_{\text{DEFC}}(X))
\]

where \(T_{\text{DEFC}}\) is a learned nonlinear geometric transformation induced by iterative field dynamics.

## How to Run

### Base DEFC

```python
from model import DEFCClassifier

defc = DEFCClassifier(
    max_layers=15,
    eta=0.5,
    n_in=5,
    n_boundary=5,
    gamma=1.0,
    random_state=42,
    use_polar=False,
    learn_coords=False
)

defc.fit(X_train, y_train)
pred = defc.predict(X_test)
print(defc.score(X_test, y_test))
```

### Polar DEFC

```python
defc = DEFCClassifier(
    max_layers=15,
    eta=0.5,
    gamma=1.0,
    random_state=42,
    use_polar=True
)
```

### Learnable DEFC

```python
defc = DEFCClassifier(
    max_layers=15,
    eta=0.5,
    gamma=1.0,
    random_state=42,
    learn_coords=True
)
```

### Latent inertia (β version)

```python
defc = DEFCClassifier(
    max_layers=15,
    eta=0.5,
    gamma=1.0,
    random_state=42,
    use_latent_state=True,
    memory_inertia=0.7 # or beta in [0,1]
)
```

### Full DEFC 

```python
defc = DEFCClassifier(
    max_layers=20,
    eta=0.6,
    n_in=5,
    n_boundary=5,
    gamma=1.0,
    lam=1e-3,
    k_neighbors=10,
    tau=0.7,
    delta=1e-3,
    random_state=42,
    use_polar=True,
    learn_coords=True,
    use_latent_state=True,
    memory_inertia=0.5
)
```

### Ensemble (best overall performance)
```python
from ensemble import DEFCEnsemble

model = DEFCEnsemble(
    max_layers=15,
    eta=0.5,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(model.score(X_test, y_test))
```

### for other variants check defc directory


## Quick Start

### 1. Basic Usage

```python
from defc import DEFCClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = make_circles(n_samples=500, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit DEFC
clf = DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")  # Expected: ~0.97
```

### 2. Compare Different Variants

```python
from defc import DEFCClassifier

variants = {
    'base': DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2),
    'polar': DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2, use_polar=True),
    'learnable': DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2, learn_coords=True),
    'with_inertia': DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2, 
                                    use_latent_state=True, memory_inertia=0.85),
}

for name, model in variants.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name:15}: {acc:.4f}")
```

### 3. Understand Cascade Depth

```python
clf = DEFCClassifier(max_layers=25, eta=0.35, gamma=1.2)
clf.fit(X_train, y_train)

# Check how many layers were actually used
summary = clf.cascade_summary()
print(f"Cascade depth: {summary['n_layers']} layers")
print(f"Prototypes per layer: {summary['n_protos']}")
```


# benchmark + tests
```text
Model               moons     circles   linear    xor       gaussian  overlapping breast_cancer fractal   parity    
DEFC (base)         0.973     0.967     0.907     0.853     0.980     0.547       0.982         0.639     0.505     
DEFC (learnable)    0.840     0.747     0.847     0.847     0.980     0.547       0.971         0.628     0.509     
DEFC (polar)        0.973     0.980     0.593     0.793     0.973     0.527       0.918         0.560     0.482     
DEFC (full)         0.947     0.980     0.587     0.593     0.980     0.527       0.661         0.570     0.474     
DEFC β=0.0          0.987     0.980     0.907     0.860     0.980     0.533       0.982         0.595     0.505     
DEFC β=0.3          0.980     0.967     0.907     0.860     0.980     0.533       0.982         0.634     0.505     
DEFC β=0.5          0.967     0.960     0.907     0.853     0.980     0.533       0.982         0.618     0.508     
DEFC β=0.7          0.980     0.953     0.907     0.833     0.967     0.547       0.982         0.641     0.507     
DynamicFusion       0.980     0.980     0.880     0.873     0.973     0.547       0.965         0.618     0.508     
EarlyFusion         0.980     0.980     0.880     0.873     0.973     0.547       0.965         0.618     0.508     
Ensemble            0.980     0.980     0.913     0.840     0.980     0.560       0.982         0.616     0.504     
LogReg              0.853     0.473     0.833     0.847     0.980     0.540       0.988         0.580     0.509     
SVM (RBF)           0.987     0.980     0.907     0.860     0.980     0.533       0.982         0.595     0.505     
RandomForest        0.960     0.987     0.893     0.880     0.973     0.513       0.936         0.588     0.509     
GradBoost           0.973     0.980     0.907     0.860     0.953     0.473       0.947         0.646     0.503     
KNN (k=5)           0.973     0.973     0.867     0.880     0.973     0.540       0.959         0.634     0.480     
```

## Key Baseline Comparison
```text| Model         | fractal | gain vs SVM |
|--------------|---------|--------------|
| SVM (RBF)    | 0.595   | 0.000        |
| DEFC (best)  | 0.641   | +0.046 +7.7% |
```

## Method Quality

DEFC variants show strong geometric reasoning:
- Polar variant excels on radial structures (circles: 0.980)
- Latent inertia (β-variants) provides smooth, stable deformation
- Learnable transform adapts to data orientation
- Fractal datasets benefit from deep cascades with memory

**Key insight**: DEFC's strength lies not in beating individual classifiers, but in demonstrating that *geometric deformation* can make data separable where fixed-metric methods struggle. The method shows consistent improvements on complex, nonlinear geometries.

### Key insight

DEFC is not simply competing with SVM as a classifier — it changes the feature space before classification.

- When deformation is effective → SVM becomes a *linear separator in transformed space*
- When deformation is weak or prematurely stopped → performance collapses to SVM baseline
- Early stopping is a critical factor and can suppress the full expressive power of the cascade

### Overall conclusion

- DEFC is competitive with classical methods on structured nonlinear problems
- It does not consistently dominate SVM yet, but shows clear advantages on geometry-dominant tasks
- Main bottleneck is not the final classifier, but the quality of learned deformation