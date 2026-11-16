"""Evaluation metrics for the three hypotheses (paper, Section 6)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


def _centroids(u, y):
    classes = np.unique(y)
    return classes, normalize(np.stack([u[y == c].mean(0) for c in classes]))


# --------------------------------------------------------------------------- H1
def angular_organisation(z, y):
    """Intra-class cos: mean cosine between a sample and its class centroid.
    Inter-class cos: mean cosine between distinct class centroids.

    With C classes, inter-class cos is bounded below by -1/(C-1) (regular simplex).
    """
    u = normalize(z)
    classes, cents = _centroids(u, y)
    intra = np.mean([(u[y == c] @ cents[i]).mean() for i, c in enumerate(classes)])
    gram = cents @ cents.T
    inter = gram[~np.eye(len(classes), dtype=bool)].mean()
    return {"intra_cos": float(intra), "inter_cos": float(inter),
            "cos_gap": float(intra - inter), "simplex_bound": -1.0 / (len(classes) - 1)}


def pairwise_cosines(z, y, n_pairs=6000, seed=0):
    """Sampled intra-/inter-class pairwise cosines, for the distribution plot (Fig. 7)."""
    rng = np.random.default_rng(seed)
    u = normalize(z)
    i, j = rng.integers(0, len(y), (2, n_pairs * 20))
    keep = i != j
    i, j = i[keep], j[keep]
    cos = np.sum(u[i] * u[j], axis=1)
    same = y[i] == y[j]
    return cos[same][:n_pairs], cos[~same][:n_pairs]


# --------------------------------------------------------------------------- H2
def radial_invariance(z_train, y_train, z_test, y_test, gammas):
    """Linear probe trained on unscaled z, evaluated on gamma * z (Table 2, Fig. 8)."""
    probe = LogisticRegression(max_iter=400).fit(z_train, y_train)
    return np.array([(probe.predict(g * z_test) == y_test).mean() for g in gammas])


# --------------------------------------------------------------------------- H3
def mi_radius_class(z, y, cv=5):
    """I(||z||; class) in nats, as H(y) - cross-validated log-loss of a probe on ||z||."""
    r = np.linalg.norm(z, axis=1, keepdims=True)
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_y = -np.sum(p * np.log(p))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    proba = cross_val_predict(clf, r, y, cv=cv, method="predict_proba")
    return float(max(h_y - log_loss(y, proba, labels=np.unique(y)), 0.0))


def fit_image_pca(x_train, n_components=32):
    return PCA(n_components, svd_solver="randomized", random_state=0).fit(
        x_train.reshape(len(x_train), -1))


def radial_feature_distance(decode, z, image_pca, gammas, x_ref):
    """Mean distance (in image-PCA space) between x and decode(gamma * z), averaged over gammas."""
    ref = image_pca.transform(x_ref.reshape(len(x_ref), -1))
    dists = []
    for g in gammas:
        x_g = decode(g * z)
        feat = image_pca.transform(x_g.reshape(len(x_g), -1))
        dists.append(np.linalg.norm(feat - ref, axis=1).mean())
    return float(np.mean(dists))
