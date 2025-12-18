"""Regularized logistic regression with soft (probabilistic) targets.

`sklearn.linear_model.LogisticRegression` expects hard class labels, but the
comparisons in this pipeline carry a preference *probability* `y in [0, 1]`
(needed for datasets like HelpSteer2, where "preference" is derived from a
continuous helpfulness-score gap rather than a binary vote). This module
implements binary-cross-entropy logistic regression directly via L-BFGS so
soft labels work without a synthetic hard/soft split.

Shared by the Bradley-Terry baseline (features = embedding of one response,
fit on differenced feature rows) and the Diff Pairwise-Baseline model
(features = difference vector plus a per-user bias, via one-hot columns).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    reg: float | np.ndarray = 1e-3,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    """Minimize mean BCE(sigmoid(X @ w), y) + reg * ||w||^2 (no intercept).

    `reg` may be a scalar (applied uniformly) or a per-feature array, e.g. to
    leave one-hot bias columns unregularized while penalizing the shared
    embedding weights.
    """
    n, d = X.shape
    reg_vec = np.full(d, reg, dtype=np.float64) if np.isscalar(reg) else np.asarray(reg)
    w0 = np.zeros(d) if w0 is None else w0

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        p = _sigmoid(z)
        eps = 1e-9
        bce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss = bce.mean() + np.sum(reg_vec * w**2)
        grad = X.T @ (p - y) / n + 2 * reg_vec * w
        return loss, grad

    result = minimize(loss_and_grad, w0, jac=True, method="L-BFGS-B")
    return result.x