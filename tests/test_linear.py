import numpy as np

from rme.linear import fit_logistic


def test_fit_logistic_recovers_separating_direction():
    rng = np.random.RandomState(0)
    n, d = 500, 5
    true_w = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    X = rng.normal(size=(n, d))
    probs = 1.0 / (1.0 + np.exp(-(X @ true_w)))
    y = (rng.uniform(size=n) < probs).astype(float)

    w = fit_logistic(X, y, reg=1e-4)
    # sign and dominant dimension should match the generating direction
    assert w[0] > 0
    assert w[0] > np.abs(w[1:]).max()


def test_fit_logistic_supports_per_feature_regularization():
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    reg = np.array([1e-6, 10.0])  # second feature should be squashed toward 0
    w = fit_logistic(X, y, reg=reg)
    assert abs(w[1]) < abs(w[0])