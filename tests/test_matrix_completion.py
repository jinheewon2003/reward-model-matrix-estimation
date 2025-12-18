import numpy as np

from rme.matrix_completion import LatentFactorImputer


def _make_embeddings(n, dim, seed):
    rng = np.random.RandomState(seed)
    return {f"item{i}": rng.normal(size=dim) for i in range(n)}, rng.normal(size=dim)


def _true_value(w, e):
    return 1.0 / (1.0 + np.exp(-(w @ e)))


def test_imputer_fits_known_observations_closely():
    embeddings, w = _make_embeddings(20, dim=8, seed=0)
    observations = [("u1", k, _true_value(w, e)) for k, e in embeddings.items()]
    imputer = LatentFactorImputer(n_factors=4, n_epochs=200, lr=0.05, reg=0.01, seed=0).fit(
        observations, embeddings
    )
    errs = [abs(imputer.predict("u1", k, e) - _true_value(w, e)) for k, e in embeddings.items()]
    assert np.mean(errs) < 0.15


def test_imputer_generalizes_to_unseen_item_via_embedding():
    embeddings, w = _make_embeddings(30, dim=8, seed=1)
    keys = list(embeddings)
    train_keys, held_out_keys = keys[:20], keys[20:]
    observations = [("u1", k, _true_value(w, embeddings[k])) for k in train_keys]
    imputer = LatentFactorImputer(n_factors=4, n_epochs=200, lr=0.05, reg=0.01, seed=0).fit(
        observations, {k: embeddings[k] for k in train_keys}
    )
    preds = [imputer.predict("u1", k, embeddings[k]) for k in held_out_keys]
    truths = [_true_value(w, embeddings[k]) for k in held_out_keys]
    assert all(0.0 <= p <= 1.0 for p in preds)
    assert np.corrcoef(preds, truths)[0, 1] > 0.5


def test_unknown_user_falls_back_to_population_average():
    embeddings, w = _make_embeddings(10, dim=6, seed=2)
    observations = [("u1", k, _true_value(w, e)) for k, e in embeddings.items()]
    imputer = LatentFactorImputer(n_factors=3, n_epochs=50, seed=0).fit(observations, embeddings)
    k = next(iter(embeddings))
    pred = imputer.predict("never_seen_user", k, embeddings[k])
    assert 0.0 <= pred <= 1.0