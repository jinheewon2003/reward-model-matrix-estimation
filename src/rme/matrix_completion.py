"""Content-augmented latent-factor matrix completion.

Algorithms 1 and 2 in the proposal both call for "SVD++ with item embeddings"
to fully impute a sparse user x (prompt, response) matrix. Classic SVD++
(Koren, 2008) has no notion of item *content* -- it only uses the pattern of
which items a user rated. We need item factors that generalize to
(prompt, response) pairs never seen during training (that's the entire point
of using embeddings), so this module implements a content-augmented variant
instead of depending on `scikit-surprise` (a fragile, C-extension-heavy
install, especially on Windows):

    pred[u, i] = mu + b_u + b_i + p_u . q_i
    q_i        = W @ embedding_i + r_i      (learned projection + free residual)

`W` lets an unseen item's factor be produced purely from its embedding
(`r_i` defaults to 0 for items not seen during training); the residual `r_i`
lets the model fit training-set idiosyncrasies beyond what the embedding
geometry captures. This is the mechanism that makes "impute a new row we can
average across users" (Section 3.3) possible for out-of-distribution
(prompt, response) pairs.

If you have `scikit-surprise` installed and want the literal SVD++
algorithm instead, swap this class out -- `LatentFactorImputer` and
`surprise.SVDpp` are interchangeable at the call sites in `models/`.
"""

from __future__ import annotations

import numpy as np


class LatentFactorImputer:
    def __init__(
        self,
        n_factors: int = 16,
        reg: float = 0.02,
        lr: float = 0.02,
        n_epochs: int = 60,
        seed: int = 0,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.rng = np.random.RandomState(seed)

        self.user_index_: dict[str, int] = {}
        self.item_index_: dict[str, int] = {}
        self.mu_: float = 0.0
        self.b_u_: np.ndarray | None = None
        self.b_i_: np.ndarray | None = None
        self.p_: np.ndarray | None = None  # user factors
        self.q_free_: np.ndarray | None = None  # per-item residual factors
        self.W_: np.ndarray | None = None  # embedding_dim x n_factors projection

    def fit(
        self,
        observations: list[tuple[str, str, float]],
        item_embeddings: dict[str, np.ndarray],
    ) -> "LatentFactorImputer":
        """`observations` is a list of (user_id, item_id, value in [0, 1])."""
        if not observations:
            raise ValueError("need at least one observation to fit")

        users = sorted({u for u, _, _ in observations})
        items = sorted({i for _, i, _ in observations})
        self.user_index_ = {u: idx for idx, u in enumerate(users)}
        self.item_index_ = {i: idx for idx, i in enumerate(items)}

        n_users, n_items = len(users), len(items)
        emb_dim = len(next(iter(item_embeddings.values())))

        self.mu_ = float(np.mean([v for _, _, v in observations]))
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        self.p_ = self.rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.q_free_ = self.rng.normal(0, 0.1, size=(n_items, self.n_factors))
        self.W_ = self.rng.normal(0, 0.1, size=(emb_dim, self.n_factors))

        item_emb_matrix = np.stack(
            [item_embeddings[i] for i in items]
        )  # (n_items, emb_dim)

        rows = np.array([self.user_index_[u] for u, _, _ in observations])
        cols = np.array([self.item_index_[i] for _, i, _ in observations])
        vals = np.array([v for _, _, v in observations], dtype=np.float64)
        order = np.arange(len(observations))

        for _ in range(self.n_epochs):
            self.rng.shuffle(order)
            for k in order:
                u, i, r = rows[k], cols[k], vals[k]
                q_i = self.W_.T @ item_emb_matrix[i] + self.q_free_[i]
                pred = self.mu_ + self.b_u_[u] + self.b_i_[i] + self.p_[u] @ q_i
                err = r - pred

                p_u_old = self.p_[u].copy()
                self.b_u_[u] += self.lr * (err - self.reg * self.b_u_[u])
                self.b_i_[i] += self.lr * (err - self.reg * self.b_i_[i])
                self.p_[u] += self.lr * (err * q_i - self.reg * self.p_[u])
                self.q_free_[i] += self.lr * (err * p_u_old - self.reg * self.q_free_[i])
                self.W_ += self.lr * (
                    err * np.outer(item_emb_matrix[i], p_u_old) - self.reg * self.W_
                )

        self._item_emb_matrix = item_emb_matrix
        self._items = items
        return self

    def _item_factor(self, item_id: str, embedding: np.ndarray | None) -> np.ndarray:
        if item_id in self.item_index_:
            idx = self.item_index_[item_id]
            return self.W_.T @ self._item_emb_matrix[idx] + self.q_free_[idx]
        if embedding is None:
            raise KeyError(f"unknown item {item_id!r} and no embedding supplied")
        return self.W_.T @ embedding

    def predict(
        self, user_id: str, item_id: str, embedding: np.ndarray | None = None
    ) -> float:
        """Predict a value in roughly [0, 1] for (user_id, item_id).

        `embedding` must be supplied for items not seen during `fit` (this is
        how the model generalizes to novel (prompt, response) pairs); unseen
        users fall back to the population average (`mu_` + mean item effect).
        """
        q_i = self._item_factor(item_id, embedding)
        if user_id in self.user_index_:
            u = self.user_index_[user_id]
            pred = self.mu_ + self.b_u_[u] + self._item_bias(item_id) + self.p_[u] @ q_i
        else:
            pred = self.mu_ + self._item_bias(item_id) + self.p_.mean(axis=0) @ q_i
        return float(np.clip(pred, 0.0, 1.0))

    def _item_bias(self, item_id: str) -> float:
        if item_id in self.item_index_:
            return float(self.b_i_[self.item_index_[item_id]])
        return 0.0

    def users(self) -> list[str]:
        return list(self.user_index_.keys())