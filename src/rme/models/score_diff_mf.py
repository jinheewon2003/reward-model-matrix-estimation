"""Algorithm 3: ME Score-Diff MF Model.

A personalized matrix-factorization scorer

    s(u, P, r) = b_i + p_u . (W @ embed(P, r) + q_free_i)

trained directly on the score *difference* with a Bradley-Terry logistic
loss: s(u,P,r1) - s(u,P,r2) ~ logit(y). `p_u` is a per-user weighting over a
shared, embedding-derived factor space, `q_free_i` a free per-item residual
factor, and `W` the embedding-to-factor projection that lets the model score
items it never trained on.

Note: a per-user *scalar* bias `b_u` would be unidentifiable from pairwise
differences alone (it cancels in s1 - s2), so unlike Algorithm 1/2's matrix
completion (which fits absolute win-rates, where a user bias is meaningful),
personalization here lives entirely in `p_u`.
"""

from __future__ import annotations

import numpy as np

from rme.embeddings import CachingEmbedder
from rme.types import POPULATION_USER, Comparison, Embedder, RewardModel
from rme.utils import item_key


class ScoreDiffMFModel(RewardModel):
    def __init__(
        self,
        n_factors: int = 16,
        reg: float = 0.01,
        lr: float = 0.02,
        n_epochs: int = 80,
        seed: int = 0,
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.rng = np.random.RandomState(seed)
        self.embedder: CachingEmbedder | None = None

    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "ScoreDiffMFModel":
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

        users = sorted({c.user for c in comparisons})
        self.user_index_ = {u: i for i, u in enumerate(users)}

        items: dict[str, np.ndarray] = {}
        rows = []
        for c in comparisons:
            k1, k2 = item_key(c.prompt, c.response_1), item_key(c.prompt, c.response_2)
            items.setdefault(k1, self.embedder.embed_pair(c.prompt, c.response_1))
            items.setdefault(k2, self.embedder.embed_pair(c.prompt, c.response_2))
            rows.append((self.user_index_[c.user], k1, k2, c.label))

        self.item_index_ = {k: i for i, k in enumerate(items)}
        item_emb = np.stack([items[k] for k in self.item_index_])
        emb_dim = item_emb.shape[1]
        n_users, n_items = len(users), len(items)

        self.p_ = self.rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.q_free_ = self.rng.normal(0, 0.1, size=(n_items, self.n_factors))
        self.W_ = self.rng.normal(0, 0.1, size=(emb_dim, self.n_factors))
        self.b_i_ = np.zeros(n_items)
        self._item_emb = item_emb

        order = np.arange(len(rows))
        for _ in range(self.n_epochs):
            self.rng.shuffle(order)
            for idx in order:
                u, k1, k2, y = rows[idx]
                i1, i2 = self.item_index_[k1], self.item_index_[k2]
                q1 = self.W_.T @ item_emb[i1] + self.q_free_[i1]
                q2 = self.W_.T @ item_emb[i2] + self.q_free_[i2]
                s1 = self.b_i_[i1] + self.p_[u] @ q1
                s2 = self.b_i_[i2] + self.p_[u] @ q2
                pred = 1.0 / (1.0 + np.exp(-np.clip(s1 - s2, -30, 30)))
                err = y - pred  # d(BCE)/d(diff) = -err

                p_u_old = self.p_[u].copy()
                self.b_i_[i1] += self.lr * (err - self.reg * self.b_i_[i1])
                self.b_i_[i2] += self.lr * (-err - self.reg * self.b_i_[i2])
                self.p_[u] += self.lr * (err * (q1 - q2) - self.reg * self.p_[u])
                self.q_free_[i1] += self.lr * (err * p_u_old - self.reg * self.q_free_[i1])
                self.q_free_[i2] += self.lr * (-err * p_u_old - self.reg * self.q_free_[i2])
                self.W_ += self.lr * (
                    err * np.outer(item_emb[i1] - item_emb[i2], p_u_old) - self.reg * self.W_
                )
        return self

    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        k = item_key(prompt, response)
        e = self.embedder.embed_pair(prompt, response)
        if k in self.item_index_:
            i = self.item_index_[k]
            q = self.W_.T @ self._item_emb[i] + self.q_free_[i]
            b = self.b_i_[i]
        else:
            q = self.W_.T @ e
            b = 0.0
        p_u = self.p_[self.user_index_[user]] if user in self.user_index_ else self.p_.mean(axis=0)
        return float(b + p_u @ q)