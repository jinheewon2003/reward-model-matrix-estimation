"""Algorithm 4: ME Diff Pairwise-Baseline Model.

1. Every comparison becomes a directional difference vector
   `d = embed(P, winner) - embed(P, loser)`.
2. Fit a global linear scorer `score(u, d) = w . d + b_u` (per-user bias,
   shared direction `w`) via logistic regression with all-positive labels
   (every `d` is already oriented "good > bad", so this is a one-class BCE
   fit regularized to stay finite -- see `rme.linear.fit_logistic`).
3. Sample a fixed replay buffer of baseline responses from the training set.
4. Score a new (P, r) by its average predicted win probability against every
   baseline, over every user.
"""

from __future__ import annotations

import numpy as np

from rme.embeddings import CachingEmbedder
from rme.linear import fit_logistic
from rme.types import POPULATION_USER, Comparison, Embedder, RewardModel


class DiffPairwiseBaselineModel(RewardModel):
    def __init__(
        self,
        reg_w: float = 1e-3,
        reg_b: float = 1e-4,
        n_baselines: int = 50,
        seed: int = 0,
    ):
        self.reg_w = reg_w
        self.reg_b = reg_b
        self.n_baselines = n_baselines
        self.rng = np.random.RandomState(seed)
        self.embedder: CachingEmbedder | None = None

    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "DiffPairwiseBaselineModel":
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

        users = sorted({c.user for c in comparisons})
        self.user_index_ = {u: i for i, u in enumerate(users)}
        n_users = len(users)

        diffs = []
        user_rows = []
        for c in comparisons:
            e1 = self.embedder.embed_pair(c.prompt, c.response_1)
            e2 = self.embedder.embed_pair(c.prompt, c.response_2)
            d = (e1 - e2) if c.label >= 0.5 else (e2 - e1)
            diffs.append(d)
            user_rows.append(self.user_index_[c.user])
        D = np.stack(diffs)
        emb_dim = D.shape[1]

        one_hot = np.zeros((len(comparisons), n_users))
        one_hot[np.arange(len(comparisons)), user_rows] = 1.0
        X = np.hstack([D, one_hot])
        y = np.ones(len(comparisons))
        reg_vec = np.concatenate([np.full(emb_dim, self.reg_w), np.full(n_users, self.reg_b)])

        coef = fit_logistic(X, y, reg=reg_vec)
        self.w_ = coef[:emb_dim]
        self.b_u_ = coef[emb_dim:]

        # Step 3: replay buffer of baseline responses drawn from training data.
        pool = [(c.prompt, c.response_1) for c in comparisons] + [
            (c.prompt, c.response_2) for c in comparisons
        ]
        n = min(self.n_baselines, len(pool))
        chosen = self.rng.choice(len(pool), size=n, replace=False)
        self.baselines_ = [
            self.embedder.embed_pair(*pool[i]) for i in chosen
        ]
        return self

    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        e_star = self.embedder.embed_pair(prompt, response)
        if user in self.user_index_:
            biases = [self.b_u_[self.user_index_[user]]]
        else:
            biases = list(self.b_u_)  # average over all users (Section 3.3, step 4)

        probs = []
        for b_j in self.baselines_:
            d = e_star - b_j
            z = self.w_ @ d
            for b_u in biases:
                probs.append(1.0 / (1.0 + np.exp(-np.clip(z + b_u, -30, 30))))
        # R(P, r) is itself an averaged probability, not a score on an
        # unbounded scale; that's consistent with the other models here
        # since `predict_proba` only ever uses score *differences*.
        return float(np.mean(probs))