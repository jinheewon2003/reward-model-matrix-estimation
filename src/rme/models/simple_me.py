"""Algorithm 1: Simple ME (Unary) Model.

Build a sparse user x (prompt, response) win-rate matrix directly from the
comparison data, then fully impute it with a content-augmented latent-factor
model (see `rme.matrix_completion`) so every response gets a score, not just
the ones a user directly compared.
"""

from __future__ import annotations

import numpy as np

from rme.embeddings import CachingEmbedder
from rme.matrix_completion import LatentFactorImputer
from rme.types import POPULATION_USER, Comparison, Embedder, RewardModel
from rme.utils import item_key


class SimpleMEModel(RewardModel):
    def __init__(self, n_factors: int = 16, reg: float = 0.02, lr: float = 0.02, n_epochs: int = 60):
        self.n_factors = n_factors
        self.reg = reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.embedder: CachingEmbedder | None = None
        self.imputer: LatentFactorImputer | None = None

    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "SimpleMEModel":
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

        wins: dict[tuple[str, str], float] = {}
        games: dict[tuple[str, str], float] = {}
        item_embeddings: dict[str, np.ndarray] = {}

        for c in comparisons:
            k1 = item_key(c.prompt, c.response_1)
            k2 = item_key(c.prompt, c.response_2)
            item_embeddings.setdefault(k1, self.embedder.embed_pair(c.prompt, c.response_1))
            item_embeddings.setdefault(k2, self.embedder.embed_pair(c.prompt, c.response_2))

            wins[(c.user, k1)] = wins.get((c.user, k1), 0.0) + c.label
            games[(c.user, k1)] = games.get((c.user, k1), 0.0) + 1.0
            wins[(c.user, k2)] = wins.get((c.user, k2), 0.0) + (1.0 - c.label)
            games[(c.user, k2)] = games.get((c.user, k2), 0.0) + 1.0

        observations = [
            (u, k, wins[(u, k)] / games[(u, k)]) for (u, k) in games if games[(u, k)] > 0
        ]

        self.imputer = LatentFactorImputer(
            n_factors=self.n_factors, reg=self.reg, lr=self.lr, n_epochs=self.n_epochs
        ).fit(observations, item_embeddings)
        return self

    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        k = item_key(prompt, response)
        e = self.embedder.embed_pair(prompt, response)
        if user in self.imputer.user_index_:
            return self.imputer.predict(user, k, e)
        # Aggregate-user datasets, or a user never seen at training time:
        # "impute a new row which we can average across all users" (Section 3.3).
        return float(np.mean([self.imputer.predict(u, k, e) for u in self.imputer.users()]))