"""Algorithm 2: ME Pairwise-to-Response Model.

1. Build a user x (P, r1, r2) matrix of preference probabilities and fully
   impute it with the same content-augmented latent-factor completion used
   by Algorithm 1 (`rme.matrix_completion.LatentFactorImputer`), keying each
   pairwise "item" by the difference embedding `embed(P,r1) - embed(P,r2)`
   so unseen triplets can still be scored.
2. Greedily fold the imputed pairwise probabilities into a per-response
   distribution `S[u, P, r]`, following the odds-ratio update / conflict
   rules in the proposal: the first triplet touching a prompt sets both
   responses from `p`, a triplet touching one known response infers the
   other via the odds ratio, and a triplet touching two already-set
   responses either reconciles them (keeping the more extreme of the old and
   new odds ratio, mass held fixed) or -- if the two ratios disagree in
   direction -- resets that prompt's distribution to uniform.

Ambiguity note: the proposal doesn't specify how to score a (P, r) that
never appears in any training triplet. `score()` resolves this by reusing
the fitted pairwise imputer to estimate the query response's odds relative
to whatever responses *are* known for that prompt, converting each known
response's own `S` value to log-odds and back; a prompt with no known
responses at all falls back to the global mean `S` value.
"""

from __future__ import annotations

import numpy as np

from rme.embeddings import CachingEmbedder
from rme.matrix_completion import LatentFactorImputer
from rme.types import POPULATION_USER, Comparison, Embedder, RewardModel
from rme.utils import triplet_key

_EPS = 1e-4


class PairwiseToResponseModel(RewardModel):
    def __init__(self, n_factors: int = 16, reg: float = 0.02, lr: float = 0.02, n_epochs: int = 60):
        self.n_factors = n_factors
        self.reg = reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.embedder: CachingEmbedder | None = None

    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "PairwiseToResponseModel":
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

        triplet_embeddings: dict[str, np.ndarray] = {}
        ordered_triplets: list[tuple[str, str, str]] = []
        seen_triplets: set[str] = set()
        label_sum: dict[tuple[str, str], float] = {}
        label_count: dict[tuple[str, str], float] = {}

        for c in comparisons:
            tkey = triplet_key(c.prompt, c.response_1, c.response_2)
            if tkey not in seen_triplets:
                seen_triplets.add(tkey)
                ordered_triplets.append((c.prompt, c.response_1, c.response_2))
                e1 = self.embedder.embed_pair(c.prompt, c.response_1)
                e2 = self.embedder.embed_pair(c.prompt, c.response_2)
                triplet_embeddings[tkey] = e1 - e2
            label_sum[(c.user, tkey)] = label_sum.get((c.user, tkey), 0.0) + c.label
            label_count[(c.user, tkey)] = label_count.get((c.user, tkey), 0.0) + 1.0

        observations = [
            (u, tkey, label_sum[(u, tkey)] / label_count[(u, tkey)])
            for (u, tkey) in label_count
        ]
        self.imputer = LatentFactorImputer(
            n_factors=self.n_factors, reg=self.reg, lr=self.lr, n_epochs=self.n_epochs
        ).fit(observations, triplet_embeddings)

        # Step 2: fold imputed pairwise probabilities into per-response scores.
        self.S: dict[str, dict[str, dict[str, float]]] = {}
        for prompt, r1, r2 in ordered_triplets:
            tkey = triplet_key(prompt, r1, r2)
            emb = triplet_embeddings[tkey]
            for u in self.imputer.users():
                p = float(np.clip(self.imputer.predict(u, tkey, emb), _EPS, 1 - _EPS))
                prompt_scores = self.S.setdefault(u, {}).setdefault(prompt, {})
                _update_prompt_scores(prompt_scores, r1, r2, p)

        for prompt_scores in self.S.values():
            for responses in prompt_scores.values():
                total = sum(responses.values())
                if total > 0:
                    for r in responses:
                        responses[r] /= total

        self._global_mean = float(
            np.mean([s for prompts in self.S.values() for r in prompts.values() for s in r.values()])
        ) if self.S else 0.5
        return self

    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        candidate_users = [user] if user in self.S else list(self.S.keys())
        known = None
        for u in candidate_users:
            prompt_scores = self.S.get(u, {}).get(prompt)
            if prompt_scores and response in prompt_scores:
                return prompt_scores[response]
            if prompt_scores:
                known = (u, prompt_scores)

        e_query = self.embedder.embed_pair(prompt, response)
        log_odds = []
        users_to_check = [user] if user in self.S else self.S.keys()
        for u in users_to_check:
            prompt_scores = self.S.get(u, {}).get(prompt)
            if not prompt_scores:
                continue
            for r_known, s_known in prompt_scores.items():
                e_known = self.embedder.embed_pair(prompt, r_known)
                p = float(
                    np.clip(
                        self.imputer.predict(u, "__query__", e_query - e_known),
                        _EPS,
                        1 - _EPS,
                    )
                )
                s_known_clipped = min(max(s_known, _EPS), 1 - _EPS)
                log_odds.append(np.log(p / (1 - p)) + np.log(s_known_clipped / (1 - s_known_clipped)))

        if log_odds:
            return float(1.0 / (1.0 + np.exp(-np.mean(log_odds))))
        return self._global_mean


def _update_prompt_scores(
    prompt_scores: dict[str, float], r1: str, r2: str, p: float
) -> None:
    has1, has2 = r1 in prompt_scores, r2 in prompt_scores
    if not has1 and not has2:
        prompt_scores[r1] = p
        prompt_scores[r2] = 1 - p
        return
    if has1 and not has2:
        prompt_scores[r2] = prompt_scores[r1] * (1 - p) / p
        return
    if has2 and not has1:
        prompt_scores[r1] = prompt_scores[r2] * p / (1 - p)
        return

    r_rel = prompt_scores[r1] / prompt_scores[r2]
    r_new = p / (1 - p)
    if (r_rel - 1) * (r_new - 1) < 0:
        for r in prompt_scores:
            prompt_scores[r] = 1.0
        return

    ratio = max(r_rel, r_new)
    total = prompt_scores[r1] + prompt_scores[r2]
    prompt_scores[r2] = total / (1 + ratio)
    prompt_scores[r1] = total - prompt_scores[r2]