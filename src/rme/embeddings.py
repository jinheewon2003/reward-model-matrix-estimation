"""Text embedders.

The proposal's Figure 1 sweeps two embedder families:

- ``emb=sentence_transformer`` -> :class:`SentenceTransformerEmbedder`
- ``emb=llm``                  -> :class:`OpenAIEmbedder`

:class:`HashingEmbedder` is a third, deterministic, network-free option used
by the test suite and for smoke-testing the pipeline without any model
downloads or API keys.

All embedders are wrapped in :class:`CachingEmbedder` by callers so that the
same (prompt, response) text is never re-embedded twice within a run.
"""

from __future__ import annotations

import hashlib

import numpy as np

from rme.types import Embedder


class HashingEmbedder(Embedder):
    """Deterministic bag-of-hashed-tokens embedding. No model, no network.

    Not meant to produce competitive reward models -- it exists so the rest
    of the pipeline (matrix completion, training loops, eval) can be unit
    tested and smoke-run without downloading a sentence-transformer or
    calling an embeddings API.
    """

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(f"{self.seed}:{token}".encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder(Embedder):
    """Wraps a `sentence-transformers` model. Requires `pip install rme[embeddings]`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SentenceTransformerEmbedder needs `pip install sentence-transformers`."
            ) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, show_progress_bar=False))


class OpenAIEmbedder(Embedder):
    """Wraps the OpenAI embeddings API (the `emb=llm` condition in Figure 1).

    Requires `OPENAI_API_KEY` and `pip install rme[embeddings]`.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 64):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenAIEmbedder needs `pip install openai`.") from e
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = OpenAI()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return np.asarray(vectors)


class CachingEmbedder(Embedder):
    """Memoizes `embed_pair` calls for an inner embedder, keyed by exact text.

    Matrix-estimation training re-touches the same (prompt, response) many
    times (once per comparison it appears in), so caching is not an
    optimization detail -- without it, training cost scales with the number
    of comparisons instead of the number of unique responses.
    """

    def __init__(self, inner: Embedder):
        self.inner = inner
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        misses = [t for t in texts if t not in self._cache]
        if misses:
            vecs = self.inner.embed(misses)
            for t, v in zip(misses, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def embed_pair(self, prompt: str, response: str) -> np.ndarray:
        return self.embed([f"{prompt}\n\n{response}"])[0]

    def clear(self) -> None:
        self._cache.clear()