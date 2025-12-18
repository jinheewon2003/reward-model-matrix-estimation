import numpy as np

from rme.embeddings import CachingEmbedder, HashingEmbedder


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=32, seed=0)
    v1 = e.embed(["hello world"])[0]
    v2 = e.embed(["hello world"])[0]
    assert np.allclose(v1, v2)


def test_hashing_embedder_distinguishes_different_text():
    e = HashingEmbedder(dim=32, seed=0)
    v_good = e.embed(["excellent"])[0]
    v_bad = e.embed(["terrible"])[0]
    assert not np.allclose(v_good, v_bad)


def test_caching_embedder_only_calls_inner_once_per_text():
    calls = []

    class CountingEmbedder(HashingEmbedder):
        def embed(self, texts):
            calls.append(list(texts))
            return super().embed(texts)

    e = CachingEmbedder(CountingEmbedder(dim=16))
    e.embed_pair("p", "r")
    e.embed_pair("p", "r")
    e.embed_pair("p", "r2")
    assert sum(len(c) for c in calls) == 2  # second "p\n\nr" call was cached