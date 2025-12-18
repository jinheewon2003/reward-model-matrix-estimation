from rme.evaluation import pairwise_accuracy
from rme.models.simple_me import SimpleMEModel


def test_simple_me_fits_train_and_generalizes(embedder, train_comparisons, held_out_comparisons):
    model = SimpleMEModel(n_factors=8, n_epochs=100).fit(train_comparisons, embedder)
    assert pairwise_accuracy(model, train_comparisons) > 0.75
    assert pairwise_accuracy(model, held_out_comparisons) > 0.6


def test_simple_me_score_is_bounded(embedder, train_comparisons):
    model = SimpleMEModel(n_factors=8, n_epochs=50).fit(train_comparisons, embedder)
    c = train_comparisons[0]
    s = model.score(c.prompt, c.response_1)
    assert 0.0 <= s <= 1.0