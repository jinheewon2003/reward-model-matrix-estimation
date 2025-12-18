from rme.evaluation import pairwise_accuracy
from rme.models.bradley_terry import BradleyTerryModel


def test_bradley_terry_fits_train_and_generalizes(embedder, train_comparisons, held_out_comparisons):
    model = BradleyTerryModel(reg=1e-3).fit(train_comparisons, embedder)
    assert pairwise_accuracy(model, train_comparisons) > 0.85
    assert pairwise_accuracy(model, held_out_comparisons) > 0.8