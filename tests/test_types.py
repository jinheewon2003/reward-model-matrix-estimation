import pytest

from rme.types import POPULATION_USER, Comparison


def test_comparison_accepts_valid_label():
    c = Comparison(POPULATION_USER, "p", "r1", "r2", 1.0)
    assert c.label == 1.0


def test_comparison_rejects_out_of_range_label():
    with pytest.raises(ValueError):
        Comparison(POPULATION_USER, "p", "r1", "r2", 1.5)