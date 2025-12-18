"""Loader logic tests, using monkeypatched fake HF datasets (plain lists of
dicts shaped like the real schemas -- see `rme/data/datasets.py`'s module
docstring for how those schemas were confirmed) so these run offline.
"""

from __future__ import annotations

import pytest

import rme.data.datasets as datasets_mod
from rme.types import POPULATION_USER


def _patch_load_hf(monkeypatch, rows):
    monkeypatch.setattr(datasets_mod, "_load_hf", lambda *a, **k: rows)


def test_load_anthropic_hh_golden(monkeypatch):
    rows = [
        {
            "chosen": "\n\nHuman: hi\n\nAssistant: good answer",
            "rejected": "\n\nHuman: hi\n\nAssistant: bad answer",
        }
    ]
    _patch_load_hf(monkeypatch, rows)
    comparisons = datasets_mod.load_anthropic_hh_golden()
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.response_1 == "good answer"
    assert c.response_2 == "bad answer"
    assert c.label == 1.0
    assert c.user == POPULATION_USER


def test_load_anthropic_hh_golden_skips_malformed_rows(monkeypatch):
    rows = [{"chosen": "no assistant turn", "rejected": "\n\nHuman: hi\n\nAssistant: x"}]
    _patch_load_hf(monkeypatch, rows)
    assert datasets_mod.load_anthropic_hh_golden() == []


def test_load_helpsteer2_generates_pairs_from_shared_prompt(monkeypatch):
    rows = [
        {"prompt": "p1", "response": "r_a", "helpfulness": 4},
        {"prompt": "p1", "response": "r_b", "helpfulness": 1},
        {"prompt": "p1", "response": "r_c", "helpfulness": 4},  # tie with r_a
        {"prompt": "p2", "response": "r_d", "helpfulness": 3},
    ]
    _patch_load_hf(monkeypatch, rows)
    comparisons = datasets_mod.load_helpsteer2()
    by_pair = {(c.response_1, c.response_2): c.label for c in comparisons}
    assert by_pair[("r_a", "r_b")] == 1.0
    assert ("r_a", "r_c") not in by_pair  # tie -> skipped
    assert all(c.prompt != "p2" for c in comparisons)  # p2 has only one response


def test_load_summarize_from_feedback(monkeypatch):
    rows = [
        {
            "info": {"post": "article text", "article": None},
            "summaries": [{"text": "summary A"}, {"text": "summary B"}],
            "choice": 1,
        }
    ]
    _patch_load_hf(monkeypatch, rows)
    comparisons = datasets_mod.load_summarize_from_feedback()
    assert len(comparisons) == 1
    assert comparisons[0].prompt == "article text"
    assert comparisons[0].label == 0.0  # choice=1 means summaries[1] preferred


def test_load_summarize_from_feedback_skips_non_binary_summaries(monkeypatch):
    rows = [{"info": {"post": "x"}, "summaries": [{"text": "a"}], "choice": 0}]
    _patch_load_hf(monkeypatch, rows)
    assert datasets_mod.load_summarize_from_feedback() == []


def test_load_prism_alignment(monkeypatch):
    rows = [
        {
            "conversation_id": "c1",
            "turn": 0,
            "user_id": "u1",
            "user_prompt": "explain gravity",
            "model_response": "chosen response",
            "if_chosen": True,
        },
        {
            "conversation_id": "c1",
            "turn": 0,
            "user_id": "u1",
            "user_prompt": "explain gravity",
            "model_response": "rejected response",
            "if_chosen": False,
        },
    ]
    _patch_load_hf(monkeypatch, rows)
    comparisons = datasets_mod.load_prism_alignment()
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.user == "u1"
    assert c.response_1 == "chosen response"
    assert c.response_2 == "rejected response"


def test_load_prism_alignment_skips_turns_without_both_sides(monkeypatch):
    rows = [
        {
            "conversation_id": "c1",
            "turn": 0,
            "user_id": "u1",
            "user_prompt": "p",
            "model_response": "only response",
            "if_chosen": True,
        }
    ]
    _patch_load_hf(monkeypatch, rows)
    assert datasets_mod.load_prism_alignment() == []


def test_load_collective_alignment_1_raises_clear_error_on_schema_mismatch(monkeypatch):
    _patch_load_hf(monkeypatch, [{"unexpected_key": 1}])
    with pytest.raises(datasets_mod.DatasetSchemaError):
        datasets_mod.load_collective_alignment_1()