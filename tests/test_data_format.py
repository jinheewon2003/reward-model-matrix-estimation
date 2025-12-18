import pytest

from rme.data.hh_format import split_hh_transcript
from rme.utils import item_key, triplet_key


def test_split_hh_transcript_basic():
    transcript = "\n\nHuman: hi there\n\nAssistant: hello!"
    prompt, response = split_hh_transcript(transcript)
    assert prompt == "\n\nHuman: hi there\n\nAssistant:"
    assert response == "hello!"


def test_split_hh_transcript_uses_last_assistant_turn():
    transcript = (
        "\n\nHuman: first\n\nAssistant: reply1\n\nHuman: second\n\nAssistant: reply2"
    )
    prompt, response = split_hh_transcript(transcript)
    assert response == "reply2"
    assert prompt.endswith("second\n\nAssistant:")


def test_split_hh_transcript_raises_without_assistant_turn():
    with pytest.raises(ValueError):
        split_hh_transcript("\n\nHuman: no reply here")


def test_item_key_and_triplet_key_are_stable_and_distinct():
    assert item_key("p", "r") == item_key("p", "r")
    assert item_key("p", "r1") != item_key("p", "r2")
    assert triplet_key("p", "a", "b") != triplet_key("p", "b", "a")