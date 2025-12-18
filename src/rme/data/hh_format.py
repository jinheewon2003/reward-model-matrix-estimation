"""Parsing for the Anthropic HH `\n\nHuman: ...\n\nAssistant: ...` transcript format."""

from __future__ import annotations

_ASSISTANT_TAG = "\n\nAssistant:"


def split_hh_transcript(transcript: str) -> tuple[str, str]:
    """Split a full HH transcript into (prompt, final_assistant_response).

    The prompt is everything up to and including the last "\\n\\nAssistant:"
    tag; the response is the text that follows it.
    """
    idx = transcript.rfind(_ASSISTANT_TAG)
    if idx == -1:
        raise ValueError(f"no '{_ASSISTANT_TAG.strip()}' turn found in transcript")
    prompt = transcript[: idx + len(_ASSISTANT_TAG)]
    response = transcript[idx + len(_ASSISTANT_TAG) :].strip()
    return prompt, response