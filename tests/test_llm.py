from __future__ import annotations
from unittest.mock import MagicMock, patch
import json

import pytest

import yt_ingest.llm as llm_module
from yt_ingest.llm import CacheStats, chat_json


def _make_response(content: str, cached_tokens: int = 0, prompt_tokens: int = 100) -> MagicMock:
    prompt_details = MagicMock()
    prompt_details.cached_tokens = cached_tokens

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.prompt_tokens_details = prompt_details

    choice = MagicMock()
    choice.message.content = content

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.fixture(autouse=True)
def reset_client() -> None:
    llm_module._client = None


def test_chat_json_returns_parsed_object() -> None:
    payload = {"summary": "hello world"}
    response = _make_response(json.dumps(payload))

    with patch("yt_ingest.llm._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_get.return_value = mock_client

        result, stats = chat_json(system="sys", user="usr")

    assert result == payload
    assert stats.total_calls == 1


def test_chat_json_cache_stats_hit() -> None:
    response = _make_response(json.dumps({}), cached_tokens=80, prompt_tokens=100)

    with patch("yt_ingest.llm._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_get.return_value = mock_client

        _, stats = chat_json(system="sys", user="usr")

    assert stats.hit_tokens == 80
    assert stats.miss_tokens == 20


def test_chat_json_no_cache_details() -> None:
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.prompt_tokens_details = None

    choice = MagicMock()
    choice.message.content = json.dumps({"x": 1})

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage

    with patch("yt_ingest.llm._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_get.return_value = mock_client

        _, stats = chat_json(system="sys", user="usr")

    assert stats.hit_tokens == 0
    assert stats.miss_tokens == 50


def test_cache_stats_merge() -> None:
    a = CacheStats(hit_tokens=10, miss_tokens=90, total_calls=1)
    b = CacheStats(hit_tokens=20, miss_tokens=30, total_calls=1)
    merged = a.merge(b)
    assert merged.hit_tokens == 30
    assert merged.miss_tokens == 120
    assert merged.total_calls == 2
