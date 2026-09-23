from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import yt_ingest.llm as llm_module
from yt_ingest.llm import CacheStats, LLMConfigurationError, chat_json


def _make_openai_response(content: str, cached_tokens: int = 0, prompt_tokens: int = 100) -> MagicMock:
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
def reset_provider() -> None:
    llm_module._provider = None


@pytest.fixture
def openai_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.openai_api_key = "test-key"
    cfg.openai_base_url = "https://api.deepseek.com"
    cfg.anthropic_api_key = ""
    return cfg


@pytest.fixture
def anthropic_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.openai_api_key = ""
    cfg.openai_base_url = ""
    cfg.anthropic_api_key = "test-key"
    return cfg


@pytest.fixture
def no_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.openai_api_key = ""
    cfg.anthropic_api_key = ""
    return cfg


# ---- provider selection ----


def test_openai_takes_precedence() -> None:
    cfg = MagicMock()
    cfg.openai_api_key = "oa-key"
    cfg.openai_base_url = "https://api.deepseek.com"
    cfg.anthropic_api_key = "an-key"
    with patch("yt_ingest.llm.get_config", return_value=cfg):
        assert isinstance(llm_module._select_provider(), llm_module.OpenAICompatProvider)


def test_selects_openai_provider(openai_cfg: MagicMock) -> None:
    with patch("yt_ingest.llm.get_config", return_value=openai_cfg):
        provider = llm_module._select_provider()
    assert isinstance(provider, llm_module.OpenAICompatProvider)


def test_selects_anthropic_provider(anthropic_cfg: MagicMock) -> None:
    with patch("yt_ingest.llm.get_config", return_value=anthropic_cfg):
        provider = llm_module._select_provider()
    assert isinstance(provider, llm_module.AnthropicProvider)


def test_raises_without_any_key(no_cfg: MagicMock) -> None:
    with patch("yt_ingest.llm.get_config", return_value=no_cfg):
        with pytest.raises(LLMConfigurationError):
            llm_module._select_provider()


# ---- openai-compatible provider ----


def test_chat_json_returns_parsed_object(openai_cfg: MagicMock) -> None:
    payload = {"summary": "hello world"}
    response = _make_openai_response(json.dumps(payload))

    with (
        patch("yt_ingest.llm.get_config", return_value=openai_cfg),
        patch("openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls = mock_openai
        mock_openai_cls.return_value = mock_client

        result, stats = chat_json(system="sys", user="usr")

    assert result == payload
    assert stats.total_calls == 1
    assert mock_openai_cls.call_args.kwargs["base_url"] == "https://api.deepseek.com"


def test_chat_json_model_tier_resolution(openai_cfg: MagicMock) -> None:
    response = _make_openai_response(json.dumps({}))

    with (
        patch("yt_ingest.llm.get_config", return_value=openai_cfg),
        patch("openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        _, _ = chat_json(system="sys", user="usr", model="fast")

    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "deepseek-v4-flash"


def test_chat_json_explicit_model_passthrough(openai_cfg: MagicMock) -> None:
    response = _make_openai_response(json.dumps({}))

    with (
        patch("yt_ingest.llm.get_config", return_value=openai_cfg),
        patch("openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        _, _ = chat_json(system="sys", user="usr", model="gpt-4o")

    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"


def test_chat_json_cache_stats_hit(openai_cfg: MagicMock) -> None:
    response = _make_openai_response(json.dumps({}), cached_tokens=80, prompt_tokens=100)

    with (
        patch("yt_ingest.llm.get_config", return_value=openai_cfg),
        patch("openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        _, stats = chat_json(system="sys", user="usr")

    assert stats.hit_tokens == 80
    assert stats.miss_tokens == 20


def test_chat_json_no_cache_details(openai_cfg: MagicMock) -> None:
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.prompt_tokens_details = None

    choice = MagicMock()
    choice.message.content = json.dumps({"x": 1})

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage

    with (
        patch("yt_ingest.llm.get_config", return_value=openai_cfg),
        patch("openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        _, stats = chat_json(system="sys", user="usr")

    assert stats.hit_tokens == 0
    assert stats.miss_tokens == 50


# ---- anthropic provider ----


def _make_anthropic_response(content: str, input_tokens: int, cache_read: int = 0) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.cache_read_input_tokens = cache_read

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content

    message = MagicMock()
    message.content = [text_block]
    message.usage = usage
    return message


def test_anthropic_provider_basic(anthropic_cfg: MagicMock) -> None:
    message = _make_anthropic_response('{"answer": "hi"}', input_tokens=120, cache_read=40)

    with (
        patch("yt_ingest.llm.get_config", return_value=anthropic_cfg),
        patch("anthropic.Anthropic") as mock_anthropic_cls,
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = message
        mock_anthropic_cls.return_value = mock_client

        result, stats = chat_json(system="sys", user="usr")

    assert result == {"answer": "hi"}
    assert stats.hit_tokens == 40
    assert stats.miss_tokens == 120


def test_anthropic_strips_code_fence(anthropic_cfg: MagicMock) -> None:
    message = _make_anthropic_response('```json\n{"a": 1}\n```', input_tokens=10)

    with (
        patch("yt_ingest.llm.get_config", return_value=anthropic_cfg),
        patch("anthropic.Anthropic") as mock_anthropic_cls,
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = message
        mock_anthropic_cls.return_value = mock_client

        result, _ = chat_json(system="sys", user="usr")

    assert result == {"a": 1}


def test_anthropic_explicit_model_passthrough(anthropic_cfg: MagicMock) -> None:
    message = _make_anthropic_response("{}", input_tokens=10)

    with (
        patch("yt_ingest.llm.get_config", return_value=anthropic_cfg),
        patch("anthropic.Anthropic") as mock_anthropic_cls,
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = message
        mock_anthropic_cls.return_value = mock_client

        _, _ = chat_json(system="sys", user="usr", model="claude-opus-4-5")

    assert mock_client.messages.create.call_args.kwargs["model"] == "claude-opus-4-5"


# ---- misc ----


def test_chat_json_raises_without_any_key(no_cfg: MagicMock) -> None:
    with patch("yt_ingest.llm.get_config", return_value=no_cfg):
        with pytest.raises(LLMConfigurationError):
            chat_json(system="sys", user="usr")


def test_cache_stats_merge() -> None:
    a = CacheStats(hit_tokens=10, miss_tokens=90, total_calls=1)
    b = CacheStats(hit_tokens=20, miss_tokens=30, total_calls=1)
    merged = a.merge(b)
    assert merged.hit_tokens == 30
    assert merged.miss_tokens == 120
    assert merged.total_calls == 2