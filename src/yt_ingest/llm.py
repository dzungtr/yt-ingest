"""Provider-abstract LLM layer.

Two backends are supported, selected purely by environment-variable presence:

1. **OpenAI-compatible** (precedence): any endpoint speaking the OpenAI Chat
   Completions API. Enabled when ``OPENAI_API_KEY`` is set; endpoint defaults
   to ``OPENAI_BASE_URL`` or, if unset, ``https://api.deepseek.com``.
   ``DEEPSEEK_API_KEY`` is still honoured as a legacy fallback for this path.
2. **Anthropic-compatible**: the Anthropic Messages API. Enabled when
   ``ANTHROPIC_API_KEY`` is set (and no OpenAI-style key is present).

Model selection uses two tiers (``strong`` / ``fast``) so prompts can ask for
a cheap model without hard-coding provider-specific names. Per-provider
defaults can be overridden with ``YT_INGEST_MODEL`` and ``YT_INGEST_FAST_MODEL``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import openai

from yt_ingest.config import get_config

logger = logging.getLogger(__name__)

_OPENAI_DEFAULT_BASE_URL = "https://api.deepseek.com"

_OPENAI_DEFAULT_STRONG_MODEL = "deepseek-v4-pro"
_OPENAI_DEFAULT_FAST_MODEL = "deepseek-v4-flash"
_ANTHROPIC_DEFAULT_STRONG_MODEL = "claude-sonnet-4-5"
_ANTHROPIC_DEFAULT_FAST_MODEL = "claude-haiku-4-5"


@dataclass
class CacheStats:
    hit_tokens: int = 0
    miss_tokens: int = 0
    total_calls: int = 0

    def merge(self, other: CacheStats) -> CacheStats:
        return CacheStats(
            hit_tokens=self.hit_tokens + other.hit_tokens,
            miss_tokens=self.miss_tokens + other.miss_tokens,
            total_calls=self.total_calls + other.total_calls,
        )


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can answer a system+user prompt with parsed JSON."""

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        model_tier: str = "strong",
        temperature: float = 0.0,
    ) -> tuple[Any, CacheStats]: ...


class OpenAICompatProvider:
    """OpenAI Chat Completions against any compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        strong_model: str,
        fast_model: str,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._models = {"strong": strong_model, "fast": fast_model}

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        model_tier: str = "strong",
        temperature: float = 0.0,
    ) -> tuple[Any, CacheStats]:
        model = self._models.get(model_tier, model_tier)
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        usage = response.usage
        stats = CacheStats(total_calls=1)
        if usage is not None:
            prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_tokens_details is not None:
                stats.hit_tokens = (
                    getattr(prompt_tokens_details, "cached_tokens", 0) or 0
                )
                stats.miss_tokens = (usage.prompt_tokens or 0) - stats.hit_tokens
            else:
                stats.miss_tokens = usage.prompt_tokens or 0

        logger.debug(
            "LLM cache: hit=%d miss=%d provider=openai-compat model=%s",
            stats.hit_tokens,
            stats.miss_tokens,
            model,
        )

        content = response.choices[0].message.content or ""
        return json.loads(content), stats


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class AnthropicProvider:
    """Anthropic Messages API (JSON extracted from plain-text response)."""

    def __init__(self, api_key: str, strong_model: str, fast_model: str) -> None:
        import anthropic  # lazy: only needed when this provider is selected

        self._client = anthropic.Anthropic(api_key=api_key)
        self._models = {"strong": strong_model, "fast": fast_model}

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        model_tier: str = "strong",
        temperature: float = 0.0,
    ) -> tuple[Any, CacheStats]:
        model = self._models.get(model_tier, model_tier)
        # Note: anthropic>=1.8 removed the `temperature` parameter; prompts here
        # rely on JSON-instruction discipline instead of sampling control.
        message = self._client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        stats = CacheStats(total_calls=1)
        usage = message.usage
        if usage is not None:
            stats.hit_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            stats.miss_tokens = usage.input_tokens or 0

        logger.debug(
            "LLM cache: hit=%d miss=%d provider=anthropic model=%s",
            stats.hit_tokens,
            stats.miss_tokens,
            model,
        )

        content = "".join(
            block.text for block in message.content if block.type == "text"
        )
        cleaned = _FENCE_RE.sub("", content.strip())
        return json.loads(cleaned), stats


class LLMConfigurationError(RuntimeError):
    """Raised when no LLM provider credentials are configured."""


def _select_provider() -> LLMProvider:
    cfg = get_config()
    strong = os.environ.get("YT_INGEST_MODEL") or None
    fast = os.environ.get("YT_INGEST_FAST_MODEL") or None

    openai_key = cfg.openai_api_key
    if openai_key:
        return OpenAICompatProvider(
            api_key=openai_key,
            base_url=cfg.openai_base_url,
            strong_model=strong or _OPENAI_DEFAULT_STRONG_MODEL,
            fast_model=fast or _OPENAI_DEFAULT_FAST_MODEL,
        )

    anthropic_key = cfg.anthropic_api_key
    if anthropic_key:
        return AnthropicProvider(
            api_key=anthropic_key,
            strong_model=strong or _ANTHROPIC_DEFAULT_STRONG_MODEL,
            fast_model=fast or _ANTHROPIC_DEFAULT_FAST_MODEL,
        )

    raise LLMConfigurationError(
        "No LLM provider configured. Set OPENAI_API_KEY (OpenAI-compatible "
        "endpoint, base URL via OPENAI_BASE_URL) or ANTHROPIC_API_KEY. "
        "DEEPSEEK_API_KEY is still accepted as a legacy OpenAI-compatible "
        "fallback."
    )


_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = _select_provider()
    return _provider


def reset_provider() -> None:
    """Drop the cached provider (used by tests)."""
    global _provider
    _provider = None


def chat_json(
    *,
    system: str,
    user: str,
    model: str = "strong",
    temperature: float = 0.0,
) -> tuple[Any, CacheStats]:
    """
    Ask the configured LLM provider, parse JSON from the response.

    ``model`` is a tier (``strong`` / ``fast``, resolved to a provider default
    or ``YT_INGEST_MODEL`` / ``YT_INGEST_FAST_MODEL``) or an explicit model
    name passed through as-is.

    Returns (parsed_object, CacheStats).
    """
    cfg = get_config()
    if not (cfg.openai_api_key or cfg.anthropic_api_key):
        raise LLMConfigurationError(
            "No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
    return _get_provider().chat_json(
        system=system, user=user, model_tier=model, temperature=temperature
    )