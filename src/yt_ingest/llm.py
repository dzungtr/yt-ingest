from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from yt_ingest.config import get_config

logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        _client = OpenAI(api_key=cfg.deepseek_api_key, base_url=_DEEPSEEK_BASE_URL)
    return _client


@dataclass
class CacheStats:
    hit_tokens: int = 0
    miss_tokens: int = 0
    total_calls: int = 0

    def merge(self, other: "CacheStats") -> "CacheStats":
        return CacheStats(
            hit_tokens=self.hit_tokens + other.hit_tokens,
            miss_tokens=self.miss_tokens + other.miss_tokens,
            total_calls=self.total_calls + other.total_calls,
        )


def chat_json(
    *,
    system: str,
    user: str,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
) -> tuple[Any, CacheStats]:
    """
    Call DeepSeek chat, parse JSON from the response.
    Returns (parsed_object, CacheStats).
    """
    client = _get_client()
    response = client.chat.completions.create(
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
            stats.hit_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
            stats.miss_tokens = (usage.prompt_tokens or 0) - stats.hit_tokens
        else:
            stats.miss_tokens = usage.prompt_tokens or 0

        logger.debug(
            "DeepSeek cache: hit=%d miss=%d model=%s",
            stats.hit_tokens,
            stats.miss_tokens,
            model,
        )

    content = response.choices[0].message.content or ""
    return json.loads(content), stats
