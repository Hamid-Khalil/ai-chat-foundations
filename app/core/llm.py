"""The ONLY place the LangChain chat model is constructed."""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from app.core.config import get_settings


@lru_cache
def _build_chat_model(thinking_level: str) -> BaseChatModel:
    """Build and cache one chat model per thinking level.

    Uses ChatGroq while reading the configured settings (model, API key,
    max_tokens, timeout, and retries).
    """
    settings = get_settings()
    api_key = (
        settings.groq_api_key.get_secret_value()
        or settings.google_api_key.get_secret_value()
    )

    return ChatGroq(
        model=settings.gemini_model,
        api_key=api_key,
        max_tokens=settings.gemini_max_tokens,
        timeout=settings.gemini_timeout_seconds,
        max_retries=settings.gemini_max_retries,
    )


def get_chat_model(thinking_level: str | None = None) -> BaseChatModel:
    """Return the cached chat model for `thinking_level` (default from Settings)."""
    level = thinking_level or get_settings().gemini_thinking_level
    return _build_chat_model(level)
