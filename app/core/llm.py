"""The ONLY place the LangChain chat model is constructed."""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from app.core.config import get_settings


@lru_cache
def _build_chat_model(thinking_level: str) -> BaseChatModel:
    """Build and cache one chat model per thinking level."""
    settings = get_settings()

    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        max_tokens=settings.gemini_max_tokens,
        timeout=30.0,
        max_retries=2,
    )


def get_chat_model(thinking_level: str | None = None) -> BaseChatModel:
    """Return the cached chat model for `thinking_level` (default from Settings)."""
    level = thinking_level or get_settings().gemini_thinking_level
    return _build_chat_model(level)
