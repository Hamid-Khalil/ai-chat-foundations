"""Application settings, loaded once from environment variables / `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the allowed levels (used by Settings, schemas, /chat/model).
ThinkingLevel = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    """One typed field per variable in `.env`; defaults are the fixed values."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Groq API key configuration
    groq_api_key: SecretStr = SecretStr("")
    # Fallback/legacy reference in case existing tests read google_api_key
    google_api_key: SecretStr = SecretStr("")

    gemini_model: str = "llama-3.3-70b-versatile"
    gemini_thinking_level: ThinkingLevel = "low"
    gemini_max_tokens: int = 2048
    gemini_timeout_seconds: int = 30
    gemini_max_retries: int = 0
    chat_max_messages: int = 20
    chat_max_message_chars: int = 4000

    # --- Week 8 RAG / Knowledge Base Settings ---
    embedding_model: str = "gemini-embedding-001"
    embedding_batch_size: int = 32
    # Supports both VECTOR_STORE_DIR and legacy CHROMA_PERSIST_DIR
    vector_store_dir: str = Field(default="data/chroma", alias="VECTOR_STORE_DIR")
    chroma_collection: str = "knowledge_base"

    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    rag_top_k_default: int = 4
    rag_top_k_max: int = 8
    rag_min_score: float = 0.0
    rag_max_context_chars: int = 6000
    rag_max_question_chars: int = 500
    rag_max_upload_bytes: int = 200000
    rag_max_documents: int = 20

    # --- Week 9 MongoDB Persistence Settings ---
    mongodb_uri: SecretStr = Field(
        default=SecretStr(
            "mongodb://app_user:CHANGE_ME@127.0.0.1:27017/ai_chat_foundations?authSource=ai_chat_foundations"
        )
    )
    mongodb_db: str = Field(default="ai_chat_foundations")
    mongodb_timeout_ms: int = Field(default=3000)
    history_max_messages: int = Field(default=18)
    conversation_max_messages: int = Field(default=200)

    @field_validator("mongodb_timeout_ms")
    @classmethod
    def validate_mongodb_timeout_ms(cls, v: int) -> int:
        if not (500 <= v <= 30000):
            raise ValueError(
                "MONGODB_TIMEOUT_MS must be between 500 and 30000 milliseconds."
            )
        return v

    @field_validator("history_max_messages")
    @classmethod
    def validate_history_max_messages(cls, v: int) -> int:
        if not (2 <= v <= 18):
            raise ValueError("HISTORY_MAX_MESSAGES must be between 2 and 18.")
        if v % 2 != 0:
            raise ValueError(
                "HISTORY_MAX_MESSAGES must be an even number so history starts with a user message."
            )
        return v

    @model_validator(mode="after")
    def validate_conversation_limits(self) -> "Settings":
        if self.conversation_max_messages < self.history_max_messages:
            raise ValueError(
                f"CONVERSATION_MAX_MESSAGES ({self.conversation_max_messages}) must be "
                f"at least HISTORY_MAX_MESSAGES ({self.history_max_messages})."
            )
        return self

    @model_validator(mode="after")
    def validate_rag_settings(self) -> "Settings":
        """Validate RAG configuration bounds per project technical standards."""
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                "rag_chunk_overlap must be strictly less than rag_chunk_size"
            )
        if self.rag_top_k_default > self.rag_top_k_max:
            raise ValueError("rag_top_k_default cannot exceed rag_top_k_max")
        if not (0.0 <= self.rag_min_score <= 1.0):
            raise ValueError("rag_min_score must be between 0.0 and 1.0")
        return self

    def require_api_key(self) -> None:
        """Fail fast with a clear message when the key is missing or blank."""
        key = (
            self.groq_api_key.get_secret_value().strip()
            or self.google_api_key.get_secret_value().strip()
        )
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is missing or empty. Obtain a key from https://console.groq.com/keys "
                "and set GROQ_API_KEY in your local .env file."
            )


@lru_cache
def get_settings() -> Settings:
    """Return the one cached Settings object."""
    return Settings()
