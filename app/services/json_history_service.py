"""Thread-safe JSON-backed chat history manager with atomic writes."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.errors import AppError
from app.schemas.chat import ChatMessage
from app.schemas.chat_session import (
    ChatSessionDetail,
    ChatSessionSummary,
    CreateChatResponse,
    StoredMessage,
)

logger = logging.getLogger(__name__)

HISTORY_FILE = "data/chats.json"
HISTORY_TMP_FILE = "data/chats.json.tmp"
_FILE_LOCK = asyncio.Lock()


class JsonHistoryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"chats": {}}, f, indent=2)

    def _read_data_unlocked(self) -> dict:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"chats": {}}

    def _write_data_unlocked(self, data: dict) -> None:
        """Atomic write: write to tmp file first, then replace target."""
        with open(HISTORY_TMP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(HISTORY_TMP_FILE, HISTORY_FILE)

    async def create_chat(self) -> CreateChatResponse:
        chat_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        async with _FILE_LOCK:
            data = self._read_data_unlocked()
            data["chats"][chat_id] = {
                "title": "New Chat",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            self._write_data_unlocked(data)

        return CreateChatResponse(chat_id=chat_id, title="New Chat", created_at=now)

    async def list_chats(self) -> list[ChatSessionSummary]:
        async with _FILE_LOCK:
            data = self._read_data_unlocked()

        summaries = []
        for cid, info in data.get("chats", {}).items():
            summaries.append(
                ChatSessionSummary(
                    chat_id=cid,
                    title=info.get("title", "New Chat"),
                    created_at=info.get("created_at", ""),
                    updated_at=info.get("updated_at", ""),
                    message_count=len(info.get("messages", [])),
                )
            )
        summaries.sort(key=lambda x: x.updated_at, reverse=True)
        return summaries

    async def get_chat(self, chat_id: str) -> ChatSessionDetail:
        async with _FILE_LOCK:
            data = self._read_data_unlocked()

        chat = data.get("chats", {}).get(chat_id)
        if not chat:
            raise AppError("CHAT_NOT_FOUND", "Chat session not found.", 404)

        return ChatSessionDetail(
            chat_id=chat_id,
            title=chat.get("title", "New Chat"),
            created_at=chat.get("created_at", ""),
            updated_at=chat.get("updated_at", ""),
            messages=[StoredMessage(**m) for m in chat.get("messages", [])],
        )

    async def get_context_window(
        self, chat_id: str, max_turns: int = 6
    ) -> list[ChatMessage]:
        """Fetch the last N turns converted to ChatMessage objects for RAG."""
        async with _FILE_LOCK:
            data = self._read_data_unlocked()

        chat = data.get("chats", {}).get(chat_id)
        if not chat:
            raise AppError("CHAT_NOT_FOUND", "Chat session not found.", 404)

        raw_msgs = chat.get("messages", [])[-max_turns:]
        return [ChatMessage(role=m["role"], content=m["content"]) for m in raw_msgs]

    async def append_turn(
        self,
        chat_id: str,
        user_text: str,
        assistant_text: str,
        sources: list | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        async with _FILE_LOCK:
            data = self._read_data_unlocked()
            chat = data.get("chats", {}).get(chat_id)
            if not chat:
                raise AppError("CHAT_NOT_FOUND", "Chat session not found.", 404)

            if chat.get("title") == "New Chat":
                chat["title"] = user_text[:30].strip() + (
                    "..." if len(user_text) > 30 else ""
                )

            chat["updated_at"] = now
            chat["messages"].append(
                {"role": "user", "content": user_text, "timestamp": now}
            )

            serialized_sources = (
                [s.model_dump() if hasattr(s, "model_dump") else s for s in sources]
                if sources
                else []
            )

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "timestamp": now,
                    "sources": serialized_sources,
                }
            )

            self._write_data_unlocked(data)

    async def delete_chat(self, chat_id: str) -> None:
        async with _FILE_LOCK:
            data = self._read_data_unlocked()
            if chat_id not in data.get("chats", {}):
                raise AppError("CHAT_NOT_FOUND", "Chat session not found.", 404)
            del data["chats"][chat_id]
            self._write_data_unlocked(data)


def get_json_history_service() -> JsonHistoryService:
    return JsonHistoryService()
