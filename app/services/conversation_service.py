try:
    from app.core.pagination import utcnow
except ImportError:
    from datetime import datetime, timezone

    def utcnow():
        now = datetime.now(timezone.utc)
        return now.replace(microsecond=(now.microsecond // 1000) * 1000)


from typing import Any, Dict

from app.core.errors import AppError
from app.core.pagination import encode_cursor
from app.repositories.protocols import ConversationRepo, MessageRepo


async def create_conversation(
    repo: ConversationRepo, title: str = None
) -> Dict[str, Any]:
    final_title = title if title else "New conversation"
    return await repo.create(title=final_title)


async def get_conversation(
    repo: ConversationRepo, conversation_id: str
) -> Dict[str, Any]:
    conv = await repo.get(conversation_id)
    if not conv:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    return conv


async def list_conversations(
    repo: ConversationRepo, archived: bool, limit: int, cursor: str = None
) -> Dict[str, Any]:
    # Ask for limit + 1 to determine if there is a next page
    rows = await repo.list(archived=archived, limit=limit + 1, cursor=cursor)
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last_item = rows[-1]
        next_cursor = encode_cursor(last_item["updated_at"], last_item["id"])
    return {"items": rows, "next_cursor": next_cursor}


async def update_conversation(
    repo: ConversationRepo,
    conversation_id: str,
    title: str = None,
    archived: bool = None,
) -> Dict[str, Any]:
    # Validate existence first
    await get_conversation(repo, conversation_id)
    updated = await repo.update(conversation_id, title=title, archived=archived)
    if not updated:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    return updated


async def delete_conversation(
    conv_repo: ConversationRepo, msg_repo: MessageRepo, conversation_id: str
) -> Dict[str, Any]:
    # Verify exists
    await get_conversation(conv_repo, conversation_id)

    # ORDER MATTERS: Delete messages BEFORE deleting the conversation to prevent orphans
    deleted_count = await msg_repo.delete_for_conversation(conversation_id)
    success = await conv_repo.delete(conversation_id)
    if not success:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")

    return {"conversation_id": conversation_id, "deleted_messages": deleted_count}


async def list_messages(
    conv_repo: ConversationRepo,
    msg_repo: MessageRepo,
    conversation_id: str,
    after_seq: int,
    limit: int,
) -> Dict[str, Any]:
    await get_conversation(conv_repo, conversation_id)

    rows = await msg_repo.list_after(
        conversation_id, after_seq=after_seq, limit=limit + 1
    )
    next_seq = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_seq = rows[-1]["seq"]

    return {"items": rows, "next_after_seq": next_seq}


import logging
import re
import time
from typing import Any, Dict, List

from app.core.config import get_settings
from app.schemas.chat import ChatMessage
from app.schemas.conversations import PersistedChatRequest
from app.services import chat_service

logger = logging.getLogger(__name__)


def history_for_model(stored_messages: list[dict], per_message_limit: int) -> list[dict]:
    clean_history = []
    for msg in stored_messages:
        # Drop tool_calls and metadata so Groq doesn't crash on unresolved tools
        clean_history.append({
            "role": msg["role"],
            "content": msg["content"][:per_message_limit] if msg["content"] else ""
        })
    
    # Drop leading assistant messages
    while clean_history and clean_history[0]["role"] == "assistant":
        clean_history.pop(0)
        
    return clean_history

def build_metadata(mode: str, reply: Any) -> Dict[str, Any]:
    usage_dict = {
        "input_tokens": int(getattr(reply.usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(reply.usage, "output_tokens", 0) or 0),
        "reasoning_tokens": getattr(reply.usage, "reasoning_tokens", None),
        "total_tokens": int(getattr(reply.usage, "total_tokens", 0) or 0),
    }
    meta = {
        "model": getattr(reply, "model", "unknown"),
        "thinking_level": getattr(reply, "thinking_level", None),
        "finish_reason": getattr(reply, "finish_reason", "STOP"),
        "usage": usage_dict,
    }
    if mode == "tools":
        meta["rounds"] = getattr(reply, "rounds", 1)
        tc = getattr(reply, "trace", getattr(reply, "tool_calls", []))
        meta["tool_calls"] = [
            t.model_dump(mode="json") if hasattr(t, "model_dump") else t for t in tc
        ]
    elif mode == "rag":
        meta["answer_found"] = getattr(reply, "answer_found", False)
        meta["answered_from_context"] = getattr(reply, "answered_from_context", False)
        meta["standalone_question"] = getattr(reply, "standalone_question", None)

        sources = getattr(reply, "sources", [])
        meta["sources"] = [
            s.model_dump(mode="json") if hasattr(s, "model_dump") else s
            for s in sources
        ]

        retrieval = getattr(reply, "retrieval", None)
        if hasattr(retrieval, "model_dump"):
            meta["retrieval"] = retrieval.model_dump(mode="json")
        elif hasattr(retrieval, "__dict__"):
            meta["retrieval"] = retrieval.__dict__
        else:
            meta["retrieval"] = retrieval

    return meta


async def chat(
    conversation_id: str,
    request: PersistedChatRequest,
    conversations: ConversationRepo,
    messages: MessageRepo,
) -> Dict[str, Any]:
    started = time.perf_counter()
    settings = get_settings()

    conv = await conversations.get(conversation_id)
    if not conv:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    if conv.get("archived"):
        raise AppError.from_code("CONVERSATION_ARCHIVED")

    if conv.get("message_count", 0) + 2 > settings.conversation_max_messages:
        raise AppError.from_code("CONVERSATION_FULL")

    stored_recent = await messages.recent(
        conversation_id, settings.history_max_messages
    )
    model_history = history_for_model(stored_recent, settings.chat_max_message_chars)
    model_history.append({"role": "user", "content": request.content})

    # Dispatch by mode
    if request.mode == "chat":
        from app.services import chat_service

        reply = await chat_service.converse(
            [ChatMessage(**m) for m in model_history], thinking_level=request.thinking_level
        )
    elif request.mode == "tools":
        from app.api.v1.assistant import AssistantReply
        from app.services.tool_runner import run_tool_loop
        from app.tools import get_tool_catalog

        catalog_names = [t["name"] for t in get_tool_catalog()]
        for t in request.tools:
            if t not in catalog_names:
                raise AppError.from_code("TOOL_UNKNOWN")

        result_dict = await run_tool_loop(
            history=[ChatMessage(**m) for m in model_history],
            thinking_level=request.thinking_level or "low",
            allowed_tools=request.tools,
        )
        reply = AssistantReply(**result_dict)
        # Dynamic alias removed for Pydantic V2 compatibility
    elif request.mode == "rag":
        from app.services.rag_service import get_rag_service

        thinking = request.thinking_level or getattr(
            settings, "gemini_thinking_level", "low"
        )
        reply = await get_rag_service().answer_conversation(
            [ChatMessage(**m) for m in model_history], request.top_k, request.document_ids, thinking
        )
    else:
        raise AppError.from_code("INTERNAL_ERROR")

    new_last_seq = await conversations.reserve_seq(conversation_id, 2)
    if new_last_seq is None:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    user_seq = new_last_seq - 1
    assistant_seq = new_last_seq

    assistant_metadata = build_metadata(request.mode, reply)

    user_doc = {
        "conversation_id": conversation_id,
        "seq": user_seq,
        "role": "user",
        "content": request.content,
        "mode": request.mode,
        "metadata": None,
        "feedback": None,
    }

    reply_text = getattr(reply, "reply", "") or getattr(reply, "content", "")
    assistant_doc = {
        "conversation_id": conversation_id,
        "seq": assistant_seq,
        "role": "assistant",
        "content": reply_text,
        "mode": request.mode,
        "metadata": assistant_metadata,
        "feedback": None,
    }

    try:
        await messages.insert_many([user_doc, assistant_doc])
    except Exception:
        try:
            await messages.delete_by_seqs(conversation_id, [user_seq, assistant_seq])
        except Exception:
            pass
        logger.error("Failed to insert chat turn")
        raise

    if "_id" in user_doc:
        user_doc["id"] = str(user_doc.pop("_id"))
    if "_id" in assistant_doc:
        assistant_doc["id"] = str(assistant_doc.pop("_id"))

    preview = reply_text[:120]
    await conversations.record_turn(
        conversation_id,
        added_messages=2,
        usage=assistant_metadata["usage"],
        preview=preview,
    )
    collapsed_title = re.sub(r"\s+", " ", request.content).strip()[:60]
    await conversations.set_title_if_default(conversation_id, collapsed_title)

    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "persisted_chat conversation_id=%s mode=%s seqs=(%d,%d) duration_ms=%d",
        conversation_id,
        request.mode,
        user_seq,
        assistant_seq,
        duration_ms,
    )

    return {"user_message": user_doc, "assistant_message": assistant_doc}


async def set_message_feedback(
    conversation_id: str,
    message_id: str,
    rating: str,
    comment: Optional[str],
    conversations: ConversationRepo,
    messages: MessageRepo,
) -> Dict[str, Any]:
    # 1. Unknown conversation -> 404 CONVERSATION_NOT_FOUND (allowed on archived)
    conv = await conversations.get(conversation_id)
    if not conv:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")

    # 2. Message does not exist, malformed id, or belongs to different conversation -> 404 MESSAGE_NOT_FOUND
    msg = await messages.get(conversation_id, message_id)
    if not msg:
        raise AppError.from_code("MESSAGE_NOT_FOUND")

    # 3. Cannot provide feedback on a user message -> 400 FEEDBACK_INVALID_TARGET
    if msg.get("role") != "assistant":
        raise AppError.from_code("FEEDBACK_INVALID_TARGET")

    # Feedback does not change the conversation's updated_at (so rating an old answer
    # does not alter the conversation ordering or push the chat to the top of the list).
    feedback_doc = {
        "rating": rating,
        "comment": comment,
        "created_at": utcnow(),
    }
    updated_msg = await messages.set_feedback(conversation_id, message_id, feedback_doc)
    if not updated_msg:
        raise AppError.from_code("MESSAGE_NOT_FOUND")
    return updated_msg


async def delete_message_feedback(
    conversation_id: str,
    message_id: str,
    conversations: ConversationRepo,
    messages: MessageRepo,
) -> Dict[str, Any]:
    # 1. Unknown conversation -> 404 CONVERSATION_NOT_FOUND
    conv = await conversations.get(conversation_id)
    if not conv:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")

    # 2. Check message existence and target validation
    msg = await messages.get(conversation_id, message_id)
    if not msg:
        raise AppError.from_code("MESSAGE_NOT_FOUND")

    if msg.get("role") != "assistant":
        raise AppError.from_code("FEEDBACK_INVALID_TARGET")

    # Feedback does not touch the conversation updated_at; removing non-existent feedback is harmless (200)
    updated_msg = await messages.set_feedback(conversation_id, message_id, None)
    if not updated_msg:
        raise AppError.from_code("MESSAGE_NOT_FOUND")
    return updated_msg


async def get_system_stats(
    conversations: ConversationRepo, messages: MessageRepo
) -> Dict[str, Any]:
    try:
        c_stats = await conversations.get_stats()
        m_stats = await messages.get_stats()
        return {
            "conversations": {
                "active": c_stats.get("active", 0),
                "archived": c_stats.get("archived", 0),
            },
            "messages": {
                "total": m_stats.get("total", 0),
                "by_mode": m_stats.get("by_mode", {}),
                "by_role": m_stats.get("by_role", {}),
            },
            "tokens": {
                "input_tokens": c_stats.get("input_tokens", 0),
                "output_tokens": c_stats.get("output_tokens", 0),
                "total_tokens": c_stats.get("total_tokens", 0),
            },
            "feedback": m_stats.get("feedback", {}),
        }
    except Exception as e:
        if (
            getattr(e, "code", None) == "DB_UNAVAILABLE"
            or "db_unavailable" in str(e).lower()
        ):
            raise AppError.from_code("DB_UNAVAILABLE")
        raise
