from typing import Dict, Any
from app.repositories.protocols import ConversationRepo, MessageRepo
from app.core.errors import AppError
from app.core.pagination import encode_cursor

async def create_conversation(repo: ConversationRepo, title: str = None) -> Dict[str, Any]:
    final_title = title if title else "New conversation"
    return await repo.create(title=final_title)

async def get_conversation(repo: ConversationRepo, conversation_id: str) -> Dict[str, Any]:
    conv = await repo.get(conversation_id)
    if not conv:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    return conv

async def list_conversations(repo: ConversationRepo, archived: bool, limit: int, cursor: str = None) -> Dict[str, Any]:
    # Ask for limit + 1 to determine if there is a next page
    rows = await repo.list(archived=archived, limit=limit + 1, cursor=cursor)
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last_item = rows[-1]
        next_cursor = encode_cursor(last_item["updated_at"], last_item["id"])
    return {"items": rows, "next_cursor": next_cursor}

async def update_conversation(repo: ConversationRepo, conversation_id: str, title: str = None, archived: bool = None) -> Dict[str, Any]:
    # Validate existence first
    await get_conversation(repo, conversation_id)
    updated = await repo.update(conversation_id, title=title, archived=archived)
    if not updated:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
    return updated

async def delete_conversation(conv_repo: ConversationRepo, msg_repo: MessageRepo, conversation_id: str) -> Dict[str, Any]:
    # Verify exists
    await get_conversation(conv_repo, conversation_id)
    
    # ORDER MATTERS: Delete messages BEFORE deleting the conversation to prevent orphans
    deleted_count = await msg_repo.delete_for_conversation(conversation_id)
    success = await conv_repo.delete(conversation_id)
    if not success:
        raise AppError.from_code("CONVERSATION_NOT_FOUND")
        
    return {"conversation_id": conversation_id, "deleted_messages": deleted_count}

async def list_messages(conv_repo: ConversationRepo, msg_repo: MessageRepo, conversation_id: str, after_seq: int, limit: int) -> Dict[str, Any]:
    await get_conversation(conv_repo, conversation_id)
    
    rows = await msg_repo.list_after(conversation_id, after_seq=after_seq, limit=limit + 1)
    next_seq = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_seq = rows[-1]["seq"]
        
    return {"items": rows, "next_after_seq": next_seq}
