import pytest

from app.core.errors import AppError
from app.services.conversation_service import (
    delete_message_feedback,
    set_message_feedback,
)
from tests.fakes import InMemoryConversationRepo, InMemoryMessageRepo


@pytest.mark.asyncio
async def test_feedback_rules_complete():
    conv_repo = InMemoryConversationRepo()
    msg_repo = InMemoryMessageRepo()

    conv = await conv_repo.create("Test")
    cid = conv["id"]
    inserted = await msg_repo.insert_many(
        [
            {"conversation_id": cid, "seq": 1, "role": "user", "content": "hi"},
            {"conversation_id": cid, "seq": 2, "role": "assistant", "content": "hello"},
        ]
    )
    asst_id = inserted[1]["id"]
    user_id = inserted[0]["id"]

    # 1. Success, Replace, and updated_at untouched
    initial_updated = (await conv_repo.get(cid))["updated_at"]
    await set_message_feedback(cid, asst_id, "up", "good", conv_repo, msg_repo)
    await set_message_feedback(cid, asst_id, "down", None, conv_repo, msg_repo)
    assert (await conv_repo.get(cid))["updated_at"] == initial_updated  # Untouched

    # 2. Rejections (400 / 404)
    with pytest.raises(AppError) as e:
        await set_message_feedback("bad-conv", asst_id, "up", None, conv_repo, msg_repo)
    assert e.value.code == "CONVERSATION_NOT_FOUND"

    with pytest.raises(AppError) as e:
        await set_message_feedback(cid, "wrong-id", "up", None, conv_repo, msg_repo)
    assert e.value.code == "MESSAGE_NOT_FOUND"

    with pytest.raises(AppError) as e:
        await set_message_feedback(cid, user_id, "up", None, conv_repo, msg_repo)
    assert e.value.code == "FEEDBACK_INVALID_TARGET"

    # 3. Delete idempotent
    await delete_message_feedback(cid, asst_id, conv_repo, msg_repo)
    await delete_message_feedback(
        cid, asst_id, conv_repo, msg_repo
    )  # 2nd time doesn't fail
