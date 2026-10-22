import pytest

from app.services.conversation_service import get_system_stats
from tests.fakes import InMemoryConversationRepo, InMemoryMessageRepo


@pytest.mark.asyncio
async def test_get_system_stats_fakes():
    conv_repo = InMemoryConversationRepo()
    msg_repo = InMemoryMessageRepo()

    # Empty state checks
    stats = await get_system_stats(conv_repo, msg_repo)
    assert stats["conversations"]["active"] == 0
    assert stats["messages"]["total"] == 0

    # Populated state
    c1 = await conv_repo.create("Active")
    await conv_repo.record_turn(
        c1["id"],
        2,
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        "msg",
    )

    c2 = await conv_repo.create("Archived")
    await conv_repo.update(c2["id"], archived=True)
    await conv_repo.record_turn(
        c2["id"],
        1,
        {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
        "msg2",
    )

    await msg_repo.insert_many(
        [
            {"conversation_id": c1["id"], "seq": 1, "role": "user", "mode": "chat"},
            {
                "conversation_id": c1["id"],
                "seq": 2,
                "role": "assistant",
                "mode": "chat",
                "feedback": {"rating": "up"},
            },
            {"conversation_id": c2["id"], "seq": 1, "role": "user", "mode": "tools"},
        ]
    )

    stats2 = await get_system_stats(conv_repo, msg_repo)
    assert stats2["conversations"]["active"] == 1
    assert stats2["conversations"]["archived"] == 1
    assert stats2["tokens"]["total_tokens"] == 450
    assert stats2["messages"]["total"] == 3
    assert stats2["messages"]["by_mode"].get("chat") == 2
    assert stats2["messages"]["by_mode"].get("tools") == 1
    assert stats2["feedback"].get("up") == 1
    assert stats2["feedback"].get("down", 0) == 0
