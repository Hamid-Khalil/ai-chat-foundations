import asyncio
import os
from uuid import uuid4

import pytest
from pymongo import AsyncMongoClient

from app.core.pagination import encode_cursor, utcnow
from app.repositories.indexes import ensure_indexes
from app.repositories.mongo_conversations import MongoConversationRepo
from app.repositories.mongo_documents import MongoDocumentRepo
from app.repositories.mongo_messages import MongoMessageRepo

pytestmark = pytest.mark.integration


@pytest.fixture
async def test_db():
    uri = os.getenv("MONGODB_TEST_URI")
    if not uri:
        pytest.skip("MONGODB_TEST_URI not set")
    db_name = uri.split("/")[-1].split("?")[0]
    if not db_name.endswith("_test"):
        raise RuntimeError(
            "CRITICAL: Test database name must end with '_test' to prevent real data loss."
        )

    client = AsyncMongoClient(uri, tz_aware=True)
    await client.drop_database(db_name)
    db = client[db_name]
    await ensure_indexes(db)
    yield db
    await client.drop_database(db_name)
    await client.close()


@pytest.mark.asyncio
async def test_full_integration_rules(test_db):
    conv_repo = MongoConversationRepo(test_db)
    msg_repo = MongoMessageRepo(test_db)
    doc_repo = MongoDocumentRepo(test_db)

    # 1. ensure_indexes twice is harmless
    await ensure_indexes(test_db)

    # 2. Duplicate (conversation_id, seq) refused
    await msg_repo.insert_many(
        [
            {
                "conversation_id": "c1",
                "seq": 1,
                "role": "user",
                "content": "h1",
                "mode": "chat",
                "metadata": None,
                "feedback": None,
            }
        ]
    )
    with pytest.raises(Exception):
        await msg_repo.insert_many(
            [
                {
                    "conversation_id": "c1",
                    "seq": 1,
                    "role": "user",
                    "content": "h1-dup",
                    "mode": "chat",
                    "metadata": None,
                    "feedback": None,
                }
            ]
        )

    # 3. 20 concurrent reserve_seq calls return 20 different numbers
    conv = await conv_repo.create("Test")
    cid = conv["id"]
    # Verify stored datetimes come back timezone-aware
    assert conv["created_at"].tzinfo is not None
    assert conv["updated_at"].tzinfo is not None

    results = await asyncio.gather(*(conv_repo.reserve_seq(cid, 1) for _ in range(20)))
    assert len(set(results)) == 20

    # 4. recent returns last N oldest first & list_after is ordered and respects limit
    await msg_repo.insert_many(
        [
            {
                "conversation_id": cid,
                "seq": i,
                "role": "user" if i % 2 == 1 else "assistant",
                "content": f"msg-{i}",
                "mode": "chat",
                "metadata": None,
                "feedback": None,
            }
            for i in range(1, 7)
        ]
    )
    recent_msgs = await msg_repo.recent(cid, limit=4)
    assert [m["seq"] for m in recent_msgs] == [3, 4, 5, 6]

    after_msgs = await msg_repo.list_after(cid, after_seq=2, limit=3)
    assert [m["seq"] for m in after_msgs] == [3, 4, 5]

    # 5. set_feedback on a user message returns None; on assistant succeeds
    user_msg_id = recent_msgs[0]["id"]  # seq 3 (user)
    asst_msg_id = recent_msgs[1]["id"]  # seq 4 (assistant)
    assert (
        await msg_repo.set_feedback(
            cid, user_msg_id, {"rating": "up", "comment": None, "created_at": utcnow()}
        )
        is None
    )
    updated_asst = await msg_repo.set_feedback(
        cid, asst_msg_id, {"rating": "up", "comment": "good", "created_at": utcnow()}
    )
    assert updated_asst is not None
    assert updated_asst["feedback"]["rating"] == "up"

    # 6. Keyset pagination over 7 conversations with identical updated_at never repeats or skips
    now = utcnow()
    for i in range(7):
        await test_db.conversations.insert_one(
            {
                "_id": uuid4().hex,
                "title": f"C{i}",
                "archived": False,
                "created_at": now,
                "updated_at": now,
                "last_seq": 0,
                "message_count": 0,
                "last_message_preview": "",
                "usage_totals": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "schema_version": 1,
            }
        )
    seen, cursor = set(), None
    for _ in range(5):
        items = await conv_repo.list(archived=False, limit=2, cursor=cursor)
        for item in items:
            assert item["id"] not in seen
            seen.add(item["id"])
        if not items:
            break
        cursor = encode_cursor(items[-1]["updated_at"], items[-1]["id"])
    assert len(seen) >= 8  # 7 inserted + 1 created earlier

    # 7. set_title_if_default only works once
    default_conv = await conv_repo.create("New conversation")
    default_cid = default_conv["id"]
    await conv_repo.set_title_if_default(default_cid, "First Title")
    await conv_repo.set_title_if_default(default_cid, "Second Title")
    assert (await conv_repo.get(default_cid))["title"] == "First Title"

    # 8. record_turn adds up
    await conv_repo.record_turn(
        cid, 6, {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}, "prev"
    )
    fetched_conv = await conv_repo.get(cid)
    assert fetched_conv["message_count"] == 6
    assert fetched_conv["usage_totals"]["total_tokens"] == 30

    # 9. Task 4.4 Stats aggregation check
    c_stats = await conv_repo.get_stats()
    m_stats = await msg_repo.get_stats()
    assert c_stats["active"] >= 8
    assert c_stats["total_tokens"] == 30
    assert m_stats["total"] == 7  # 6 in cid + 1 in c1
    assert m_stats["feedback"]["up"] == 1

    # 10. delete_for_conversation count
    deleted = await msg_repo.delete_for_conversation("c1")
    assert deleted == 1

    # 11. Malformed message id returns None
    assert await msg_repo.get(cid, "not-an-object-id") is None

    # 12. Registry sha256 unique
    doc1 = {
        "_id": uuid4().hex,
        "filename": "f1.txt",
        "sha256": "hash_unique_1",
        "chunks": 1,
        "characters": 10,
        "schema_version": 1,
    }
    doc2 = {
        "_id": uuid4().hex,
        "filename": "f2.txt",
        "sha256": "hash_unique_1",
        "chunks": 2,
        "characters": 20,
        "schema_version": 1,
    }
    await doc_repo.insert(doc1)
    with pytest.raises(Exception):
        await doc_repo.insert(doc2)
