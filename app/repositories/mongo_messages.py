from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import handle_db_errors
from app.core.pagination import utcnow


def map_msg(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc:
        return None
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    return out


class MongoMessageRepo:
    def __init__(self, db: AsyncDatabase):
        self.coll = db.messages

    @handle_db_errors
    async def insert_many(self, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return
        now = utcnow()
        for d in docs:
            d["created_at"] = now
            d["schema_version"] = 1
        await self.coll.insert_many(docs)

    @handle_db_errors
    async def recent(self, conversation_id: str, limit: int) -> List[Dict[str, Any]]:
        cursor = (
            self.coll.find({"conversation_id": conversation_id})
            .sort("seq", -1)
            .limit(limit)
        )
        docs = await cursor.to_list(None)
        docs.reverse()
        return [map_msg(d) for d in docs if d is not None]  # type: ignore[misc]

    @handle_db_errors
    async def list_after(
        self, conversation_id: str, after_seq: int, limit: int
    ) -> List[Dict[str, Any]]:
        cursor = (
            self.coll.find(
                {"conversation_id": conversation_id, "seq": {"$gt": after_seq}}
            )
            .sort("seq", 1)
            .limit(limit)
        )
        return [map_msg(d) for d in await cursor.to_list(None) if d is not None]  # type: ignore[misc]

    @handle_db_errors
    async def get(
        self, conversation_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        if not ObjectId.is_valid(message_id):
            return None
        return map_msg(
            await self.coll.find_one(
                {"_id": ObjectId(message_id), "conversation_id": conversation_id}
            )
        )

    @handle_db_errors
    async def delete_for_conversation(self, conversation_id: str) -> int:
        res = await self.coll.delete_many({"conversation_id": conversation_id})
        return int(res.deleted_count)

    @handle_db_errors
    async def delete_by_seqs(self, conversation_id: str, seqs: List[int]) -> None:
        if not seqs:
            return
        await self.coll.delete_many(
            {"conversation_id": conversation_id, "seq": {"$in": seqs}}
        )

    @handle_db_errors
    async def set_feedback(
        self,
        conversation_id: str,
        message_id: str,
        feedback_or_none: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not ObjectId.is_valid(message_id):
            return None
        doc = await self.coll.find_one_and_update(
            {
                "_id": ObjectId(message_id),
                "conversation_id": conversation_id,
                "role": "assistant",
            },
            {"$set": {"feedback": feedback_or_none}},
            return_document=ReturnDocument.AFTER,
        )
        return map_msg(doc)

    async def get_stats(self) -> Dict[str, Any]:
        total_docs = await (await self.coll.aggregate([{"$count": "total"}])).to_list(
            length=1
        )
        total = total_docs[0]["total"] if total_docs else 0

        by_mode = {
            d["_id"]: d["count"]
            for d in await (
                await self.coll.aggregate(
                    [
                        {"$match": {"mode": {"$exists": True}}},
                        {"$group": {"_id": "$mode", "count": {"$sum": 1}}},
                    ]
                )
            ).to_list(length=None)
            if d["_id"]
        }

        by_role = {
            d["_id"]: d["count"]
            for d in await (
                await self.coll.aggregate(
                    [{"$group": {"_id": "$role", "count": {"$sum": 1}}}]
                )
            ).to_list(length=None)
            if d["_id"]
        }

        feedback = {
            d["_id"]: d["count"]
            for d in await (
                await self.coll.aggregate(
                    [
                        {
                            "$match": {
                                "feedback": {"$ne": None},
                                "feedback.rating": {"$exists": True},
                            }
                        },
                        {"$group": {"_id": "$feedback.rating", "count": {"$sum": 1}}},
                    ]
                )
            ).to_list(length=None)
            if d["_id"]
        }

        res_by_mode = {"chat": 0, "tools": 0, "rag": 0}
        res_by_mode.update(by_mode)
        res_by_role = {"user": 0, "assistant": 0}
        res_by_role.update(by_role)
        res_feedback = {"up": 0, "down": 0}
        res_feedback.update(feedback)
        return {
            "total": total,
            "by_mode": res_by_mode,
            "by_role": res_by_role,
            "feedback": res_feedback,
        }
