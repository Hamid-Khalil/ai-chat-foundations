import uuid
from typing import Any, Dict, List, Optional

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import handle_db_errors
from app.core.pagination import decode_cursor, utcnow


def map_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc:
        return None
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    return out


class MongoConversationRepo:
    def __init__(self, db: AsyncDatabase):
        self.coll = db.conversations

    @handle_db_errors
    async def create(self, title: str) -> Dict[str, Any]:
        now = utcnow()
        doc: Dict[str, Any] = {
            "_id": uuid.uuid4().hex[:32],
            "title": title,
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
        await self.coll.insert_one(doc)
        return map_doc(doc)  # type: ignore[return-value]

    @handle_db_errors
    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        return map_doc(await self.coll.find_one({"_id": id}))

    @handle_db_errors
    async def list(
        self, archived: bool, limit: int, cursor: Optional[str]
    ) -> List[Dict[str, Any]]:
        filter_q: Dict[str, Any] = {"archived": archived}
        if cursor:
            dt, cid = decode_cursor(cursor)
            filter_q["$or"] = [
                {"updated_at": {"$lt": dt}},
                {"updated_at": dt, "_id": {"$lt": cid}},
            ]
        cursor_obj = (
            self.coll.find(filter_q)
            .sort([("updated_at", -1), ("_id", -1)])
            .limit(limit)
        )
        return [map_doc(d) for d in await cursor_obj.to_list(None) if d is not None]  # type: ignore[misc]

    @handle_db_errors
    async def update(
        self, id: str, title: Optional[str], archived: Optional[bool]
    ) -> Optional[Dict[str, Any]]:
        set_fields: Dict[str, Any] = {"updated_at": utcnow()}
        if title is not None:
            set_fields["title"] = title
        if archived is not None:
            set_fields["archived"] = archived
        doc = await self.coll.find_one_and_update(
            {"_id": id},
            {"$set": set_fields},
            return_document=ReturnDocument.AFTER,
        )
        return map_doc(doc)

    @handle_db_errors
    async def delete(self, id: str) -> bool:
        res = await self.coll.delete_one({"_id": id})
        return res.deleted_count > 0

    @handle_db_errors
    async def reserve_seq(self, id: str, count: int) -> Optional[int]:
        doc = await self.coll.find_one_and_update(
            {"_id": id},
            {"$inc": {"last_seq": count}},
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["last_seq"]) if doc else None

    @handle_db_errors
    async def record_turn(
        self, id: str, added_messages: int, usage: Dict[str, int], preview: str
    ) -> Optional[Dict[str, Any]]:
        doc = await self.coll.find_one_and_update(
            {"_id": id},
            {
                "$inc": {
                    "message_count": added_messages,
                    "usage_totals.input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "usage_totals.output_tokens": int(
                        usage.get("output_tokens", 0) or 0
                    ),
                    "usage_totals.total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
                "$set": {
                    "updated_at": utcnow(),
                    "last_message_preview": preview,
                },
            },
            return_document=ReturnDocument.AFTER,
        )
        return map_doc(doc)

    @handle_db_errors
    async def set_title_if_default(self, id: str, title: str) -> None:
        await self.coll.update_one(
            {"_id": id, "title": "New conversation"},
            {"$set": {"title": title, "updated_at": utcnow()}},
        )

    async def get_stats(self) -> Dict[str, Any]:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "active": {
                        "$sum": {"$cond": [{"$eq": ["$archived", False]}, 1, 0]}
                    },
                    "archived": {
                        "$sum": {"$cond": [{"$eq": ["$archived", True]}, 1, 0]}
                    },
                    "input_tokens": {"$sum": "$usage_totals.input_tokens"},
                    "output_tokens": {"$sum": "$usage_totals.output_tokens"},
                    "total_tokens": {"$sum": "$usage_totals.total_tokens"},
                }
            }
        ]
        cursor = await self.coll.aggregate(pipeline)
        docs = await cursor.to_list(length=1)
        return (
            docs[0]
            if docs
            else {
                "active": 0,
                "archived": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        )
