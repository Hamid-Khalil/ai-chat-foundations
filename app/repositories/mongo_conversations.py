import uuid
from typing import List, Optional, Dict, Any
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from app.core.pagination import utcnow, decode_cursor
from app.core.errors import from_db_exception, AppError

def map_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc: return None
    if "_id" in doc: doc["id"] = str(doc.pop("_id"))
    return doc

class MongoConversationRepo:
    def __init__(self, db: AsyncDatabase):
        self.coll = db.conversations

    async def create(self, title: str) -> Dict[str, Any]:
        try:
            doc = {
                "_id": uuid.uuid4().hex[:32],
                "title": title,
                "archived": False,
                "created_at": utcnow(),
                "updated_at": utcnow(),
                "last_seq": 0,
                "message_count": 0,
                "last_message_preview": "",
                "usage_totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "schema_version": 1
            }
            await self.coll.insert_one(doc)
            return map_doc(doc)
        except Exception as e: raise from_db_exception(e)

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        try:
            return map_doc(await self.coll.find_one({"_id": id}))
        except Exception as e: raise from_db_exception(e)

    async def list(self, archived: bool, limit: int, cursor: Optional[str]) -> List[Dict[str, Any]]:
        try:
            filter_q: Dict[str, Any] = {"archived": archived}
            if cursor:
                dt, _id = decode_cursor(cursor)
                filter_q["$or"] = [
                    {"updated_at": {"$lt": dt}},
                    {"updated_at": dt, "_id": {"$lt": _id}}
                ]
            cursor_obj = self.coll.find(filter_q).sort([("updated_at", -1), ("_id", -1)]).limit(limit)
            return [map_doc(d) for d in await cursor_obj.to_list(None)]
        except AppError: raise
        except Exception as e: raise from_db_exception(e)

    async def update(self, id: str, title: Optional[str], archived: Optional[bool]) -> Optional[Dict[str, Any]]:
        try:
            set_fields = {"updated_at": utcnow()}
            if title is not None: set_fields["title"] = title
            if archived is not None: set_fields["archived"] = archived
            doc = await self.coll.find_one_and_update({"_id": id}, {"$set": set_fields}, return_document=ReturnDocument.AFTER)
            return map_doc(doc)
        except Exception as e: raise from_db_exception(e)

    async def delete(self, id: str) -> bool:
        try:
            res = await self.coll.delete_one({"_id": id})
            return res.deleted_count > 0
        except Exception as e: raise from_db_exception(e)

    async def reserve_seq(self, id: str, count: int) -> Optional[int]:
        try:
            doc = await self.coll.find_one_and_update({"_id": id}, {"$inc": {"last_seq": count}}, return_document=ReturnDocument.AFTER)
            return doc["last_seq"] if doc else None
        except Exception as e: raise from_db_exception(e)

    async def record_turn(self, id: str, added_messages: int, usage: Dict[str, int], preview: str) -> Optional[Dict[str, Any]]:
        try:
            doc = await self.coll.find_one_and_update(
                {"_id": id},
                {
                    "$inc": {"message_count": added_messages, "usage_totals.input_tokens": usage.get("input_tokens", 0), "usage_totals.output_tokens": usage.get("output_tokens", 0), "usage_totals.total_tokens": usage.get("total_tokens", 0)},
                    "$set": {"updated_at": utcnow(), "last_message_preview": preview}
                },
                return_document=ReturnDocument.AFTER
            )
            return map_doc(doc)
        except Exception as e: raise from_db_exception(e)

    async def set_title_if_default(self, id: str, title: str) -> None:
        try:
            await self.coll.update_one({"_id": id, "title": "New conversation"}, {"$set": {"title": title, "updated_at": utcnow()}})
        except Exception as e: raise from_db_exception(e)
