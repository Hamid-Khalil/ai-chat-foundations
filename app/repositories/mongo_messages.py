from typing import List, Optional, Dict, Any
from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from app.core.pagination import utcnow
from app.core.errors import from_db_exception

def map_msg(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc: return None
    if "_id" in doc: doc["id"] = str(doc.pop("_id"))
    return doc

class MongoMessageRepo:
    def __init__(self, db: AsyncDatabase):
        self.coll = db.messages

    async def insert_many(self, docs: List[Dict[str, Any]]) -> None:
        if not docs: return
        try:
            for d in docs:
                d["created_at"] = utcnow()
                d["schema_version"] = 1
            await self.coll.insert_many(docs)
        except Exception as e: raise from_db_exception(e)

    async def recent(self, conversation_id: str, limit: int) -> List[Dict[str, Any]]:
        try:
            cursor = self.coll.find({"conversation_id": conversation_id}).sort("seq", -1).limit(limit)
            docs = await cursor.to_list(None)
            docs.reverse()
            return [map_msg(d) for d in docs]
        except Exception as e: raise from_db_exception(e)

    async def list_after(self, conversation_id: str, after_seq: int, limit: int) -> List[Dict[str, Any]]:
        try:
            cursor = self.coll.find({"conversation_id": conversation_id, "seq": {"$gt": after_seq}}).sort("seq", 1).limit(limit)
            return [map_msg(d) for d in await cursor.to_list(None)]
        except Exception as e: raise from_db_exception(e)

    async def get(self, conversation_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        if not ObjectId.is_valid(message_id): return None
        try:
            return map_msg(await self.coll.find_one({"_id": ObjectId(message_id), "conversation_id": conversation_id}))
        except Exception as e: raise from_db_exception(e)

    async def delete_for_conversation(self, conversation_id: str) -> int:
        try:
            return (await self.coll.delete_many({"conversation_id": conversation_id})).deleted_count
        except Exception as e: raise from_db_exception(e)

    async def delete_by_seqs(self, conversation_id: str, seqs: List[int]) -> None:
        if not seqs: return
        try:
            await self.coll.delete_many({"conversation_id": conversation_id, "seq": {"$in": seqs}})
        except Exception as e: raise from_db_exception(e)

    async def set_feedback(self, conversation_id: str, message_id: str, feedback_or_none: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not ObjectId.is_valid(message_id): return None
        try:
            doc = await self.coll.find_one_and_update(
                {"_id": ObjectId(message_id), "conversation_id": conversation_id, "role": "assistant"},
                {"$set": {"feedback": feedback_or_none}},
                return_document=ReturnDocument.AFTER
            )
            return map_msg(doc)
        except Exception as e: raise from_db_exception(e)
