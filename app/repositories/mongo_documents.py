from typing import Any, Dict, List, Optional

from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import handle_db_errors
from app.core.pagination import utcnow


class MongoDocumentRepo:
    def __init__(self, target: Any):
        if hasattr(target, "documents"):
            self._col: AsyncCollection = target.documents
        else:
            self._col: AsyncCollection = target

    @handle_db_errors
    async def insert(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc_to_save = dict(doc)
        if "id" in doc_to_save and "_id" not in doc_to_save:
            doc_to_save["_id"] = doc_to_save.pop("id")
        if "created_at" not in doc_to_save:
            doc_to_save["created_at"] = utcnow()
        if "schema_version" not in doc_to_save:
            doc_to_save["schema_version"] = 1

        await self._col.insert_one(doc_to_save)
        doc_to_save["id"] = str(doc_to_save.pop("_id"))
        return doc_to_save

    @handle_db_errors
    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        doc = await self._col.find_one({"_id": id})
        if not doc:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    @handle_db_errors
    async def get_by_sha256(self, sha256: str) -> Optional[Dict[str, Any]]:
        doc = await self._col.find_one({"sha256": sha256})
        if not doc:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    @handle_db_errors
    async def list(self) -> List[Dict[str, Any]]:
        cursor = self._col.find({}).sort("created_at", -1)
        items = await cursor.to_list(None)
        for item in items:
            item["id"] = str(item.pop("_id"))
        return items

    @handle_db_errors
    async def count(self) -> int:
        return await self._col.count_documents({})

    @handle_db_errors
    async def delete(self, id: str) -> bool:
        res = await self._col.delete_one({"_id": id})
        return res.deleted_count > 0
