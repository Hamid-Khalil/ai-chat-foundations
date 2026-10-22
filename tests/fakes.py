import asyncio
from typing import Any, Dict, List, Optional

from app.core.errors import AppError
from app.core.pagination import utcnow


class FakeRepoBase:
    def __init__(self):
        self.call_log = []
        self.fail_on = None
        self._db = {}

    def _check_failure(self, method_name: str):
        if self.fail_on == method_name:
            raise AppError.from_code("DB_UNAVAILABLE")
        self.call_log.append(method_name)


class InMemoryConversationRepo(FakeRepoBase):
    async def get_stats(self) -> Dict[str, Any]:
        self._check_failure("get_stats")
        return {
            "active": sum(1 for d in self._db.values() if not d.get("archived")),
            "archived": sum(1 for d in self._db.values() if d.get("archived")),
            "input_tokens": sum(
                d.get("usage_totals", {}).get("input_tokens", 0)
                for d in self._db.values()
            ),
            "output_tokens": sum(
                d.get("usage_totals", {}).get("output_tokens", 0)
                for d in self._db.values()
            ),
            "total_tokens": sum(
                d.get("usage_totals", {}).get("total_tokens", 0)
                for d in self._db.values()
            ),
        }

    def __init__(self):
        super().__init__()
        self._db: Dict[str, Dict[str, Any]] = {}
        self._seqs: Dict[str, int] = {}

    async def create(self, title: str) -> Dict[str, Any]:
        self._check_failure("create")
        import uuid

        cid = uuid.uuid4().hex
        now = utcnow()
        doc = {
            "id": cid,
            "title": title,
            "archived": False,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_message_preview": "",
            "usage_totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "last_seq": 0,
            "schema_version": 1,
        }
        self._db[cid] = doc
        self._seqs[cid] = 0
        return dict(doc)

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        self._check_failure("get")
        doc = self._db.get(id)
        return dict(doc) if doc else None

    async def list(
        self, archived: bool, limit: int, cursor: Optional[tuple]
    ) -> List[Dict[str, Any]]:
        self._check_failure("list")
        docs = [d for d in self._db.values() if d.get("archived", False) == archived]
        docs.sort(key=lambda x: (x["updated_at"], x["id"]), reverse=True)
        if cursor:
            ct, cid = cursor
            docs = [
                d
                for d in docs
                if d["updated_at"] < ct or (d["updated_at"] == ct and d["id"] < cid)
            ]
        return [dict(d) for d in docs[:limit]]

    async def update(
        self, id: str, title: Optional[str] = None, archived: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        self._check_failure("update")
        if id not in self._db:
            return None
        if title is not None:
            self._db[id]["title"] = title
        if archived is not None:
            self._db[id]["archived"] = archived
        self._db[id]["updated_at"] = utcnow()
        return dict(self._db[id])

    async def delete(self, id: str) -> bool:
        self._check_failure("delete")
        return bool(self._db.pop(id, None))

    async def reserve_seq(self, id: str, count: int) -> Optional[int]:
        self._check_failure("reserve_seq")
        if id not in self._db:
            return None
        self._seqs[id] += count
        return self._seqs[id]

    async def record_turn(
        self, id: str, added_messages: int, usage: Dict[str, int], preview: str
    ) -> Optional[Dict[str, Any]]:
        self._check_failure("record_turn")
        if id not in self._db:
            return None
        doc = self._db[id]
        doc["message_count"] += added_messages
        doc["last_message_preview"] = preview
        doc["updated_at"] = utcnow()
        doc["usage_totals"]["input_tokens"] += usage.get("input_tokens", 0)
        doc["usage_totals"]["output_tokens"] += usage.get("output_tokens", 0)
        doc["usage_totals"]["total_tokens"] += usage.get("total_tokens", 0)
        return dict(doc)

    async def set_title_if_default(
        self, id: str, title: str
    ) -> Optional[Dict[str, Any]]:
        self._check_failure("set_title_if_default")
        if id not in self._db:
            return None
        if self._db[id]["title"] == "New conversation":
            self._db[id]["title"] = title
            self._db[id]["updated_at"] = utcnow()
        return dict(self._db[id])


class InMemoryMessageRepo(FakeRepoBase):
    async def get_stats(self) -> Dict[str, Any]:
        self._check_failure("get_stats")
        total = len(self._db)
        by_mode, by_role, feedback = {}, {}, {}
        for d in self._db:
            if d.get("mode"):
                by_mode[d["mode"]] = by_mode.get(d["mode"], 0) + 1
            if d.get("role"):
                by_role[d["role"]] = by_role.get(d["role"], 0) + 1
            if d.get("feedback") and "rating" in d["feedback"]:
                r = d["feedback"]["rating"]
                feedback[r] = feedback.get(r, 0) + 1

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

    def __init__(self):
        super().__init__()
        self._db: List[Dict[str, Any]] = []

    async def insert_many(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._check_failure("insert_many")
        for doc in docs:
            # Enforce unique index (conversation_id, seq)
            if any(
                d["conversation_id"] == doc["conversation_id"]
                and d["seq"] == doc["seq"]
                for d in self._db
            ):
                raise Exception("Duplicate key error fake")
            import uuid

            d = dict(doc)
            d["id"] = uuid.uuid4().hex[:24]
            d["created_at"] = utcnow()
            d["schema_version"] = 1
            self._db.append(d)
            doc["id"] = d["id"]  # Match PyMongo behavior by updating the input dicts
        return docs

    async def recent(self, conversation_id: str, limit: int) -> List[Dict[str, Any]]:
        self._check_failure("recent")
        docs = [d for d in self._db if d["conversation_id"] == conversation_id]
        docs.sort(key=lambda x: x["seq"])
        return [dict(d) for d in docs[-limit:]]

    async def list_after(
        self, conversation_id: str, after_seq: int, limit: int
    ) -> List[Dict[str, Any]]:
        self._check_failure("list_after")
        docs = [
            d
            for d in self._db
            if d["conversation_id"] == conversation_id and d["seq"] > after_seq
        ]
        docs.sort(key=lambda x: x["seq"])
        return [dict(d) for d in docs[:limit]]

    async def get(
        self, conversation_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        self._check_failure("get")
        for d in self._db:
            if d["id"] == message_id and d["conversation_id"] == conversation_id:
                return dict(d)
        return None

    async def delete_for_conversation(self, conversation_id: str) -> int:
        self._check_failure("delete_for_conversation")
        initial_len = len(self._db)
        self._db = [d for d in self._db if d["conversation_id"] != conversation_id]
        return initial_len - len(self._db)

    async def delete_by_seqs(self, conversation_id: str, seqs: List[int]) -> int:
        self._check_failure("delete_by_seqs")
        initial_len = len(self._db)
        self._db = [
            d
            for d in self._db
            if not (d["conversation_id"] == conversation_id and d["seq"] in seqs)
        ]
        return initial_len - len(self._db)

    async def set_feedback(
        self,
        conversation_id: str,
        message_id: str,
        feedback_or_none: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        self._check_failure("set_feedback")
        for d in self._db:
            if (
                d["id"] == message_id
                and d["conversation_id"] == conversation_id
                and d["role"] == "assistant"
            ):
                d["feedback"] = feedback_or_none
                return dict(d)
        return None


class InMemoryDocumentRepo(FakeRepoBase):
    def __init__(self):
        super().__init__()
        self._db: Dict[str, Dict[str, Any]] = {}

    async def insert(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        self._check_failure("insert")
        if any(
            d.get("sha256") == doc.get("sha256")
            for d in self._db.values()
            if "sha256" in doc
        ):
            raise Exception("Duplicate key error fake")
        import uuid

        did = doc.get("id") or uuid.uuid4().hex
        d = dict(doc)
        d["id"] = did
        self._db[did] = d
        return d

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        self._check_failure("get")
        return dict(self._db[id]) if id in self._db else None

    async def get_by_sha256(self, sha: str) -> Optional[Dict[str, Any]]:
        self._check_failure("get_by_sha256")
        for d in self._db.values():
            if d.get("sha256") == sha:
                return dict(d)
        return None

    async def list(self) -> List[Dict[str, Any]]:
        self._check_failure("list")
        return [dict(d) for d in self._db.values()]

    async def count(self) -> int:
        self._check_failure("count")
        return len(self._db)

    async def delete(self, id: str) -> bool:
        self._check_failure("delete")
        return bool(self._db.pop(id, None))


class FakeAIResponse:
    def __init__(
        self, reply, usage, finish_reason="STOP", model="fake-model", **kwargs
    ):
        self.reply = reply
        self.usage = usage
        self.finish_reason = finish_reason
        self.model = model
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeAIService:
    def __init__(self):
        self.call_args = []
        self.raise_exc = None

    async def chat(self, history, thinking_level):
        if self.raise_exc:
            raise self.raise_exc
        self.call_args.append((history, thinking_level))
        return FakeAIResponse(
            "Test reply", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )

    async def chat_with_tools(self, history, tools, thinking_level):
        if self.raise_exc:
            raise self.raise_exc
        self.call_args.append((history, tools, thinking_level))
        return FakeAIResponse(
            "Tool reply",
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            rounds=1,
            tool_calls=[{"name": "test"}],
        )


class FakeRagService:
    def __init__(self):
        self.call_args = []
        self.raise_exc = None

    async def answer_conversation(self, history, top_k, document_ids, thinking_level):
        if self.raise_exc:
            raise self.raise_exc
        self.call_args.append((history, top_k, document_ids, thinking_level))
        from app.schemas.knowledge import RagReply

        return RagReply(
            reply="Rag reply",
            sources=[{"filename": "test.txt", "document_id": "1", "chunk_index": 0}],
            answer_found=True,
            answered_from_context=True,
            retrieval={"k": 1, "duration_ms": 10.0, "returned": 1},
            standalone_question="standalone",
            model="fake",
            thinking_level=thinking_level,
            finish_reason="STOP",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
