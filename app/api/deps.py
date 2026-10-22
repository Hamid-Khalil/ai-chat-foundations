"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


# --- Week 9 MongoDB Repository Providers ---
from app.core.mongo import get_database
from app.repositories.mongo_conversations import MongoConversationRepo
from app.repositories.mongo_messages import MongoMessageRepo


def get_conversation_repo() -> MongoConversationRepo:
    return MongoConversationRepo(get_database())


def get_message_repo() -> MongoMessageRepo:
    return MongoMessageRepo(get_database())


from app.repositories.mongo_documents import MongoDocumentRepo


def get_document_repo() -> MongoDocumentRepo:
    from app.core.mongo import get_database

    db = get_database()
    return MongoDocumentRepo(db.documents)
