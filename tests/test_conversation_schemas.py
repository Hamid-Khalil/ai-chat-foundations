import pytest
from pydantic import ValidationError

from app.schemas.conversations import (
    ConversationCreate,
    ConversationUpdate,
    PersistedChatRequest,
)


def test_conversation_create_rules():
    with pytest.raises(ValidationError):
        ConversationCreate(title="")
    with pytest.raises(ValidationError):
        ConversationCreate(title="x" * 101)
    with pytest.raises(ValidationError):
        ConversationCreate(**{"$ne": None})  # Operator dict rejected
    with pytest.raises(ValidationError):
        ConversationCreate(unknown="bad")  # Unknown field rejected


def test_conversation_update_rules():
    with pytest.raises(ValidationError):
        ConversationUpdate()  # Requires at least one field
    with pytest.raises(ValidationError):
        ConversationUpdate(**{"$set": {"archived": True}})  # Operator rejected


def test_persisted_chat_request_rules():
    with pytest.raises(ValidationError):
        PersistedChatRequest(content="hi", role="user")  # role rejected
    with pytest.raises(ValidationError):
        PersistedChatRequest(content="hi", history=[])  # history rejected
    with pytest.raises(ValidationError):
        PersistedChatRequest(content="hi", mode="tools")  # missing tools
    with pytest.raises(ValidationError):
        PersistedChatRequest(content="x" * 501, mode="rag")  # rag length limit
