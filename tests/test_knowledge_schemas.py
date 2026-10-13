import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatMessage
from app.schemas.knowledge import SearchRequest
from app.schemas.rag import RagAskRequest, RagMessagesRequest


def test_search_request_valid():
    req = SearchRequest(query="valid query", top_k=4, min_score=0.5)
    assert req.query == "valid query"
    assert req.top_k == 4
    assert req.min_score == 0.5


def test_search_request_bounds():
    # blank query
    with pytest.raises(ValidationError):
        SearchRequest(query="")
    # query > 500
    with pytest.raises(ValidationError):
        SearchRequest(query="a" * 501)
    # top_k < 1 or > 8
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=9)
    # min_score < 0.0 or > 1.0
    with pytest.raises(ValidationError):
        SearchRequest(query="test", min_score=-0.1)
    with pytest.raises(ValidationError):
        SearchRequest(query="test", min_score=1.1)


def test_rag_ask_request_bounds():
    req = RagAskRequest(question="valid question", top_k=5)
    assert req.question == "valid question"
    assert req.top_k == 5

    with pytest.raises(ValidationError):
        RagAskRequest(question="")
    with pytest.raises(ValidationError):
        RagAskRequest(question="q" * 501)
    with pytest.raises(ValidationError):
        RagAskRequest(question="test", top_k=0)
    with pytest.raises(ValidationError):
        RagAskRequest(question="test", top_k=9)
    with pytest.raises(ValidationError):
        RagAskRequest(question="test", document_ids=["id"] * 21)


def test_rag_messages_request_rules():
    # Valid conversation
    req = RagMessagesRequest(messages=[ChatMessage(role="user", content="Hello")])
    assert len(req.messages) == 1

    # Empty list
    with pytest.raises(ValidationError):
        RagMessagesRequest(messages=[])

    # Rejects system role
    with pytest.raises(ValidationError):
        RagMessagesRequest(messages=[ChatMessage(role="system", content="sys")])

    # Rejects non-alternating
    with pytest.raises(ValidationError):
        RagMessagesRequest(
            messages=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="user", content="Hi again"),
            ]
        )

    # Rejects ending with assistant
    with pytest.raises(ValidationError):
        RagMessagesRequest(
            messages=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello"),
            ]
        )

    # Rejects last user message over 500 characters
    with pytest.raises(ValidationError):
        RagMessagesRequest(
            messages=[
                ChatMessage(role="user", content="a" * 501),
            ]
        )
