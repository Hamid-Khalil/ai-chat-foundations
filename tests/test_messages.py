from langchain.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.chatbot import SYSTEM_PROMPT
from app.schemas.chat import ChatMessage
from app.services.chat_service import to_langchain_messages

HISTORY = [
    ChatMessage(role="user", content="My name is Sara."),
    ChatMessage(role="assistant", content="Nice to meet you, Sara!"),
    ChatMessage(role="user", content="What is my name?"),
]


def test_first_item_is_system_message_with_system_prompt() -> None:
    result = to_langchain_messages(HISTORY)
    assert isinstance(result[0], SystemMessage)
    assert result[0].text == SYSTEM_PROMPT


def test_roles_map_to_langchain_types() -> None:
    result = to_langchain_messages(HISTORY)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], HumanMessage)


def test_order_and_content_are_preserved() -> None:
    result = to_langchain_messages(HISTORY)
    assert [m.text for m in result[1:]] == [m.content for m in HISTORY]


def test_length_is_history_plus_one() -> None:
    assert len(to_langchain_messages(HISTORY)) == len(HISTORY) + 1
