import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.chat import AskRequest, ConversationRequest

# Limits come from Settings (defaults 20 / 4000) so the tests still make sense if a
# developer changes them in .env.
MAX_MESSAGES = get_settings().chat_max_messages
MAX_CHARS = get_settings().chat_max_message_chars


def _alternating(count: int) -> list[dict[str, str]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(count)
    ]


def test_valid_history_is_accepted() -> None:
    request = ConversationRequest(messages=_alternating(5))
    assert len(request.messages) == 5


def test_content_is_trimmed() -> None:
    request = ConversationRequest(messages=[{"role": "user", "content": "  hi  "}])
    assert request.messages[0].content == "hi"


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "system", "content": "x"}], id="role-system"),
        pytest.param(
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            id="not-alternating",
        ),
        pytest.param(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            id="ends-with-assistant",
        ),
        pytest.param([], id="empty-list"),
        pytest.param(_alternating(MAX_MESSAGES + 2)[:-1], id="too-many-messages"),
        pytest.param(
            [{"role": "user", "content": "x" * (MAX_CHARS + 1)}], id="content-too-long"
        ),
        pytest.param([{"role": "user", "content": "   "}], id="content-blank"),
    ],
)
def test_invalid_histories_are_rejected(messages: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        ConversationRequest(messages=messages)


def test_too_many_messages_case_really_has_one_over_the_limit() -> None:
    # Guards the parametrised case above: MAX+1 messages, ending with 'user'.
    assert len(_alternating(MAX_MESSAGES + 2)[:-1]) == MAX_MESSAGES + 1


def test_error_message_names_the_wrong_position() -> None:
    with pytest.raises(ValidationError, match=r"messages\[1\]"):
        ConversationRequest(
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )


@pytest.mark.parametrize("model", [AskRequest, ConversationRequest])
def test_minimal_thinking_level_is_rejected(model: type) -> None:
    payload: dict[str, object] = {"thinking_level": "minimal"}
    if model is AskRequest:
        payload["message"] = "hi"
    else:
        payload["messages"] = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValidationError):
        model(**payload)


def test_ask_rejects_blank_and_overlong_messages() -> None:
    for bad in ["", "   ", "x" * (MAX_CHARS + 1)]:
        with pytest.raises(ValidationError):
            AskRequest(message=bad)
