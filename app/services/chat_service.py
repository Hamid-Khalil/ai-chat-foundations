"""Chat business logic: build messages, call LLM safely with tools, shape the response."""

import inspect
import logging
import time
from collections.abc import Sequence

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage, ToolMessage

from app.core.config import Settings, get_settings
from app.core.errors import AppError, from_llm_exception
from app.core.llm import get_chat_model
from app.prompts.chatbot import SYSTEM_PROMPT
from app.schemas.chat import (
    ALLOWED_THINKING_LEVELS,
    ChatMessage,
    ChatReply,
    ModelInfo,
    ModelLimits,
    Usage,
)
from app.services.tools import PROJECT_TOOLS

logger = logging.getLogger(__name__)

# Map tools by name for fast runtime execution
TOOLS_BY_NAME = {tool.name: tool for tool in PROJECT_TOOLS}


def to_langchain_messages(history: Sequence[ChatMessage]) -> list[BaseMessage]:
    """Pure mapping: server SystemMessage first, then the client's turns in order."""
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    for item in history:
        if item.role == "user":
            messages.append(HumanMessage(item.content))
        else:
            messages.append(AIMessage(item.content))
    return messages


def _extract_text(content: object) -> str:
    """Safely extract plain text from string or structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content) if content is not None else ""


def _to_chat_reply(
    response: AIMessage, model_name: str, thinking_level: str
) -> ChatReply:
    """Turn a model response into a ChatReply (shared by every chat endpoint)."""
    text = _extract_text(response.content)
    if not text.strip():
        raise AppError.from_code("LLM_EMPTY_RESPONSE")

    usage = response.usage_metadata or {}
    output_details = usage.get("output_token_details") or {}
    reasoning = output_details.get("reasoning")
    finish_reason = (response.response_metadata or {}).get("finish_reason")

    return ChatReply(
        reply=text,
        model=model_name,
        thinking_level=thinking_level,  # type: ignore[arg-type]
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        usage=Usage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reasoning_tokens=int(reasoning) if reasoning is not None else None,
            total_tokens=int(usage.get("total_tokens", 0)),
        ),
    )


async def _call_model(
    endpoint: str, messages: list[BaseMessage], thinking_level: str | None
) -> ChatReply:
    """Call the model with bound tools, with a failsafe for empty LLM summaries."""
    settings = get_settings()
    level = thinking_level or settings.gemini_thinking_level
    base_model = get_chat_model(level)

    # Bind tools to the model
    model = base_model.bind_tools(PROJECT_TOOLS) if PROJECT_TOOLS else base_model

    started = time.perf_counter()
    working_messages = list(messages)
    last_tool_output = ""

    try:
        # Loop up to 3 times to allow the model to call tools and then generate final text
        max_tool_iterations = 3
        response = None

        for _ in range(max_tool_iterations):
            response = await model.ainvoke(working_messages)

            # Check if the model requested any tool executions
            if not getattr(response, "tool_calls", None):
                # No more tools needed; this is the final conversational reply
                break

            working_messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")

                selected_tool = TOOLS_BY_NAME.get(tool_name)
                if selected_tool:
                    # Support both sync and async tool execution
                    if inspect.iscoroutinefunction(selected_tool.func):
                        tool_output = await selected_tool.ainvoke(tool_args)
                    else:
                        tool_output = selected_tool.invoke(tool_args)
                else:
                    tool_output = f"Tool '{tool_name}' not found."

                # Save the output in case the LLM fails to summarize it
                last_tool_output = str(tool_output)

                working_messages.append(
                    ToolMessage(
                        content=last_tool_output,
                        tool_call_id=tool_id,
                    )
                )

        if response is None:
            raise AppError.from_code("LLM_EMPTY_RESPONSE")

        # FAILSAFE: If the LLM returned an empty string but we have tool data, force the data into the response.
        extracted = _extract_text(response.content)
        if not extracted.strip() and last_tool_output:
            response.content = (
                f"Here is the real-time data I found:\n\n{last_tool_output}"
            )

    except Exception as exc:  # noqa: BLE001
        raise from_llm_exception(exc) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    reply = _to_chat_reply(response, settings.gemini_model, level)

    logger.info(
        "chat call endpoint=%s model=%s thinking_level=%s latency_ms=%d total_tokens=%d",
        endpoint,
        settings.gemini_model,
        level,
        latency_ms,
        reply.usage.total_tokens,
    )
    return reply


async def ask(message: str, thinking_level: str | None = None) -> ChatReply:
    """Single question -> single answer, with the server's system prompt."""
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]
    return await _call_model("ask", messages, thinking_level)


async def converse(
    history: Sequence[ChatMessage], thinking_level: str | None = None
) -> ChatReply:
    """Answer the last user turn, given the full history re-sent by the client."""
    return await _call_model("messages", to_langchain_messages(history), thinking_level)


def get_model_info(settings: Settings) -> ModelInfo:
    """Describe the running configuration. The API key is never included."""
    return ModelInfo(
        provider="groq",
        model=settings.gemini_model,
        thinking_level=settings.gemini_thinking_level,
        allowed_thinking_levels=ALLOWED_THINKING_LEVELS,
        max_tokens=settings.gemini_max_tokens,
        timeout_seconds=settings.gemini_timeout_seconds,
        max_retries=settings.gemini_max_retries,
        limits=ModelLimits(
            max_messages=settings.chat_max_messages,
            max_message_chars=settings.chat_max_message_chars,
        ),
    )
