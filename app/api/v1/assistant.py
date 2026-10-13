"""Week 7 Assistant Endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.chat import ConversationRequest, Usage
from app.services.tool_runner import run_tool_loop
from app.tools import get_tool_catalog

router = APIRouter()


class AssistantReply(BaseModel):
    reply: str
    model: str
    thinking_level: str
    usage: Usage
    rounds: int
    trace: list[dict[str, Any]]


class AssistantMessagesRequest(ConversationRequest):
    tools: list[str] | None = None


@router.get("/tools", summary="List available tools", tags=["Assistant"])
async def get_tools() -> dict:
    return {"tools": get_tool_catalog()}


@router.post(
    "/messages",
    response_model=AssistantReply,
    summary="Run tool execution loop",
    tags=["Tool Chat"],
)
async def assistant_messages(payload: AssistantMessagesRequest) -> AssistantReply:
    # Task 4.1 Requirement: Validate requested tools
    catalog_names = [t["name"] for t in get_tool_catalog()]
    if payload.tools:
        for t in payload.tools:
            if t not in catalog_names:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "TOOL_UNKNOWN", "message": f"Unknown tool: {t}"},
                )

    result = await run_tool_loop(
        history=payload.messages,
        thinking_level=payload.thinking_level or "low",
        allowed_tools=payload.tools,
    )
    return AssistantReply(**result)
