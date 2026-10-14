"""Tool Registry and Safe Custom Tools."""

import ast
import logging
import operator
import os
from datetime import datetime, timezone

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.core.config import get_settings
from app.tools.knowledge import search_knowledge_base

logger = logging.getLogger(__name__)

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_math_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        return _SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        operand = _eval_math_node(node.operand)
        return _SAFE_OPERATORS[type(node.op)](operand)
    raise ValueError("Unsupported mathematical expression.")


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    return TavilyClient(api_key=key)


class GlobalEventInput(BaseModel):
    query: str = Field(
        description="Search keywords for real-time information, sports updates, scores, world news, or current events."
    )


@tool(args_schema=GlobalEventInput)
def search_world_events(query: str) -> str:
    """CRITICAL: Invoke this tool whenever the user asks for real-world news, sports scores, schedules, or current events."""
    try:
        client = _get_tavily_client()
        response = client.search(query=query, max_results=4, search_depth="basic")
        results = response.get("results", [])

        if not results:
            return f"No live search results found for: '{query}'."

        snippets = []
        for item in results:
            title = item.get("title", "No Title")
            content = item.get("content", "")
            snippets.append(f"• {title}: {content}")

        return "\n\n".join(snippets)

    except Exception as exc:
        logger.error("Tavily search execution failed: %s", exc)
        return f"Live search failed: {exc}"


@tool
def get_current_utc_time() -> str:
    """Returns the current real-world UTC date and time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression like '15 * 2400 / 100'."""
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_math_node(parsed)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Searches the web (dummy tool)."""
    return f"Web search results for {query}"


PROJECT_TOOLS = [
    search_knowledge_base,
    calculate,
    get_current_utc_time,
    search_world_events,
    web_search,
]


def get_tool_catalog():
    return [
        {"name": "search_knowledge_base", "kind": "custom", "runs_on": "app"},
        {"name": "calculate", "kind": "custom", "runs_on": "app"},
        {"name": "get_current_utc_time", "kind": "custom", "runs_on": "app"},
        {"name": "search_world_events", "kind": "custom", "runs_on": "app"},
        {"name": "web_search", "kind": "custom", "runs_on": "app"},
        {"name": "system_time", "kind": "builtin", "runs_on": "model"},
        {"name": "system_calc", "kind": "builtin", "runs_on": "model"},
    ]
