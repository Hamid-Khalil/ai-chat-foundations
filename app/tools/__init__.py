"""Tool Registry."""

from langchain_core.tools import tool

from app.services.tools import get_current_utc_time, search_world_events
from app.tools.knowledge import search_knowledge_base


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression like '15 * 2400 / 100'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
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
