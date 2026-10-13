"""LangChain tools for real-time web retrieval and deterministic tasks."""

import logging
from datetime import datetime, timezone

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    key = settings.tavily_api_key
    if not key:
        import os

        key = os.getenv("TAVILY_API_KEY", "")
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
        # Search using Tavily (returns clean, pre-parsed content optimized for LLMs)
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


PROJECT_TOOLS = [
    search_world_events,
    get_current_utc_time,
]
