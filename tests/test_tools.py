from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.knowledge import SearchResultItem
from app.tools import PROJECT_TOOLS, get_tool_catalog
from app.tools.knowledge import KnowledgeSearchInput, search_knowledge_base


def test_tool_catalog_counts():
    # 5 app-run tools
    assert len(PROJECT_TOOLS) == 5
    app_tool_names = {t.name for t in PROJECT_TOOLS}
    assert app_tool_names == {
        "search_knowledge_base",
        "calculate",
        "get_current_utc_time",
        "search_world_events",
        "web_search",
    }

    # 7 catalog entries (5 app-run + 2 built-in)
    catalog = get_tool_catalog()
    assert len(catalog) == 7
    skb = next((t for t in catalog if t["name"] == "search_knowledge_base"), None)
    assert skb is not None
    assert skb["kind"] == "custom"
    assert skb["runs_on"] == "app"


def test_knowledge_search_input_schema():
    # Valid
    inp = KnowledgeSearchInput(query="how many leave days?", top_k=4)
    assert inp.query == "how many leave days?"
    assert inp.top_k == 4

    # Boundary: top_k > 6
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="test", top_k=7)

    # Boundary: query > 300
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="a" * 301, top_k=4)

    # Boundary: blank query
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="", top_k=4)


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_output():
    mock_retriever = AsyncMock()

    # 1. When matching chunks pass threshold
    mock_chunk = SearchResultItem(
        rank=1,
        document_id="doc1",
        filename="employee-handbook.md",
        chunk_index=2,
        score=0.95,
        passed_threshold=True,
        text="Full-time employees get 27 days leave.",
    )
    mock_retriever.retrieve.return_value = [mock_chunk]

    with patch(
        "app.tools.knowledge.get_retrieval_service", return_value=mock_retriever
    ):
        result = await search_knowledge_base.ainvoke(
            {"query": "leave days", "top_k": 4}
        )
        assert (
            "[1] employee-handbook.md (chunk 2): Full-time employees get 27 days leave."
            in result
        )

    # 2. When no chunks pass threshold
    mock_chunk_failed = SearchResultItem(
        rank=1,
        document_id="doc1",
        filename="employee-handbook.md",
        chunk_index=0,
        score=0.2,
        passed_threshold=False,
        text="Unrelated text.",
    )
    mock_retriever.retrieve.return_value = [mock_chunk_failed]

    with patch(
        "app.tools.knowledge.get_retrieval_service", return_value=mock_retriever
    ):
        result = await search_knowledge_base.ainvoke({"query": "unrelated", "top_k": 4})
        assert result == "No relevant passages found in the knowledge base."
