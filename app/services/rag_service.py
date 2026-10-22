"""Service for context building and grounded RAG generation."""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.errors import AppError, from_llm_exception
from app.core.llm import get_chat_model
from app.prompts.rag import CONDENSE_SYSTEM_PROMPT, NO_ANSWER_TEXT, RAG_SYSTEM_PROMPT
from app.schemas.chat import ChatMessage, Usage
from app.schemas.rag import RagReply, RetrievalInfo, SourceChunk
from app.services.retrieval_service import get_retrieval_service

logger = logging.getLogger(__name__)


def _extract_response_text(response: AIMessage) -> str:
    """Return response.text per project standard, falling back only if mock lacks .text."""
    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        val = text_attr()
        if isinstance(val, str):
            return val
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else ""


def _build_context(chunks: list) -> tuple[str, list[SourceChunk]]:
    settings = get_settings()
    sources = []
    context_blocks = []
    current_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        clean_text = chunk.text.replace("<context>", "[context]").replace(
            "</context>", "[/context]"
        )
        block = f"\n--- Source [{i}] | File: {chunk.filename} (Chunk {chunk.chunk_index}) ---\n{clean_text}\n"

        if current_chars + len(block) > settings.rag_max_context_chars:
            break

        current_chars += len(block)
        context_blocks.append(block)
        sources.append(
            SourceChunk(
                source_id=i,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                snippet=clean_text[:300],
                cited=False,
                page=chunk.page,
            )
        )
    return "".join(context_blocks), sources


class RagService:
    def __init__(self) -> None:
        self.retrieval_service = get_retrieval_service()
        self.settings = get_settings()

    async def answer_question(
        self,
        question: str,
        top_k: int | None,
        document_ids: list[str] | None,
        thinking_level: str,
    ) -> RagReply:
        question = question.strip()

        raw_chunks = await self.retrieval_service.retrieve(
            query=question, top_k=top_k, document_ids=document_ids, min_score=None
        )
        passed_chunks = [c for c in raw_chunks if c.passed_threshold]

        retrieval_info = RetrievalInfo(
            top_k=top_k or self.settings.rag_top_k_default,
            min_score=self.settings.rag_min_score,
            retrieved=len(raw_chunks),
            used=0,
        )

        # Architectural Rule (Task 3.1): Debug retrieval first.
        if not passed_chunks:
            return RagReply(
                reply=NO_ANSWER_TEXT,
                model="none",
                thinking_level=thinking_level,
                finish_reason="NO_CONTEXT",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                answered_from_context=False,
                answer_found=False,
                sources=[],
                retrieval=retrieval_info,
            )

        context_str, sources = _build_context(passed_chunks)
        retrieval_info.used = len(sources)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=f"<context>\n{context_str}\n</context>\n\nQuestion: {question}"
            ),
        ]

        llm = get_chat_model(thinking_level=thinking_level)
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            raise from_llm_exception(exc) from exc

        reply_text = _extract_response_text(response)
        if not reply_text or not reply_text.strip():
            raise AppError.from_code("LLM_EMPTY_RESPONSE")

        # Regex matches both standard [1] and unicode 【1】 brackets
        cited_numbers = {int(m) for m in re.findall(r"[\[【](\d+)[\]】]", reply_text)}
        valid_source_ids = {src.source_id for src in sources}
        invalid_citations = cited_numbers - valid_source_ids
        if invalid_citations:
            logger.warning(
                "Ignored non-existent citation numbers in reply: invalid_count=%d",
                len(invalid_citations),
            )

        for src in sources:
            if src.source_id in cited_numbers:
                src.cited = True

        usage = Usage(
            input_tokens=(
                response.usage_metadata.get("input_tokens", 0)
                if response.usage_metadata
                else 0
            ),
            output_tokens=(
                response.usage_metadata.get("output_tokens", 0)
                if response.usage_metadata
                else 0
            ),
            total_tokens=(
                response.usage_metadata.get("total_tokens", 0)
                if response.usage_metadata
                else 0
            ),
        )

        return RagReply(
            reply=reply_text.strip(),
            model=response.response_metadata.get("model_name", "unknown"),
            thinking_level=thinking_level,
            finish_reason="STOP",
            usage=usage,
            answered_from_context=True,
            answer_found=reply_text.strip() != NO_ANSWER_TEXT,
            sources=sources,
            retrieval=retrieval_info,
        )

    async def answer_conversation(
        self,
        history: list[ChatMessage],
        top_k: int | None,
        document_ids: list[str] | None,
        thinking_level: str,
    ) -> RagReply:
        # Fallback default: Just use the user's raw question
        standalone = history[-1].content if history else ""
        usage_add = None

        if len(history) > 1:
            try:
                condense_msgs = [SystemMessage(content=CONDENSE_SYSTEM_PROMPT)]
                for msg in history:
                    if msg.role == "user":
                        condense_msgs.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        condense_msgs.append(AIMessage(content=msg.content))

                llm = get_chat_model(thinking_level="low")
                condense_resp = await llm.ainvoke(condense_msgs)
                extracted = _extract_response_text(condense_resp).strip()
                
                if extracted and len(extracted) > 2:
                    standalone = extracted
                    
                if getattr(condense_resp, "usage_metadata", None):
                    usage_add = condense_resp.usage_metadata
            except Exception as exc:
                # GRACEFUL DEGRADATION: If Groq crashes parsing Llama-3's output, 
                # we just ignore the rewrite and use the original question!
                logger.warning(f"Groq condensation failed, using fallback question: {exc}")

        max_chars = self.settings.rag_max_question_chars
        if len(standalone) > max_chars:
            standalone = standalone[:max_chars]

        # 2. Search using the standalone question
        reply = await self.answer_question(
            standalone, top_k, document_ids, thinking_level
        )
        reply.standalone_question = standalone

        # 3. Add the rewrite tokens to the total usage if it succeeded
        if usage_add:
            reply.usage.input_tokens += usage_add.get("input_tokens", 0)
            reply.usage.output_tokens += usage_add.get("output_tokens", 0)
            reply.usage.total_tokens += usage_add.get("total_tokens", 0)

        return reply


def get_rag_service() -> RagService:
    return RagService()