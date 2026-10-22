"""Standard error shape, the AppError exception and Gemini-failure mapping."""

import logging
from collections.abc import Iterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_ERRORS: dict[str, tuple[int, str]] = {
    "LLM_RATE_LIMITED": (429, "The AI service is busy right now. Please try again in a moment."),
    "LLM_UPSTREAM_ERROR": (502, "The AI service had a problem answering. Please try again later."),
    "LLM_EMPTY_RESPONSE": (502, "The AI service returned an empty answer. Please try rephrasing your message."),
    "LLM_TIMEOUT": (504, "The AI service took too long to answer. Please try again."),
    "INTERNAL_ERROR": (500, "Something went wrong on our side. Please try again."),
    "DB_UNAVAILABLE": (503, "The database is currently unavailable. Please try again later."),
    "DB_ERROR": (500, "An unexpected database error occurred."),
    "CONVERSATION_NOT_FOUND": (404, "The requested conversation could not be found."),
    "MESSAGE_NOT_FOUND": (404, "The requested message could not be found."),
    "CONVERSATION_ARCHIVED": (409, "Cannot modify an archived conversation."),
    "CONVERSATION_FULL": (409, "This conversation has reached the maximum allowed messages."),
    "INVALID_CURSOR": (400, "The provided pagination cursor is invalid or malformed."),
    "FEEDBACK_INVALID_TARGET": (400, "Feedback can only be submitted for assistant messages."),
}


class AppError(Exception):
    """An error that maps directly onto the standard error response."""
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.status = status

    @classmethod
    def from_code(cls, code: str) -> "AppError":
        status, message = _ERRORS[code]
        return cls(code=code, message=message, status=status)


def error_payload(code: str, message: str, status: int) -> dict[str, dict[str, object]]:
    return {"error": {"code": code, "message": message, "status": status}}


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def from_llm_exception(exc: Exception) -> AppError:
    print("\n========== EXACT GROQ ERROR ==========")
    try:
        print(getattr(exc, 'response', exc).text)
    except Exception:
        print(getattr(exc, 'response', exc))
    print("======================================\n")
    return AppError.from_code("LLM_UPSTREAM_ERROR")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content=error_payload(exc.code, exc.message, exc.status),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled %s on %s", type(exc).__name__, request.url.path)
    error = AppError.from_code("INTERNAL_ERROR")
    return JSONResponse(
        status_code=error.status,
        content=error_payload(error.code, error.message, error.status),
    )


# --- Week 8 Document Error Codes ---
class DocUnsupportedTypeError(AppError):
    def __init__(self, message: str = "Invalid file format or non-UTF-8 bytes."):
        super().__init__(code="DOC_UNSUPPORTED_TYPE", status=415, message=message)

class DocTooLargeError(AppError):
    def __init__(self, message: str = "File exceeds the maximum upload size."):
        super().__init__(code="DOC_TOO_LARGE", status=413, message=message)

class DocEmptyError(AppError):
    def __init__(self, message: str = "Document contains no usable text."):
        super().__init__(code="DOC_EMPTY", status=400, message=message)

class DocDuplicateError(AppError):
    def __init__(self, doc_id: str):
        super().__init__(
            code="DOC_DUPLICATE", status=409,
            message=f"A document with identical content already exists (id: {doc_id})."
        )

class DocLimitReachedError(AppError):
    def __init__(self, limit: int):
        super().__init__(
            code="DOC_LIMIT_REACHED", status=409,
            message=f"Knowledge base document limit ({limit}) reached."
        )

class DocNotFoundError(AppError):
    def __init__(self, doc_id: str):
        super().__init__(
            code="DOC_NOT_FOUND", status=404, message=f"Document '{doc_id}' not found."
        )


def from_db_exception(exc: Exception) -> AppError:
    from pymongo.errors import (
        ConnectionFailure, NetworkTimeout, OperationFailure,
        PyMongoError, ServerSelectionTimeoutError, DuplicateKeyError, BulkWriteError
    )
    if isinstance(exc, (DuplicateKeyError, BulkWriteError)):
        raise exc
    if isinstance(exc, (ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout)):
        return AppError.from_code("DB_UNAVAILABLE")
    if isinstance(exc, OperationFailure) and getattr(exc, "code", None) == 18:
        return AppError.from_code("DB_UNAVAILABLE")
    if isinstance(exc, PyMongoError):
        return AppError.from_code("DB_ERROR")
    return AppError.from_code("INTERNAL_ERROR")


def handle_db_errors(func):
    from functools import wraps
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            from pymongo.errors import PyMongoError
            if isinstance(exc, PyMongoError):
                raise from_db_exception(exc)
            raise exc
    return wrapper