"""Standard error shape, the AppError exception and Gemini-failure mapping."""

import logging
from collections.abc import Iterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# code -> (HTTP status, friendly message). The raw upstream text is NEVER used here:
# it can leak internals, so it is only logged (type + status code) server-side.
_ERRORS: dict[str, tuple[int, str]] = {
    "LLM_RATE_LIMITED": (
        429,
        "The AI service is busy right now. Please try again in a moment.",
    ),
    "LLM_UPSTREAM_ERROR": (
        502,
        "The AI service had a problem answering. Please try again later.",
    ),
    "LLM_EMPTY_RESPONSE": (
        502,
        "The AI service returned an empty answer. Please try rephrasing your message.",
    ),
    "LLM_TIMEOUT": (
        504,
        "The AI service took too long to answer. Please try again.",
    ),
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
        """Build an AppError from one of the fixed codes."""
        status, message = _ERRORS[code]
        return cls(code=code, message=message, status=status)


def error_payload(code: str, message: str, status: int) -> dict[str, dict[str, object]]:
    """The standard error body: {"error": {"code", "message", "status"}}."""
    return {"error": {"code": code, "message": message, "status": status}}


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield `exc` and everything it was raised from (bounded, cycle-safe)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


# Exception classes observed/confirmed in the installed packages
# (langchain-google-genai 4.4.0 + google-genai; discovered by reading their source —
# re-confirm each against a live failure per Task 1.3 and update this note):
#   * rate limit / bad key / bad model: google.genai.errors.ClientError (has `.code`
#     = 429 / 400-403 / 404) is re-raised by LangChain as a wrapper such as
#     GoogleRateLimitError / GoogleModelNotFoundError. The WRAPPER has no `.code`;
#     the status lives on `__cause__`, so the whole chain is inspected below.
#   * 5xx: google.genai.errors.ServerError (has `.code`), wrapped as GoogleAPIError.
#   * timeout: httpx.TimeoutException (e.g. httpx.ReadTimeout) from the google-genai
#     HTTP client, retried `max_retries` times and then raised.
def from_llm_exception(exc: Exception) -> AppError:
    """Map any failure from the Gemini call to an AppError (friendly text only)."""
    chain = list(_cause_chain(exc))

    status_code: int | None = None
    for item in chain:
        code = getattr(item, "code", None)  # fake exceptions in tests expose `.code`
        if isinstance(code, int):
            status_code = code
            break

    is_timeout = any(
        isinstance(item, (TimeoutError, httpx.TimeoutException)) for item in chain
    )

    if is_timeout:
        mapped = "LLM_TIMEOUT"
    elif status_code == 429:
        mapped = "LLM_RATE_LIMITED"
    else:
        mapped = "LLM_UPSTREAM_ERROR"

    # Type + status only — never the prompt, the key or the raw upstream message.
    logger.error(
        "LLM call failed: exception_types=%s status_code=%s mapped=%s",
        [type(item).__name__ for item in chain],
        status_code,
        mapped,
    )
    return AppError.from_code(mapped)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Turn AppError into the standard error response."""
    return JSONResponse(
        status_code=exc.status,
        content=error_payload(exc.code, exc.message, exc.status),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log the type only, return a generic 500 in the standard shape."""
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
            code="DOC_DUPLICATE",
            status=409,
            message=f"A document with identical content already exists (id: {doc_id}).",
        )


class DocLimitReachedError(AppError):
    def __init__(self, limit: int):
        super().__init__(
            code="DOC_LIMIT_REACHED",
            status=409,
            message=f"Knowledge base document limit ({limit}) reached.",
        )


class DocNotFoundError(AppError):
    def __init__(self, doc_id: str):
        super().__init__(
            code="DOC_NOT_FOUND", status=404, message=f"Document '{doc_id}' not found."
        )


from pymongo.errors import (
    ServerSelectionTimeoutError,
    ConnectionFailure,
    NetworkTimeout,
    OperationFailure,
    PyMongoError,
)


# Why DuplicateKeyError is NOT handled here:
# Duplicate-key errors (e.g., duplicate document SHA256, duplicate message sequence)
# are business-logic violations, not system crashes. They must be caught in the
# repository or service layer where context is known, so they can be mapped to
# a 409 Conflict rather than a generic 500 error.
def from_db_exception(exc: Exception) -> AppError:
    """Map PyMongo driver exceptions to AppError (503/500) without leaking driver details."""
    from pymongo.errors import (
        ConnectionFailure,
        NetworkTimeout,
        OperationFailure,
        PyMongoError,
        ServerSelectionTimeoutError,
    )

    if isinstance(
        exc,
        (
            ServerSelectionTimeoutError,
            ConnectionFailure,
            NetworkTimeout,
            OperationFailure,
        ),
    ):
        logger.error("Database connection error: %s", type(exc).__name__)
        return AppError.from_code("DB_UNAVAILABLE")
    if isinstance(exc, PyMongoError):
        logger.error("Unexpected database error: %s", type(exc).__name__)
        return AppError.from_code("DB_ERROR")

    logger.error("Unhandled exception in DB layer: %s", type(exc).__name__)
    return AppError.from_code("INTERNAL_ERROR")
