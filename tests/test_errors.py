import httpx

from app.core.errors import AppError, from_llm_exception


class FakeUpstreamError(Exception):
    """Stands in for google.genai.errors.APIError, which exposes `.code`."""

    def __init__(self, code: int, text: str = "secret upstream detail") -> None:
        super().__init__(text)
        self.code = code


class FakeWrapperError(Exception):
    """Stands in for LangChain's wrapper classes: NO `.code`, status is on __cause__."""


def test_429_maps_to_rate_limited() -> None:
    error = from_llm_exception(FakeUpstreamError(429))
    assert (error.code, error.status) == ("LLM_RATE_LIMITED", 429)


def test_500_maps_to_upstream_error_502() -> None:
    error = from_llm_exception(FakeUpstreamError(500))
    assert (error.code, error.status) == ("LLM_UPSTREAM_ERROR", 502)


def test_timeout_maps_to_504() -> None:
    # httpx.TimeoutException is the timeout class used by the google-genai HTTP client.
    error = from_llm_exception(httpx.ReadTimeout("timed out"))
    assert (error.code, error.status) == ("LLM_TIMEOUT", 504)


def test_status_is_found_on_the_cause_chain() -> None:
    wrapper = FakeWrapperError("wrapped")
    wrapper.__cause__ = FakeUpstreamError(429)
    assert from_llm_exception(wrapper).code == "LLM_RATE_LIMITED"


def test_unknown_exception_maps_to_upstream_error() -> None:
    assert from_llm_exception(ValueError("boom")).code == "LLM_UPSTREAM_ERROR"


def test_message_never_contains_raw_upstream_text() -> None:
    error = from_llm_exception(FakeUpstreamError(500, "key=AIzaSECRET"))
    assert isinstance(error, AppError)
    assert "AIzaSECRET" not in error.message
    assert "secret" not in error.message.lower()
