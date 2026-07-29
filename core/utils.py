import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .logging_config import get_logger

logger = get_logger(__name__)

# httpx is always present (required by langchain-groq → groq SDK → httpx).
# Catching httpx network errors covers Groq transient failures correctly.
try:
    import httpx
    _RETRY_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        httpx.TimeoutException,
        httpx.ConnectError,
    )
except ImportError:
    _RETRY_EXCEPTIONS = (TimeoutError, ConnectionError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def invoke_with_retry(llm, messages):
    """Invoke an LLM with automatic retry on transient network / timeout errors."""
    return llm.invoke(messages)


# Exception types that mean "the LLM provider itself is unavailable right now"
# (rate limit, quota exhausted, provider outage, network/timeout) as opposed to
# a logic bug in our own code. These must NOT be swallowed into a generic
# fallback answer — doing so previously produced confusing, misleadingly-
# labeled responses (e.g. a "🟢 Verified" badge on an answer the judge never
# actually scored, because judge.score() itself hit the same rate limit).
# Instead they should propagate up to api/routes/chat.py's top-level handler,
# which already maps them to a clear "AI service is busy" message for the user.
try:
    import groq
    # groq.APIError is the base class for the SDK's entire exception hierarchy
    # (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
    # and the "Failed to call a function" APIError Groq raises when tool-calling
    # itself glitches — observed in practice near the daily rate-limit boundary).
    # Catching the base class covers all of these without needing to enumerate
    # every subclass individually.
    PROVIDER_ERRORS = (groq.APIError, TimeoutError, ConnectionError)
except ImportError:
    PROVIDER_ERRORS = (TimeoutError, ConnectionError)


def is_provider_error(exc: Exception) -> bool:
    """True if `exc` means the LLM provider is unavailable (rate limit, outage, network)."""
    return isinstance(exc, PROVIDER_ERRORS)
