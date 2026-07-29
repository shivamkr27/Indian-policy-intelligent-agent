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
_provider_error_types = [TimeoutError, ConnectionError]
try:
    import groq
    # groq.APIError is the base class for the SDK's entire exception hierarchy
    # (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
    # and the "Failed to call a function" APIError Groq raises when tool-calling
    # itself glitches — observed in practice near the daily rate-limit boundary).
    # Catching the base class covers all of these without needing to enumerate
    # every subclass individually.
    _provider_error_types.append(groq.APIError)
except ImportError:
    pass
try:
    import openai
    # Same reasoning as groq.APIError above — the fallback LLM (HuggingFace via
    # its OpenAI-compatible router) raises this hierarchy on its own failures.
    _provider_error_types.append(openai.APIError)
except ImportError:
    pass
PROVIDER_ERRORS = tuple(_provider_error_types)


def is_provider_error(exc: Exception) -> bool:
    """True if `exc` means an LLM provider is unavailable (rate limit, outage, network)."""
    return isinstance(exc, PROVIDER_ERRORS)


def invoke_resilient(primary, messages, fallback=None):
    """
    Invoke `primary` (already tool-bound / structured-output-bound if needed);
    on a provider error, retry once against `fallback` (built the same way from
    the secondary LLM) if one was given. Raises if both fail, so the caller's
    own error handling (and ultimately api/routes/chat.py's raw-retrieval
    fallback) still applies when every provider is down.
    """
    try:
        return invoke_with_retry(primary, messages)
    except Exception as e:
        if fallback is not None and is_provider_error(e):
            logger.warning(f"Primary LLM failed ({type(e).__name__}) — retrying via fallback LLM")
            return invoke_with_retry(fallback, messages)
        raise
