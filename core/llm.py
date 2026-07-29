"""
LLM factory — Groq primary, optional HuggingFace fallback.
"""

import os

from .logging_config import get_logger

logger = get_logger(__name__)


def get_llm():
    """Return the primary LLM (Groq)."""
    from .config import GROQ_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
    from langchain_groq import ChatGroq

    logger.info(f"Initializing LLM — Groq ({GROQ_MODEL})")
    return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)


def get_fallback_llm():
    """
    Return the secondary LLM (HuggingFace, via its OpenAI-compatible router),
    used on the main chat path when Groq is rate-limited/down.

    Returns None if HF_TOKEN isn't configured — the fallback is optional; the
    app already degrades to raw local document search if this is unavailable.
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.info("HF_TOKEN not set — no fallback LLM configured.")
        return None

    from .config import HF_MODEL, HF_ROUTER_BASE_URL, LLM_TEMPERATURE, LLM_TIMEOUT
    from langchain_openai import ChatOpenAI

    logger.info(f"Initializing fallback LLM — HuggingFace ({HF_MODEL})")
    return ChatOpenAI(
        model=HF_MODEL,
        api_key=hf_token,
        base_url=HF_ROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )


def get_grader_llm():
    """Fast, cheap LLM for CRAG retrieval grading (llama-3.1-8b-instant via Groq)."""
    from .config import GRADER_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GRADER_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)
    except Exception as exc:
        logger.warning(f"Could not init grader LLM ({exc}), falling back to main LLM")
        return get_llm()
