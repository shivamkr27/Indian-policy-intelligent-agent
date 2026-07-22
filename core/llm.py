"""
LLM factory — Groq only.
"""

from .logging_config import get_logger

logger = get_logger(__name__)


def get_llm():
    """Return the primary LLM (Groq)."""
    from .config import GROQ_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
    from langchain_groq import ChatGroq

    logger.info(f"Initializing LLM — Groq ({GROQ_MODEL})")
    return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)


def get_grader_llm():
    """Fast, cheap LLM for CRAG retrieval grading (llama-3.1-8b-instant via Groq)."""
    from .config import GRADER_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GRADER_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)
    except Exception as exc:
        logger.warning(f"Could not init grader LLM ({exc}), falling back to main LLM")
        return get_llm()
