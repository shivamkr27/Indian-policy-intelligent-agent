"""
Module-level singletons for the FastAPI backend.

Mirrors the old Chainlit app's startup sequence (ui/app.py, now removed):
build the heavy objects (embeddings, reranker, LLM, graph) exactly once,
shared across all requests. No per-request/per-session state lives here —
per-request identity (api/deps.py::get_user_id) falls back to
DEFAULT_USER_ID only when a request doesn't send an X-User-Id header.
"""

from core.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, HISTORY_DB_PATH
from core.llm import get_llm, get_fallback_llm
from core.ingestion import Ingestion
from core.tools import ToolFactory
from core.judge import HallucinationJudge
from core.memory_store import UserMemoryStore
from core.rate_limiter import RateLimiter
from core.history import ConversationStore
from core.graph import build_graph, create_checkpointer
from core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_USER_ID = "default"

llm = None
fallback_llm = None
ingestion: Ingestion = None
tool_factory: ToolFactory = None
judge: HallucinationJudge = None
memory_store: UserMemoryStore = None
limiter: RateLimiter = None
upload_limiter: RateLimiter = None
history: ConversationStore = None
graph = None
checkpointer = None


def init_sync() -> None:
    """Build everything that doesn't need a running event loop."""
    global llm, fallback_llm, ingestion, tool_factory, judge, memory_store, limiter, upload_limiter, history

    logger.info("Initializing InsightEngine AI backend...")
    llm            = get_llm()
    fallback_llm   = get_fallback_llm()
    ingestion      = Ingestion()
    judge          = HallucinationJudge()
    tool_factory   = ToolFactory(ingestion)
    memory_store   = UserMemoryStore(ingestion._embeddings)
    limiter        = RateLimiter(max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW)
    upload_limiter = RateLimiter(max_requests=5, window_seconds=60)
    history        = ConversationStore(db_path=HISTORY_DB_PATH)
    logger.info("Sync singletons ready.")


async def init_async() -> None:
    """Build the LangGraph pipeline with a real async checkpointer. Needs an event loop."""
    global graph, checkpointer

    checkpointer = await create_checkpointer()
    graph = build_graph(llm, tool_factory, judge, checkpointer, fallback_llm=fallback_llm)
    logger.info("Graph ready.")
