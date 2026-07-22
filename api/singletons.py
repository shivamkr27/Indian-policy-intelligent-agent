"""
Module-level singletons for the FastAPI backend.

Mirrors the old Chainlit app's startup sequence (ui/app.py, now removed):
build the heavy objects (embeddings, reranker, LLM, graph) exactly once,
shared across all requests. No per-request/per-session state lives here —
auth was removed, so every request runs as DEFAULT_USER_ID.
"""

from core.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, HISTORY_DB_PATH
from core.llm import get_llm
from core.ingestion import Ingestion
from core.tools import ToolFactory
from core.text2sql import Text2SQLEngine
from core.judge import HallucinationJudge
from core.memory_store import UserMemoryStore
from core.rate_limiter import RateLimiter
from core.history import ConversationStore
from core.graph import build_graph, create_checkpointer
from core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_USER_ID = "default"

llm = None
ingestion: Ingestion = None
tool_factory: ToolFactory = None
sql_engine: Text2SQLEngine = None
judge: HallucinationJudge = None
memory_store: UserMemoryStore = None
limiter: RateLimiter = None
history: ConversationStore = None
graph = None
checkpointer = None


def init_sync() -> None:
    """Build everything that doesn't need a running event loop."""
    global llm, ingestion, tool_factory, sql_engine, judge, memory_store, limiter, history

    logger.info("Initializing InsightEngine AI backend...")
    llm          = get_llm()
    ingestion    = Ingestion()
    sql_engine   = Text2SQLEngine()
    judge        = HallucinationJudge()
    tool_factory = ToolFactory(ingestion)
    memory_store = UserMemoryStore(ingestion._embeddings)
    limiter      = RateLimiter(max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW)
    history      = ConversationStore(db_path=HISTORY_DB_PATH)
    logger.info("Sync singletons ready.")


async def init_async() -> None:
    """Build the LangGraph pipeline with a real async checkpointer. Needs an event loop."""
    global graph, checkpointer

    checkpointer = await create_checkpointer()
    graph = build_graph(llm, tool_factory, sql_engine, judge, checkpointer)
    logger.info("Graph ready.")
