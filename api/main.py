"""
InsightEngine AI — FastAPI backend.

Replaces the old Chainlit UI. No auth — single default user (see the
migration plan for why). Serves a JSON/SSE API for the React frontend.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make 'core' importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Enable LangSmith tracing if configured
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "insightengine-ai")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import singletons
from core.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    singletons.init_sync()
    await singletons.init_async()
    logger.info("InsightEngine AI backend ready.")
    yield


app = FastAPI(title="InsightEngine AI", lifespan=lifespan)

# Single-user local/demo deployment (no auth, no cookies) — permissive CORS is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


from api.routes import chat, documents, conversations, memories  # noqa: E402

app.include_router(chat.router,          prefix="/api")
app.include_router(documents.router,     prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(memories.router,      prefix="/api")

# Built React frontend (frontend/dist, produced by `npm run build`). Mounted
# last and at "/" so it only catches requests that didn't match an /api/*
# route above. Absent during local `uvicorn` dev (Vite's own dev server
# handles the frontend then) — only present inside the Docker image.
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
