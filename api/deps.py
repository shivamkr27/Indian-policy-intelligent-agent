"""
Per-request user identity — no login screen.

The client generates a random ID once (persisted in localStorage) and
sends it as the X-User-Id header on every request. That's enough to
activate the per-user isolation already built into core/ (ChromaDB
metadata filtering, conversation history, memories) — see
core/config.py's USER_ISOLATION flag, which only auto-disables for the
literal string "default". Falls back to "default" (shared, no isolation)
for requests that don't send the header — e.g. curl/manual testing.
"""

from typing import Optional

from fastapi import Header

from api import singletons


async def get_user_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    return x_user_id or singletons.DEFAULT_USER_ID
