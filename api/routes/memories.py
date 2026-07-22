"""
User memory endpoints — view stored memories, submit answer-quality feedback.

Feedback used to read the question from a server-side "last_question" session
key (Chainlit). That per-session state is gone — the client sends the question
directly in the request body instead.
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api import singletons
from api.deps import get_user_id

router = APIRouter(tags=["memories"])

_FEEDBACK_COPY = {
    "good":   ("preference",     2, "User confirmed accurate answer on: {q}"),
    "review": ("knowledge_gap",  2, "Partially correct response on: {q} — be more thorough next time"),
    "wrong":  ("knowledge_gap",  3, "Wrong answer flagged for: {q} — verify carefully before answering similar queries"),
}


class FeedbackRequest(BaseModel):
    question: str
    rating: Literal["good", "review", "wrong"]


@router.get("/memories")
async def get_memories(user_id: str = Depends(get_user_id)):
    return {"memories": singletons.memory_store.get_all(user_id)}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, user_id: str = Depends(get_user_id)):
    memory_type, importance, template = _FEEDBACK_COPY[req.rating]
    content = template.format(q=req.question[:80])

    await asyncio.to_thread(
        singletons.memory_store.save_direct, user_id, content, memory_type, importance
    )
    return {"saved": True}
