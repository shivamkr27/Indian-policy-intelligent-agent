"""
Chat streaming endpoint — Server-Sent Events (SSE).

Ports the old Chainlit app's _run_graph(). Key differences from the original:

1. No client-trusted "awaiting_clarification" session flag. Whether a thread is
   paused for clarification is derived fresh from the checkpointer
   (graph.aget_state(config).next) on every request — this is what the old
   Chainlit app's resume_conversation bug got wrong (a stale flag from a
   different thread could leak in and cause a silent no-op resume).

"""

import asyncio
import json
import re
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

from api import singletons
from api.deps import get_user_id
from core.tools import user_id_ctx
from core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

_SOURCE_PATTERN = re.compile(r"^Source:\s*(.+?)$", re.MULTILINE)

# Nodes whose LLM call produces the user-visible final answer text.
_ANSWER_NODES = {"aggregate_answers", "fallback_response", "reasoning_synthesizer"}


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    answer_language: str = "english"
    web_search_enabled: bool = False


def _extract_last_ai_message(state_values: dict) -> str:
    for msg in reversed(state_values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_chat(req: ChatRequest, user_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": req.thread_id}}

    singletons.history.upsert(req.thread_id, user_id)
    singletons.history.set_title(req.thread_id, req.message)
    singletons.history.touch(req.thread_id)

    user_memories = await asyncio.to_thread(
        singletons.memory_store.fetch_relevant, user_id, req.message
    )

    # Derive resume-vs-fresh from the checkpointer itself (see module docstring).
    state = await singletons.graph.aget_state(config)
    awaiting_clarification = bool(state.next) and "request_clarification" in state.next

    if awaiting_clarification:
        await singletons.graph.aupdate_state(
            config, {"messages": [HumanMessage(content=req.message)]}
        )
        graph_input = None
    else:
        graph_input = {
            "messages": [HumanMessage(content=req.message)],
            "answer_language": req.answer_language,
            "user_memories": user_memories,
            "web_search_enabled": req.web_search_enabled,
        }

    token = user_id_ctx.set(user_id)
    sources: set[str] = set()
    reasoning_steps: list[str] = []
    full_answer = ""

    try:
        async for event in singletons.graph.astream_events(graph_input, config=config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if kind == "on_chat_model_stream" and node in _ANSWER_NODES:
                chunk = event["data"]["chunk"].content
                if chunk:
                    full_answer += chunk
                    yield _sse({"type": "token", "content": chunk})

            elif kind == "on_tool_start" and event.get("name") in ("search_chunks", "web_search"):
                tool_input = event["data"].get("input") or {}
                query_str = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                if query_str:
                    yield _sse({"type": "tool_start", "tool": event["name"], "query": query_str})

            elif kind == "on_tool_end" and event.get("name") == "search_chunks":
                raw = event["data"].get("output", "")
                output = raw.content if hasattr(raw, "content") else str(raw) if raw else ""
                for hit in _SOURCE_PATTERN.finditer(output):
                    src = hit.group(1).strip()
                    if src:
                        sources.add(src)
                if output and not output.startswith("NO_"):
                    preview = output[:400]
                    yield _sse({"type": "tool_end", "tool": "search_chunks", "preview": preview})

            elif kind == "on_chain_end" and node == "reasoning_planner":
                out = event["data"].get("output") or {}
                reasoning_steps = out.get("reasoning_steps", [])
                if reasoning_steps:
                    yield _sse({"type": "reasoning_plan", "steps": reasoning_steps})

            elif kind == "on_chain_end" and node == "execute_reasoning_step":
                out = event["data"].get("output") or {}
                step_num = out.get("current_step_index", 1)
                results = out.get("step_results", [])
                preview = results[-1][:300] if results else ""
                yield _sse({
                    "type": "reasoning_step",
                    "step": step_num,
                    "total": len(reasoning_steps),
                    "preview": preview,
                })

        final_state = await singletons.graph.aget_state(config)

        if final_state.next and "request_clarification" in final_state.next:
            clarification = _extract_last_ai_message(final_state.values)
            yield _sse({
                "type": "clarification",
                "question": clarification or "Could you clarify your question?",
            })
            return

        if not full_answer:
            full_answer = _extract_last_ai_message(final_state.values) or (
                "The agent could not generate a response. Please try rephrasing."
            )

        yield _sse({
            "type": "final",
            "content": full_answer,
            "sources": sorted(sources),
            "judge_badge": final_state.values.get("judge_badge", ""),
            "judge_reason": final_state.values.get("judge_reason", ""),
        })

        msgs_for_memory = list(final_state.values.get("messages", []))
        if msgs_for_memory and not final_state.next:
            asyncio.create_task(
                asyncio.to_thread(
                    singletons.memory_store.extract_and_save, user_id, msgs_for_memory, singletons.llm
                )
            )

    except Exception as e:
        logger.error(f"chat stream error: {type(e).__name__}: {e}", exc_info=True)
        err_map = {
            "RateLimitError": "The AI service is busy right now (rate limit reached). Please try again in a few minutes.",
            "APITimeoutError": "The AI service took too long to respond. Please try again.",
            "TimeoutError": "Request timed out. Please try a shorter question.",
            "APIConnectionError": "Couldn't reach the AI service. Please check your connection and try again.",
            "ConnectionError": "Couldn't reach the AI service. Please check your connection and try again.",
            "InternalServerError": "The AI service is having issues right now. Please try again in a moment.",
        }
        friendly = err_map.get(type(e).__name__, "Something went wrong. Please try again.")
        yield _sse({"type": "error", "message": friendly})
    finally:
        user_id_ctx.reset(token)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, user_id: str = Depends(get_user_id)):
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty.")
    if not singletons.limiter.is_allowed(req.thread_id):
        raise HTTPException(429, "Too many requests. Please slow down.")

    return StreamingResponse(_stream_chat(req, user_id), media_type="text/event-stream")
