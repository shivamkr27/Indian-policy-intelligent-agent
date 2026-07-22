"""
Conversation history endpoints — list, fetch last exchange, delete.

List/delete use ConversationStore (lightweight metadata, core/history.py).

IMPORTANT: the graph (core/graph.py::rewrite_query) intentionally wipes the
checkpoint's `messages` list on every clear turn — it carries context forward
via a compacted `conversation_summary` instead, to keep token usage bounded.
One consequence: `messages` in the checkpoint only ever reflects the *latest*
turn, and even then it's polluted with the RAG agent subgraph's internal
tool-calling exchange (AgentState shares the `messages` channel with the
parent State, so a subgraph invocation's internal HumanMessage/ToolMessage/
AIMessage chatter merges back into the main checkpoint). So this endpoint
does NOT return a full multi-turn transcript — that data doesn't exist. It
returns the last real question (`original_query`, set once per turn) and the
last real answer (the final non-tool-call AIMessage), which is the same
"last Q + last A" preview the old Chainlit app showed on conversation resume.
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage

from api import singletons

router = APIRouter(tags=["conversations"])


@router.get("/conversations")
async def list_conversations():
    user_id = singletons.DEFAULT_USER_ID
    return {"conversations": singletons.history.list_user(user_id)}


@router.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = await singletons.graph.aget_state(config)
    values = state.values

    final_answer = ""
    for m in reversed(values.get("messages", [])):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            final_answer = m.content
            break

    messages = []
    if values.get("original_query"):
        messages.append({"role": "user", "content": values["original_query"]})
    if final_answer:
        messages.append({"role": "assistant", "content": final_answer})

    return {"thread_id": thread_id, "messages": messages}


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    user_id = singletons.DEFAULT_USER_ID
    deleted = singletons.history.delete(thread_id, user_id)
    if not deleted:
        raise HTTPException(404, "Conversation not found.")
    return {"deleted": True}
