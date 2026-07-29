# InsightEngine AI

A multi-agent RAG (Retrieval-Augmented Generation) system that answers questions from uploaded documents using hybrid search, parallel agent orchestration, and real-time streaming — served through a FastAPI backend and a React chat UI.

**Live:** http://80.225.212.121:8000

---

## Features

- **Document Q&A** — Upload PDFs, Word docs, or text files and ask anything
- **Hybrid Search** — Dense (ChromaDB) + Sparse (BM25) + cross-encoder reranking pipeline
- **Adaptive Retrieval** — retrieval profile (factual / conceptual / comparative) auto-selected per query
- **Multi-Agent Orchestration** — parallel LangGraph agents handle multiple sub-questions simultaneously
- **CRAG** — Corrective RAG loop with automatic query rewriting on irrelevant retrievals
- **Multi-Hop Reasoning** — breaks complex questions into ordered search steps and chains findings
- **Human-in-the-loop clarification** — the graph pauses and asks when a query is ambiguous
- **Web Search Toggle** — DuckDuckGo fallback when documents don't have the answer
- **Hallucination Judge** — LLM-as-Judge scores every answer 1–5 for factual grounding, with a bounded LRU response cache
- **Hindi Mode** — full Devanagari output with technical terms preserved
- **User Memory** — semantic memory extracted from conversations, personalizes future responses
- **Streaming** — token-by-token via Server-Sent Events
- **Rate Limiting** — 10 requests/60s per conversation thread
- **Per-browser isolation** — no login screen; each browser gets its own anonymous ID (persisted in `localStorage`), so documents/history/memories don't leak across users

---

## Architecture

```
User Message
    │
    ├─ summarize_history        Compact prior context; inject user memories
    ├─ rewrite_query            Clarify + split into sub-questions (structured output)
    │   └─[unclear]──► request_clarification   (HITL interrupt — waits for user)
    ├─ route_query              Classify: rag | multi_hop
    │
    ├─[RAG]──► Send("agent") × N   Parallel agents, one per sub-question
    │           └─ orchestrator → search_chunks → retrieval_grader
    │               ├─[irrelevant, attempts<2]──► query_rewriter_loop → orchestrator
    │               ├─[token limit]──► compress_context → orchestrator
    │               └─[done]──► collect_answer
    │           └─► aggregate_answers
    │
    ├─[multi_hop]──► reasoning_planner
    │                └─► execute_reasoning_step (self-loop)
    │                └─► reasoning_synthesizer
    │
    └─ hallucination_judge      Score answer 1–5; badge stored in state
```

**Agent subgraph** (runs N times in parallel for RAG):
```
START → orchestrator → tools (search_chunks / web_search)
             │              └─► retrieval_grader
             │                   ├─[irrelevant]──► query_rewriter_loop → orchestrator
             │                   └─[relevant]──► should_compress_context
             │                                        ├─► compress_context → orchestrator
             │                                        └─► orchestrator (continue)
             ├─[no tool call]──► collect_answer → END
             └─[max iterations]──► fallback_response → collect_answer → END
```

The FastAPI layer (`api/`) wraps this graph: `POST /api/chat/stream` derives whether a
thread is paused for clarification directly from the LangGraph checkpointer on every
request (not a client-trusted flag), and streams `on_chat_model_stream` /
`on_tool_start` / `on_tool_end` / reasoning-step events out as SSE.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Orchestration | LangGraph |
| Vector DB | ChromaDB (in-process) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Sparse Search | BM25Okapi (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Structured Data | SQLite + SQLAlchemy |
| Web Search | DuckDuckGo (`ddgs`) |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS v4 |
| Document Parsing | PyMuPDF / `pymupdf4llm` (PDF), `python-docx` (DOCX) |
| Token Counting | `tiktoken` (cl100k_base) |
| LLM Cache | SQLite (`langchain_community.cache.SQLiteCache`) |
| Deployment | Docker (multi-stage) on Oracle Cloud (OCI `e2.1.micro`) |
| Reverse Proxy | Caddy |
| CI/CD | GitHub Actions |

---

## Project Structure

```
india-policy-agent/
├── core/                     # LangGraph pipeline and supporting engines
│   ├── graph.py              # State, nodes, edges, build_graph()
│   ├── tools.py              # ToolFactory — hybrid search, BM25, cross-encoder
│   ├── ingestion.py          # PDF/DOCX/TXT ingestion — parent-child chunking, ChromaDB
│   ├── prompts.py            # All LLM system prompts
│   ├── judge.py              # Hallucination judge — LLM-as-Judge scorer, LRU cache
│   ├── retrieval_grader.py   # CRAG relevance grader
│   ├── memory_store.py       # Semantic user memory (ChromaDB collection)
│   ├── web_search.py         # DuckDuckGo search tool
│   ├── history.py            # Conversation metadata (SQLite)
│   ├── llm.py                # LLM + grader LLM factory (Groq)
│   ├── rate_limiter.py        # Sliding-window rate limiter
│   └── config.py             # All config constants from env
├── api/                      # FastAPI backend
│   ├── main.py                # App, lifespan startup, static frontend mount
│   ├── singletons.py          # Shared Ingestion/ToolFactory/graph instances
│   ├── deps.py                 # Per-browser user identity (X-User-Id header)
│   └── routes/
│       ├── chat.py            # POST /api/chat/stream (SSE)
│       ├── documents.py       # Upload, batch ingest, list
│       ├── conversations.py   # History list/get/delete
│       └── memories.py        # Memories view, feedback
├── frontend/                 # React chat UI (Vite + TypeScript + Tailwind)
│   └── src/
│       ├── App.tsx            # Layout, streaming state
│       ├── api.ts             # Backend client (fetch + SSE parsing)
│       └── components/        # Sidebar, ChatInput, MessageBubble, modals
├── tests/
│   ├── unit/                  # 159 unit tests
│   └── integration/
├── Dockerfile                 # Multi-stage: build frontend, then serve via FastAPI
├── docker-compose.yml
├── Caddyfile
└── requirements.txt
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► Dense Search      ChromaDB cosine similarity (k×2 candidates)
  ├─► Sparse Search     BM25Okapi with user/source filter masking
  │
  ├─► Score Fusion      hybrid = dense_weight×dense_norm + bm25_weight×bm25_norm
  │                     (both normalised with min-max scaling)
  │
  └─► Cross-Encoder     ms-marco-MiniLM-L-6-v2 reranks top-k candidates
        └─► top results returned with parent chunk expansion
```

**Adaptive Retrieval** — profile auto-selected by query type:

| Profile | k | Dense Weight | BM25 Weight | Top-k after rerank |
|---|---|---|---|---|
| `factual` | 3 | 0.3 | 0.7 | 2 |
| `conceptual` | 8 | 0.7 | 0.3 | 4 |
| `comparative` | 12 | 0.5 | 0.5 | 5 |
| `auto` | — | — | — | classifier picks profile |

---

## Setup

### Local (Docker)

```bash
git clone https://github.com/shivamkr27/Insight-engine-agent.git
cd Insight-engine-agent

# Create .env
cp .env.example .env
# Add your GROQ_API_KEY

docker compose up --build
# App available at http://localhost:8000
```

### Local (dev, without Docker)

```bash
# Backend
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api to :8000
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60
MAX_UPLOAD_SIZE_MB=50
```

There is no login screen. Each browser is assigned a random ID on first
visit (stored in `localStorage`, sent as `X-User-Id`), which is enough to
keep documents, conversation history, and memories separate per user.
Requests without the header fall back to a shared `"default"` identity.

---

## CI/CD Pipeline

GitHub Actions runs on every push to `master`:

```
test      →  pytest tests/unit/ (CPU-only PyTorch)
frontend  →  npm ci && npm run build (type-checks + builds the React app)
scan      →  Trivy CVE scan (CRITICAL + HIGH), needs: [test, frontend]
deploy    →  SSH to OCI VM → docker compose up -d --build, needs: [test, frontend, scan]
```

The `frontend` job exists so a broken TypeScript build fails fast in CI
instead of surfacing partway through the deploy job's Docker build on the
OCI VM.

**Deploy details:**
- `appleboy/ssh-action` with `command_timeout: 30m`
- First build ~15 min (model + npm dependency downloads cached after first run)
- Subsequent builds ~2–3 min (Docker layer cache)

---

## Deployment

Runs on Oracle Cloud Infrastructure `e2.1.micro` (1GB RAM, 1 OCPU):

- **2GB swap** configured for ML model loading
- **ChromaDB in-process** — no separate container (saves ~200MB RAM)
- **Memory limit:** `800m` container, `2500m` with swap
- **Thread control:** `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`
- **Single process, single port (8000):** FastAPI serves both the `/api/*` routes and the built React static files
- Given the tight RAM budget, there's no in-VM monitoring stack (Prometheus/Grafana) — `restart: unless-stopped` plus the container healthcheck (`/api/health`) is the safety net, and an external uptime pinger is a lighter-weight option if visibility is needed later

---

## Tests

```bash
pytest tests/unit/ -v
# tests across: ingestion, graph nodes, tools, retrieval grader,
# hallucination judge, memory store, history, multi-hop, adaptive retrieval
```
