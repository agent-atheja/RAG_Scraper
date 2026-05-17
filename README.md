# W3Schools RAG Chatbot

A production-grade, fully local AI assistant that scrapes **W3Schools**, indexes the content into a **ChromaDB** vector store, and powers a multi-agent chatbot via **LangGraph** + **Claude (Anthropic API)** with a **Streamlit** UI.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│           Streamlit Chatbot UI               │  :8501
└──────────────────┬───────────────────────────┘
                   │ SSE streaming
┌──────────────────▼───────────────────────────┐
│           FastAPI Backend                    │  :8000
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│      LangGraph Multi-Agent Orchestrator      │
│                                              │
│  [Orchestrator / Leader]                     │
│       ├── Python Agent   🐍                  │
│       ├── HTML Agent     🌐                  │
│       ├── CSS Agent      🎨                  │
│       ├── JavaScript Agent ⚡                │
│       ├── SQL Agent      🗄️                  │
│       └── General Agent  🤖                  │
└──────────────────┬───────────────────────────┘
                   │ topic-filtered RAG
┌──────────────────▼───────────────────────────┐
│   ChromaDB (local disk)                      │
│   ~50K chunks · BAAI/bge-small-en-v1.5       │
│   Sourced from: W3Schools full-site scrape   │
└──────────────────────────────────────────────┘
```

---

## Features

- **Full-site scraper** — async `httpx` + `BeautifulSoup`, respects `robots.txt`, sitemap-driven
- **Local embeddings** — `BAAI/bge-small-en-v1.5` via `sentence-transformers` (no API cost)
- **ChromaDB** — persistent local vector DB with per-topic metadata filtering
- **LangGraph orchestration** — stateful Leader → Sub-agent routing with conditional edges
- **6 specialist agents** — Python, HTML, CSS, JavaScript, SQL, General
- **Streaming chatbot** — FastAPI SSE + Streamlit with real-time token output
- **Session management** — per-user conversation history

---

## Project Structure

```
w3schools_rag/
├── scraper/
│   ├── scraper.py       # Async URL fetcher (httpx + retry)
│   ├── parser.py        # BeautifulSoup content extractor
│   ├── chunker.py       # Section-aware text chunker
│   └── pipeline.py      # End-to-end scrape → embed → store
│
├── rag/
│   ├── embedder.py      # sentence-transformers wrapper
│   ├── store.py         # ChromaDB upsert + similarity search
│   └── retriever.py     # Topic-scoped RAG tool factory
│
├── agents/
│   ├── state.py         # LangGraph AgentState TypedDict
│   ├── orchestrator.py  # Leader: classifies query → routes
│   ├── sub_agents.py    # 6 topic-specialist agent factories
│   ├── graph.py         # StateGraph wiring + compiled graph
│   └── run.py           # CLI smoke-test (no UI needed)
│
├── chatbot/
│   ├── session.py       # In-memory session store
│   ├── app.py           # FastAPI REST + SSE streaming API
│   └── ui.py            # Streamlit chatbot frontend
│
├── config.py            # Centralised settings (reads .env)
├── main.py              # Launches FastAPI + Streamlit together
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Uses `X \| None` union syntax |
| Anthropic API key | — | [Get one here](https://console.anthropic.com/settings/keys) |
| RAM | 4 GB min / 8 GB rec | Embedding model + ChromaDB |
| Disk | 2 GB min / 4 GB rec | ChromaDB ~1.2 GB |

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd w3schools_rag

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the scraper *(one-time, ~4–5 hours for full site)*

```bash
# Test with 10 pages first:
python -m scraper.pipeline 10

# Full scrape (all ~5K pages):
python -m scraper.pipeline
```

### 4. Test agents (no UI)

```bash
python agents/run.py "How do I use list comprehension in Python?"
python agents/run.py "What is CSS flexbox?"
```

### 5. Launch the chatbot

```bash
python main.py
# FastAPI  → http://localhost:8000
# Chatbot  → http://localhost:8501
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Single-shot chat (returns full JSON response) |
| `POST` | `/chat/stream` | SSE streaming chat (token-by-token) |
| `GET` | `/history/{session_id}` | Retrieve conversation history |
| `GET` | `/health` | API health + ChromaDB chunk count |

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does Python list comprehension work?"}'
```

```json
{
  "answer": "List comprehension provides a concise way...",
  "agent_used": "python",
  "sources": ["https://www.w3schools.com/python/python_lists_comprehension.asp"],
  "session_id": "abc-123"
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `CHROMA_DB_PATH` | `./chroma_db` | Path to ChromaDB storage |
| `COLLECTION_NAME` | `w3schools` | ChromaDB collection name |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `FASTAPI_PORT` | `8000` | FastAPI server port |
| `STREAMLIT_PORT` | `8501` | Streamlit UI port |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | `httpx` (async, HTTP/2) + `BeautifulSoup4` + `lxml` |
| Embeddings | `sentence-transformers` · `BAAI/bge-small-en-v1.5` |
| Vector DB | `ChromaDB` (local persistent, cosine similarity) |
| LLM | `Claude claude-sonnet-4-6` via Anthropic API |
| Agent orchestration | `LangGraph` + `LangChain-Anthropic` |
| API | `FastAPI` + `uvicorn` (SSE streaming) |
| UI | `Streamlit` |

---

## Cost Estimate

Each conversation turn makes 2 LLM calls (orchestrator + sub-agent):

| Call | Tokens | Cost (approx.) |
|---|---|---|
| Orchestrator (classification) | ~100 input | ~$0.0003 |
| Sub-agent (RAG answer) | ~2K input + ~500 output | ~$0.009 |
| **Per turn total** | | **~$0.01** |

Embeddings run fully locally — **no embedding API cost**.

---

## Development

```bash
# Run only the API server
uvicorn chatbot.app:app --reload --port 8000

# Run only the Streamlit UI
streamlit run chatbot/ui.py --server.port 8501

# Run agent CLI test
python agents/run.py "What is a CSS selector?"
```

---

## License

MIT
