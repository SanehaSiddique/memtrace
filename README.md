# MEMTRACE

**MEMTRACE gives AI agents a memory that knows what's still true.**

Most AI systems remember everything by dumping old conversation history into a search index and pulling back "similar" chunks. The problem: similar isn't the same as *current*. If your team switched from MongoDB to PostgreSQL last month, a plain similarity search will happily hand the agent the old MongoDB fact right alongside the new one — and the agent has no way to know which one is still true.

MEMTRACE fixes that. It treats memory as **structured, evolving facts** instead of a pile of text:

- When something changes, MEMTRACE keeps the old fact (marked *historical*) and creates the new one (marked *active*) — nothing is silently overwritten.
- When the agent needs to answer a question, MEMTRACE decides which facts are still valid *right now*, follows the relationships between them, and hands the agent only what's relevant.
- Every answer comes with a paper trail: which memories were used, which were excluded, and why — and, since every avoided piece of unnecessary context has a dollar cost, what it saved.

## Why this matters

> "What database are we currently using?"

A plain vector-search agent might answer **"MongoDB"** — technically once true, currently wrong. MEMTRACE answers **"PostgreSQL"**, and can explain exactly why MongoDB was left out: *"Superseded by PostgreSQL, which is now active for this subject."*

## Two ways to look at it, one running system

Open the dashboard and you get one running app with two audiences:

- **Executive View** (the default) — a live, interactive memory graph you can poke at (simulate a real update, run a real query, watch the agent decide), plus the money side of the story: AI spend avoided, a savings-over-time chart, a "what happens at your scale?" calculator, a breakdown of exactly which memory decisions saved money, and a "find the source of the mistake" replay comparing a naive agent's wrong answer to MEMTRACE's correct one.
- **Engineering View** — the same pipeline from underneath: the memories actually selected/excluded with their scores and reasons, the raw context sent to the model, and the LangSmith trace ID when tracing is enabled.

Every dollar figure and every graph edge you see is computed from real, recorded activity — nothing is drawn from fake frontend-only state. Where a number is a projection rather than something already measured, the dashboard says so explicitly (labeled ACTUAL / CALCULATED / PROJECTED / DEMO).

## Quick start

You need either **Docker** (easiest, one command) or **Python 3.11+ and Node 20+** installed locally (this project is developed and tested against Python 3.14).

### Option A — Docker (recommended)

```bash
git clone https://github.com/SanehaSiddique/memtrace.git && cd memtrace
cp .env.example .env          # optional — works with no keys at all
docker compose up --build
```

Open **http://localhost:8000/**. That's it — the container builds the React dashboard, installs the Python backend, and serves both from one process. Memory data persists in a Docker volume (`memtrace_data`) across restarts.

Common follow-ups:

```bash
docker compose up -d --build     # run in the background
docker compose logs -f           # tail logs
docker compose down              # stop (keeps the memtrace_data volume, i.e. your memories)
docker compose down -v           # stop AND wipe stored memory — start completely fresh
```

Rebuild (`--build`) whenever you change backend or frontend source; `docker compose up` alone just restarts the existing image.

### Option B — run it directly

```bash
# 1. Set up the backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # works as-is with no keys

# 2. Run the tests
pytest

# 3. Build the dashboard (one-time, or after changing frontend/)
cd frontend
npm install
npm run build
cd ..

# 4. Start the server (serves both the API and the built dashboard)
cd backend
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/** in your browser either way. Click **"Seed demo data"** to load a small evolving story about a fictional project ("Project Alpha") — a database migration, a rejected alternative, a deployment move, and a couple of decisions still being considered — and start asking it questions.

Try asking:
- *"What database are we currently using?"*
- *"What database did we use before?"*
- *"What does Project Alpha depend on?"*

**Frontend development:** instead of rebuilding on every change, run `npm run dev` inside `frontend/` (with the backend already running on port 8000, either via Docker or `uvicorn`) for hot-reload at `http://localhost:5173` — it proxies API calls back to the backend automatically.

## How it works, in plain terms

1. **Something happens** — a conversation, a decision, a status update. MEMTRACE treats it as an *event*.
2. **MEMTRACE reads the event** and pulls out the facts inside it (who/what changed, and to what).
3. **A judgment step decides what to do** with each fact: is this brand new? Does it replace something we already knew? Is it just restating something we already have? Is it too vague to trust yet?
4. **The memory graph updates** accordingly — old facts are kept for history, new facts become active, and the two are linked together so the story of *how* something changed is never lost.
5. **When a question comes in**, MEMTRACE finds the memories that are both *relevant* to the question and *currently valid*, builds a small, focused briefing for the AI model, and answers — while keeping a record of what it used and what it deliberately left out, and what that saved.

Runtime answers always come from a configured live LLM. The server fails clearly at startup when no live LLM or JEV provider is configured; it never substitutes a fabricated offline answer.

## What's actually happening under the hood

| Plain-language name | What it does |
|---|---|
| **Memory** | A single structured fact: subject, relationship, value, plus its status (active / historical / archived / pending review / deleted) and where it came from. |
| **Judgment** | The decision of what to do with a new fact — add it, update an old one, merge it with something we already know, archive it, delete it, or flag it for review. |
| **Retrieval** | Finding the memories relevant to a question by combining meaning-based search, the relationships between facts, and whether each fact is still currently valid. |
| **Context** | The short, curated briefing actually handed to the AI model — never the entire memory store. |
| **Cost** | What the curated briefing actually cost in model spend, versus what an unfiltered "send everything relevant" approach would have cost — the difference is the savings shown on the dashboard, and it's traceable down to the specific memory decision that caused it. |

Provider integrations:
- **Vercel AI Gateway + TypeSafe JEV** — required for bounded add/update/merge/archive/delete/review judgments (`AI_GATEWAY_API_KEY`).
- **LangSmith** — optional full execution tracing across every step of the pipeline.

## Project layout

```
Dockerfile             multi-stage build: React dashboard -> static files, served by the FastAPI backend
docker-compose.yml     one-command local run, with persistent storage for memory data
backend/app/
  memory/       the memory model, storage, extraction, lifecycle rules, and retrieval logic
  llm/          live OpenAI-compatible model clients
  judgment/     the ADD/UPDATE/MERGE/ARCHIVE/DELETE/REVIEW decision logic
  context/      builds the final, minimal briefing handed to the model
  cost/         turns retrieval activity into dollar figures (configurable pricing, savings, projections)
  agent/        the step-by-step pipeline (LangGraph) for ingesting events and answering queries
  tracing/      LangSmith wiring
  api/          the web API
  evaluation/   a 10-case test suite comparing MEMTRACE against naive search
  demo/         the sample "Project Alpha" story used for demos
frontend/       the React dashboard (Executive View + Engineering View)
backend/tests/  automated tests for all of the above
```

## Checking it actually works

```bash
pytest
```

This runs the hermetic test suite. The dashboard's `/evaluate` flow uses your configured live model, so its measured answers and accuracy can vary by model.

## Configuration

Copy `.env.example` to `.env`. Configure OpenAI and Vercel AI Gateway at minimum:

| Variable | What it's for |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | OpenAI model used for real answer generation, extraction, and embeddings. OpenAI takes precedence when configured. |
| `AI_GATEWAY_API_KEY`, `JEV_URL` | Vercel AI Gateway credentials and the native JEV evaluation endpoint. |
| `G8_API_KEY`, `G8_MCP_URL` | Graph8 MCP credentials and endpoint. |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Optional alternate live LLM used only when OpenAI is absent. |
| `TYPESAFE_API_KEY` | Optional direct TypeSafe JEV fallback when the Vercel gateway key is absent. |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | Enables LangSmith tracing. |
| `MEMTRACE_DB_PATH` | Where the local SQLite database file lives. Overridden to `/data/memtrace.db` inside Docker automatically. |

If live LLM or JEV credentials are missing, startup fails with a configuration error instead of serving fake answers or judgments.

### Using OpenRouter for a free real LLM

If you deliberately do not want OpenAI, [OpenRouter](https://openrouter.ai) can serve as an alternate live answer-generation provider.

1. Sign up at https://openrouter.ai and create a key at https://openrouter.ai/keys (no card required for free models).
2. Pick a current free model from https://openrouter.ai/models?max_price=0 — the list changes over time as providers rotate models in and out.
3. Set in your `.env`:
   ```bash
   OPENROUTER_API_KEY="sk-or-v1-..."
   OPENROUTER_MODEL="nvidia/nemotron-3-super-120b-a12b:free"   # or any other ":free" model id
   ```
4. Restart the backend (`docker compose up --build`, or re-run `uvicorn` if running locally).

Note: OpenRouter uses the local hashing embedder for retrieval. Memory judgments still require Vercel JEV or a direct TypeSafe JEV key.

## Troubleshooting

- **Docker build fails to reach npm/pip registries** — you're likely behind a proxy or offline; Docker needs network access during the build.
- **Port 8000 already in use** — change the host-side port in `docker-compose.yml` (`"8000:8000"` → e.g. `"8001:8000"`), or stop whatever else is using it.
- **Want a clean slate** — `docker compose down -v` removes the persisted memory database along with the containers.
- **Changes not showing up** — Docker: re-run with `--build`. Manual: re-run `npm run build` in `frontend/` before restarting `uvicorn`.
