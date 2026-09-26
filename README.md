# MEMTRACE

**MEMTRACE gives AI agents a memory that knows what's still true.**

Most AI systems remember everything by dumping old conversation history into a search index and pulling back "similar" chunks. The problem: similar isn't the same as *current*. If your team switched from MongoDB to PostgreSQL last month, a plain similarity search will happily hand the agent the old MongoDB fact right alongside the new one — and the agent has no way to know which one is still true.

MEMTRACE fixes that. It treats memory as **structured, evolving facts** instead of a pile of text:

- When something changes, MEMTRACE keeps the old fact (marked *historical*) and creates the new one (marked *active*) — nothing is silently overwritten.
- When the agent needs to answer a question, MEMTRACE decides which facts are still valid *right now*, follows the relationships between them, and hands the agent only what's relevant.
- Every answer comes with a paper trail: which memories were used, which were excluded, and why.

## Why this matters

> "What database are we currently using?"

A plain vector-search agent might answer **"MongoDB"** — technically once true, currently wrong. MEMTRACE answers **"PostgreSQL"**, and can explain exactly why MongoDB was left out: *"Superseded by PostgreSQL, which is now active for this subject."*

## Two ways to look at it

MEMTRACE ships with one dashboard, two audiences:

- **Executive View** (the default) answers *"how much money are we saving?"* — a headline savings figure, a savings-over-time chart, a "what happens at your scale?" calculator, a breakdown of where AI budget was being wasted, and a "find the source of the mistake" replay that shows an agent giving a wrong (outdated) answer versus the corrected one.
- **Engineering View** answers *"why did the agent retrieve this?"* — the same underlying pipeline, but shown as the actual memories selected/excluded, their scores and reasons, the raw context sent to the model, and the LangSmith trace ID when tracing is enabled.

Every dollar figure in the Executive View is computed from real, recorded pipeline activity (what was actually retrieved vs. what was actually kept) multiplied by a configurable price table — never a hardcoded claim. When there isn't enough recorded activity yet to project from, the dashboard says so plainly instead of guessing.

## How it works, in plain terms

1. **Something happens** — a conversation, a decision, a status update. MEMTRACE treats it as an *event*.
2. **MEMTRACE reads the event** and pulls out the facts inside it (who/what changed, and to what).
3. **A judgment step decides what to do** with each fact: is this brand new? Does it replace something we already knew? Is it just restating something we already have? Is it too vague to trust yet?
4. **The memory graph updates** accordingly — old facts are kept for history, new facts become active, and the two are linked together so the story of *how* something changed is never lost.
5. **When a question comes in**, MEMTRACE finds the memories that are both *relevant* to the question and *currently valid*, builds a small, focused briefing for the AI model, and answers — while keeping a record of what it used and what it deliberately left out.

Nothing here requires a paid API key to run — if no OpenAI/LLM key is configured, MEMTRACE falls back to deterministic, fully-tested logic so the whole pipeline still works end to end for local development and demos.

## Quick start

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

Then open **http://localhost:8000/** in your browser. Click **"Seed demo data"** to load a small evolving story about a fictional project ("Project Alpha") — a database migration, a rejected alternative, a deployment move, and a couple of decisions still being considered — and start asking it questions.

Try asking:
- *"What database are we currently using?"*
- *"What database did we use before?"*
- *"What does Project Alpha depend on?"*

**Frontend development:** instead of rebuilding on every change, run `npm run dev` inside `frontend/` (with the backend already running on port 8000) for hot-reload at `http://localhost:5173` — it proxies API calls back to the backend automatically.

## What's actually happening under the hood

| Plain-language name | What it does |
|---|---|
| **Memory** | A single structured fact: subject, relationship, value, plus its status (active / historical / archived / pending review / deleted) and where it came from. |
| **Judgment** | The decision of what to do with a new fact — add it, update an old one, merge it with something we already know, archive it, delete it, or flag it for review. |
| **Retrieval** | Finding the memories relevant to a question by combining meaning-based search, the relationships between facts, and whether each fact is still currently valid. |
| **Context** | The short, curated briefing actually handed to the AI model — never the entire memory store. |
| **Cost** | What the curated briefing actually cost in model spend, versus what an unfiltered "send everything relevant" approach would have cost — the difference is the savings shown on the dashboard. |

Optional integrations, both designed to degrade gracefully if unconfigured:
- **TypeSafe JEV** — used for the bounded add/update/merge/archive/delete/review judgment calls, in place of the built-in rule-based fallback.
- **LangSmith** — full execution tracing across every step of the pipeline, when an API key is provided.

## Project layout

```
backend/app/
  memory/       the memory model, storage, extraction, lifecycle rules, and retrieval logic
  llm/          the model client (real OpenAI-compatible client, or an offline fallback)
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

This runs the full test suite, including an end-to-end comparison: on the demo story, MEMTRACE's approach answers **10 out of 10** evaluation questions correctly, versus **5 out of 10** for a naive "search for similar text" baseline — because the baseline has no way to tell current facts from outdated ones.

## Configuration

Copy `.env.example` to `.env` and fill in only what you have — everything is optional:

| Variable | What it's for |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | Any OpenAI-compatible model provider, for real answer generation and embeddings. |
| `TYPESAFE_API_KEY` | Enables TypeSafe JEV for memory judgments. |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | Enables LangSmith tracing. |
| `MEMTRACE_DB_PATH` | Where the local SQLite database file lives. |

With none of these set, MEMTRACE still runs completely — it just uses its built-in, deterministic fallbacks instead of external services.
