# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Retrieval-Augmented Generation (RAG) system for querying course materials. FastAPI backend, vanilla JS frontend, ChromaDB for vector storage, Anthropic Claude for generation. Python 3.13, dependency/venv management via `uv`.

## Commands

Run the app (from repo root):
```bash
./run.sh
```
or manually:
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

Install/sync dependencies:
```bash
uv sync
```

**Always use `uv` to run the server and manage dependencies — never call `pip` or `python` directly.**

- Web interface: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`
- Requires `ANTHROPIC_API_KEY` set in a `.env` file at the repo root (see `.env.example`).
- On Windows, use Git Bash to run these commands (`run.sh` is a bash script).
- There is no test suite, linter, or build step configured in this repo.

## Architecture

The core flow: a query enters via FastAPI, gets routed through a RAG orchestrator that calls Claude with a search tool attached, and Claude decides whether to invoke the tool against ChromaDB before producing a final answer.

- **`backend/app.py`** — FastAPI app. `POST /api/query` (query + optional session_id → answer + sources + session_id) and `GET /api/courses` (course stats). Mounts `frontend/` as static files and auto-ingests `docs/*.txt` on startup via `RAGSystem.add_course_folder`.
- **`backend/rag_system.py`** (`RAGSystem`) — orchestrator that wires all components together. `query()` builds a prompt, pulls session history, calls the AI generator with tools enabled, retrieves sources from the tool manager after the call, and updates session history.
- **`backend/document_processor.py`** — parses course transcript `.txt` files with an expected header format (`Course Title:` / `Course Link:` / `Course Instructor:` on the first lines, then `Lesson N: <title>` markers with optional `Lesson Link:`), and does sentence-aware chunking with configurable size/overlap (see `config.py`).
- **`backend/vector_store.py`** (`VectorStore`) — wraps ChromaDB with two collections: `course_catalog` (titles/instructors/lesson metadata, used to fuzzy-resolve a course name to its exact stored title) and `course_content` (chunked text, semantically searched with optional course/lesson filters).
- **`backend/ai_generator.py`** (`AIGenerator`) — calls Anthropic's Messages API. System prompt instructs Claude to search only for course-specific questions, **one search per query maximum**. If Claude responds with `stop_reason: tool_use`, `_handle_tool_execution()` runs the tool call(s), appends results, and makes a **second** Claude call (tools disabled) to get the final synthesized answer.
- **`backend/search_tools.py`** — `Tool`/`ToolManager` abstraction for Anthropic tool-calling. `CourseSearchTool` is the one registered tool (`search_course_content`: query + optional course_name + lesson_number). Tracks the last search's sources on `self.last_sources` as a side channel — the UI's source list does not come from Claude's text response, it's read/reset by `RAGSystem` after each query via `tool_manager.get_last_sources()` / `reset_sources()`.
- **`backend/session_manager.py`** — in-memory, per-session conversation history (dict, capped length). Not persisted; restarting the server clears all sessions.
- **`backend/models.py`** — Pydantic models: `Course`, `Lesson`, `CourseChunk`.
- **`backend/config.py`** — central config dataclass (chunk size/overlap, max search results, max history, Chroma path, model name). Loaded from `.env` via `python-dotenv`.
- **`frontend/`** — static HTML/CSS/JS chat UI. `script.js` posts to `/api/query`, renders the answer as Markdown, and shows sources in a collapsible details block.
- **`docs/`** — course transcript `.txt` files, auto-loaded into the vector store on server startup (existing course titles are skipped, not re-ingested).

## Notes for making changes

- The "one search per query" behavior is enforced only by the system prompt text in `ai_generator.py`, not by code — there's no hard limit on tool-call rounds.
- Adding a new tool means implementing the `Tool` ABC in `search_tools.py` and registering it with `ToolManager` in `rag_system.py`'s `__init__`.
- Course documents must follow the exact expected format for `document_processor.py` to parse metadata and lessons correctly (see the docstring in `process_course_document`).
