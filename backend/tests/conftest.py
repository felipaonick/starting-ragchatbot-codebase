import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
PROD_CHROMA_PATH = BACKEND_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COURSE_TITLE = "Testing RAG Systems with Pytest"
COURSE_LINK = "https://example.com/rag-testing"
LESSON_LINKS = {
    1: "https://example.com/rag-testing/lesson-1",
    2: "https://example.com/rag-testing/lesson-2",
}

# A course document in the exact format DocumentProcessor expects
SAMPLE_COURSE_DOC = f"""Course Title: {COURSE_TITLE}
Course Link: {COURSE_LINK}
Course Instructor: Ada Tester

Lesson 1: Fixtures and Mocks
Lesson Link: {LESSON_LINKS[1]}
Fixtures provide reusable setup for tests. A mock object replaces a real dependency so a unit can be tested in isolation. The MagicMock class records every call made to it.

Lesson 2: Vector Search Evaluation
Lesson Link: {LESSON_LINKS[2]}
Retrieval quality is measured with recall at k. A good retriever returns the relevant chunk among the top five results. Embeddings map text into a vector space where similar meanings are close together.
"""


# ---------- sample data ----------

@pytest.fixture
def sample_course():
    return Course(
        title=COURSE_TITLE,
        course_link=COURSE_LINK,
        instructor="Ada Tester",
        lessons=[
            Lesson(lesson_number=1, title="Fixtures and Mocks", lesson_link=LESSON_LINKS[1]),
            Lesson(lesson_number=2, title="Vector Search Evaluation", lesson_link=LESSON_LINKS[2]),
        ],
    )


@pytest.fixture
def sample_chunks():
    return [
        CourseChunk(
            content="Fixtures provide reusable setup for tests. A mock object replaces a real dependency.",
            course_title=COURSE_TITLE, lesson_number=1, chunk_index=0,
        ),
        CourseChunk(
            content="Retrieval quality is measured with recall at k over the top five results.",
            course_title=COURSE_TITLE, lesson_number=2, chunk_index=1,
        ),
        CourseChunk(
            content="Embeddings map text into a vector space where similar meanings are close together.",
            course_title=COURSE_TITLE, lesson_number=2, chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_doc_path(tmp_path):
    path = tmp_path / "sample_course.txt"
    path.write_text(SAMPLE_COURSE_DOC, encoding="utf-8")
    return path


# ---------- vector stores ----------

@pytest.fixture
def temp_vector_store(tmp_path, sample_course, sample_chunks):
    """A real ChromaDB store in a temp dir, loaded with the sample course"""
    store = VectorStore(str(tmp_path / "chroma"), EMBEDDING_MODEL, max_results=5)
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)
    return store


@pytest.fixture
def prod_vector_store():
    """The real store the app uses (queried only, never written)"""
    if not PROD_CHROMA_PATH.exists():
        pytest.skip("backend/chroma_db not built yet - start the app once to ingest docs/")
    return VectorStore(str(PROD_CHROMA_PATH), EMBEDDING_MODEL, max_results=5)


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.side_effect = lambda title, n: LESSON_LINKS.get(n)
    store.get_course_link.return_value = COURSE_LINK
    return store


# ---------- fake Anthropic responses ----------

def make_text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
    )


def make_tool_use_response(name, tool_input, tool_id="toolu_test_1", preamble=None):
    content = []
    if preamble:
        content.append(SimpleNamespace(type="text", text=preamble))
    content.append(SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id))
    return SimpleNamespace(content=content, stop_reason="tool_use")


@pytest.fixture
def api_key():
    """Real API key for live tests; skips when not configured"""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


# ---------- API test app ----------
#
# backend/app.py builds a real RAGSystem at import time and mounts ../frontend,
# which doesn't resolve when pytest runs from the repo root. So API tests use a
# test app exposing the same endpoints, backed by a mock RAG system.

FRONTEND_HTML = "<!doctype html><html><body><h1>Course Materials Assistant</h1></body></html>"


def create_test_app(rag_system, static_dir=None):
    """Build a FastAPI app mirroring backend/app.py's API around the given RAG system"""
    from typing import List, Optional

    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    app = FastAPI(title="Course Materials RAG System (test)")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


@pytest.fixture
def mock_rag_system():
    """A RAGSystem stand-in with canned answers and a real SessionManager"""
    from session_manager import SessionManager

    rag = MagicMock()
    rag.session_manager = SessionManager(max_history=2)
    rag.query.return_value = (
        "Fixtures provide reusable setup.",
        [{"text": f"{COURSE_TITLE} - Lesson 1", "link": LESSON_LINKS[1]}],
    )
    rag.get_course_analytics.return_value = {"total_courses": 1, "course_titles": [COURSE_TITLE]}
    return rag


@pytest.fixture
def frontend_dir(tmp_path):
    """A stand-in for frontend/ with a minimal index.html"""
    static = tmp_path / "frontend"
    static.mkdir()
    (static / "index.html").write_text(FRONTEND_HTML, encoding="utf-8")
    return static


@pytest.fixture
def client(mock_rag_system, frontend_dir):
    from fastapi.testclient import TestClient

    with TestClient(create_test_app(mock_rag_system, frontend_dir)) as c:
        yield c
