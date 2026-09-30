import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models import Course, CourseChunk, Lesson
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
            Lesson(
                lesson_number=1, title="Fixtures and Mocks", lesson_link=LESSON_LINKS[1]
            ),
            Lesson(
                lesson_number=2,
                title="Vector Search Evaluation",
                lesson_link=LESSON_LINKS[2],
            ),
        ],
    )


@pytest.fixture
def sample_chunks():
    return [
        CourseChunk(
            content="Fixtures provide reusable setup for tests. A mock object replaces a real dependency.",
            course_title=COURSE_TITLE,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Retrieval quality is measured with recall at k over the top five results.",
            course_title=COURSE_TITLE,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Embeddings map text into a vector space where similar meanings are close together.",
            course_title=COURSE_TITLE,
            lesson_number=2,
            chunk_index=2,
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
        pytest.skip(
            "backend/chroma_db not built yet - start the app once to ingest docs/"
        )
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
    content.append(
        SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)
    )
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
