"""Tests for how the RAG system handles content-related questions"""

import sys
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from config import config as app_config
from rag_system import RAGSystem
from tests.conftest import (
    BACKEND_DIR,
    COURSE_TITLE,
    LESSON_LINKS,
    make_text_response,
    make_tool_use_response,
)


@pytest.fixture
def rag(tmp_path, sample_doc_path):
    cfg = replace(
        app_config, CHROMA_PATH=str(tmp_path / "chroma"), ANTHROPIC_API_KEY="test-key"
    )
    system = RAGSystem(cfg)
    course, n_chunks = system.add_course_document(str(sample_doc_path))
    assert course is not None and n_chunks > 0
    system.ai_generator.client = MagicMock()
    return system


def script_claude(rag, tool_input, answer="Recall at k."):
    rag.ai_generator.client.messages.create.side_effect = [
        make_tool_use_response("search_course_content", tool_input),
        make_text_response(answer),
    ]


class TestContentQuery:

    def test_document_ingestion(self, rag):
        assert rag.get_course_analytics() == {
            "total_courses": 1,
            "course_titles": [COURSE_TITLE],
        }

    def test_returns_answer_and_linked_sources(self, rag):
        script_claude(
            rag,
            {"query": "retrieval quality", "course_name": "RAG", "lesson_number": 2},
        )
        answer, sources = rag.query("How is retrieval quality measured?")

        assert answer == "Recall at k."
        assert sources
        assert all(
            s == {"text": f"{COURSE_TITLE} - Lesson 2", "link": LESSON_LINKS[2]}
            for s in sources
        )

    def test_prompt_and_tools_sent_to_claude(self, rag):
        script_claude(rag, {"query": "x"})
        rag.query("What are fixtures?")
        first = rag.ai_generator.client.messages.create.call_args_list[0].kwargs
        assert "What are fixtures?" in first["messages"][0]["content"]
        assert {t["name"] for t in first["tools"]} == {
            "search_course_content",
            "get_course_outline",
        }

    def test_sources_reset_after_query(self, rag):
        script_claude(rag, {"query": "fixtures"})
        rag.query("What are fixtures?")
        assert rag.tool_manager.get_last_sources() == []

        rag.ai_generator.client.messages.create.side_effect = None
        rag.ai_generator.client.messages.create.return_value = make_text_response(
            "Hello!"
        )
        _, sources = rag.query("hi")
        assert sources == []

    def test_session_history_updated(self, rag):
        session = rag.session_manager.create_session()
        script_claude(rag, {"query": "fixtures"})
        rag.query("What are fixtures?", session_id=session)

        history = rag.session_manager.get_conversation_history(session)
        assert "User: What are fixtures?" in history
        assert "Assistant: Recall at k." in history

    def test_history_sent_on_follow_up(self, rag):
        session = rag.session_manager.create_session()
        script_claude(rag, {"query": "fixtures"})
        rag.query("What are fixtures?", session_id=session)

        script_claude(rag, {"query": "more"})
        rag.query("Tell me more", session_id=session)
        system = rag.ai_generator.client.messages.create.call_args_list[2].kwargs[
            "system"
        ]
        assert "What are fixtures?" in system

    def test_tool_failure_does_not_crash_query(self, rag):
        script_claude(rag, {"query": "x"})
        with patch.object(rag.search_tool, "execute", side_effect=RuntimeError("boom")):
            answer, _ = rag.query("What are fixtures?")
        assert isinstance(answer, str)


class TestApiEndpoint:

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.chdir(BACKEND_DIR)  # app.py uses ./chroma_db and ../frontend
        sys.modules.pop("app", None)
        import app as app_module

        return app_module, TestClient(app_module.app)

    def test_query_success_shape(self, client):
        app_module, tc = client
        with patch.object(
            app_module.rag_system,
            "query",
            return_value=("answer", [{"text": "C - Lesson 1", "link": None}]),
        ):
            res = tc.post("/api/query", json={"query": "q"})
        assert res.status_code == 200
        body = res.json()
        assert body["answer"] == "answer"
        assert body["sources"] == [{"text": "C - Lesson 1", "link": None}]
        assert body["session_id"]

    def test_backend_exception_becomes_500(self, client):
        """This is what the frontend shows as 'Query failed'"""
        app_module, tc = client
        with patch.object(
            app_module.rag_system, "query", side_effect=RuntimeError("boom")
        ):
            res = tc.post("/api/query", json={"query": "q"})
        assert res.status_code == 500
        assert res.json()["detail"] == "boom"


@pytest.mark.live
class TestLiveProduction:

    def test_real_content_question(self, api_key, monkeypatch):
        """Exact production path: real config, real chroma_db, real Claude"""
        monkeypatch.chdir(BACKEND_DIR)
        system = RAGSystem(replace(app_config, ANTHROPIC_API_KEY=api_key))
        answer, sources = system.query(
            "What is covered in lesson 1 of the Building Towards Computer Use with Anthropic course?"
        )
        assert answer.strip()
        assert sources, f"no search sources; answer was: {answer!r}"
