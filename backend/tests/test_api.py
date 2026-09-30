"""Tests for the FastAPI endpoints' request/response handling"""

import pytest

from tests.conftest import COURSE_TITLE, FRONTEND_HTML, LESSON_LINKS

pytestmark = pytest.mark.api


class TestQueryEndpoint:

    def test_returns_answer_sources_and_session(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What are fixtures?"})

        assert resp.status_code == 200
        assert resp.json() == {
            "answer": "Fixtures provide reusable setup.",
            "sources": [
                {"text": f"{COURSE_TITLE} - Lesson 1", "link": LESSON_LINKS[1]}
            ],
            "session_id": "session_1",
        }
        mock_rag_system.query.assert_called_once_with("What are fixtures?", "session_1")

    def test_creates_new_session_per_sessionless_request(self, client):
        first = client.post("/api/query", json={"query": "a"}).json()["session_id"]
        second = client.post("/api/query", json={"query": "b"}).json()["session_id"]
        assert first != second

    def test_reuses_given_session_id(self, client, mock_rag_system):
        resp = client.post(
            "/api/query", json={"query": "follow-up", "session_id": "abc"}
        )

        assert resp.json()["session_id"] == "abc"
        mock_rag_system.query.assert_called_once_with("follow-up", "abc")
        assert mock_rag_system.session_manager.sessions == {}

    def test_source_without_link_serializes_as_null(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer", [{"text": COURSE_TITLE}])
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.json()["sources"] == [{"text": COURSE_TITLE, "link": None}]

    def test_no_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Hello!", [])
        resp = client.post("/api/query", json={"query": "hi"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []

    @pytest.mark.parametrize(
        "body", [{}, {"session_id": "abc"}, {"query": None}, {"query": ["x"]}]
    )
    def test_invalid_body_is_rejected(self, client, mock_rag_system, body):
        resp = client.post("/api/query", json=body)
        assert resp.status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_non_json_body_is_rejected(self, client):
        resp = client.post(
            "/api/query", content="just text", headers={"Content-Type": "text/plain"}
        )
        assert resp.status_code == 422

    def test_rag_error_becomes_500_with_detail(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("Anthropic API unavailable")
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Anthropic API unavailable"}

    def test_malformed_sources_become_500(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer", [{"link": "no text"}])
        assert client.post("/api/query", json={"query": "q"}).status_code == 500

    def test_get_is_not_routed_to_query(self, client, mock_rag_system):
        # Not 405: unmatched methods fall through to the static mount at "/"
        assert client.get("/api/query").status_code == 404
        mock_rag_system.query.assert_not_called()


class TestCoursesEndpoint:

    def test_returns_course_stats(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        assert resp.json() == {"total_courses": 1, "course_titles": [COURSE_TITLE]}

    def test_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        assert client.get("/api/courses").json() == {
            "total_courses": 0,
            "course_titles": [],
        }

    def test_analytics_error_becomes_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("chroma down")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "chroma down"}

    def test_missing_analytics_key_becomes_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {"total_courses": 1}
        assert client.get("/api/courses").status_code == 500


class TestSessionEndpoint:

    def test_clears_session_history(self, client, mock_rag_system):
        sm = mock_rag_system.session_manager
        sid = sm.create_session()
        sm.add_exchange(sid, "q", "a")

        resp = client.delete(f"/api/session/{sid}")

        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        assert sm.get_conversation_history(sid) is None

    def test_unknown_session_is_ok(self, client):
        assert client.delete("/api/session/does-not-exist").json() == {"success": True}


class TestRoot:

    def test_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.text == FRONTEND_HTML

    def test_unknown_static_path_is_404(self, client):
        assert client.get("/missing.js").status_code == 404

    def test_api_routes_take_precedence_over_static_mount(self, client):
        assert client.get("/api/courses").headers["content-type"] == "application/json"
