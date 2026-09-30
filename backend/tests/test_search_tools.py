"""Tests for CourseSearchTool.execute and its ToolManager dispatch"""
import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from tests.conftest import COURSE_TITLE, COURSE_LINK, LESSON_LINKS


def results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


# ---------- execute() with a mocked store: pure output logic ----------

class TestExecuteWithMockStore:

    def test_passes_parameters_to_store(self, mock_vector_store):
        mock_vector_store.search.return_value = results([], [])
        CourseSearchTool(mock_vector_store).execute(
            query="mocks", course_name="RAG", lesson_number=1
        )
        mock_vector_store.search.assert_called_once_with(
            query="mocks", course_name="RAG", lesson_number=1
        )

    def test_returns_store_error_verbatim(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults.empty(
            "No course found matching 'Nope'"
        )
        out = CourseSearchTool(mock_vector_store).execute(query="x", course_name="Nope")
        assert out == "No course found matching 'Nope'"

    @pytest.mark.parametrize("kwargs, expected", [
        ({}, "No relevant content found."),
        ({"course_name": "RAG"}, "No relevant content found in course 'RAG'."),
        ({"lesson_number": 2}, "No relevant content found in lesson 2."),
        ({"course_name": "RAG", "lesson_number": 2},
         "No relevant content found in course 'RAG' in lesson 2."),
    ])
    def test_empty_results_message(self, mock_vector_store, kwargs, expected):
        mock_vector_store.search.return_value = results([], [])
        assert CourseSearchTool(mock_vector_store).execute(query="x", **kwargs) == expected

    def test_formats_results_with_headers(self, mock_vector_store):
        mock_vector_store.search.return_value = results(
            ["chunk one", "chunk two"],
            [{"course_title": COURSE_TITLE, "lesson_number": 1},
             {"course_title": COURSE_TITLE, "lesson_number": None}],
        )
        out = CourseSearchTool(mock_vector_store).execute(query="x")
        assert out == (
            f"[{COURSE_TITLE} - Lesson 1]\nchunk one\n\n"
            f"[{COURSE_TITLE}]\nchunk two"
        )

    def test_tracks_sources_with_links(self, mock_vector_store):
        mock_vector_store.search.return_value = results(
            ["a", "b"],
            [{"course_title": COURSE_TITLE, "lesson_number": 2},
             {"course_title": COURSE_TITLE}],
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="x")
        assert tool.last_sources == [
            {"text": f"{COURSE_TITLE} - Lesson 2", "link": LESSON_LINKS[2]},
            {"text": COURSE_TITLE, "link": COURSE_LINK},
        ]


# ---------- execute() against a real (temporary) ChromaDB ----------

class TestExecuteWithRealStore:

    def test_plain_query_returns_relevant_chunk(self, temp_vector_store):
        out = CourseSearchTool(temp_vector_store).execute(query="What is a mock object?")
        assert "mock object replaces a real dependency" in out
        assert out.startswith(f"[{COURSE_TITLE} - Lesson")

    def test_partial_course_name_resolves(self, temp_vector_store):
        tool = CourseSearchTool(temp_vector_store)
        out = tool.execute(query="recall at k", course_name="RAG testing")
        assert "recall at k" in out
        assert all(s["text"].startswith(COURSE_TITLE) for s in tool.last_sources)

    def test_lesson_filter_restricts_results(self, temp_vector_store):
        tool = CourseSearchTool(temp_vector_store)
        out = tool.execute(query="testing", lesson_number=1)
        assert "Lesson 2" not in out
        assert tool.last_sources == [
            {"text": f"{COURSE_TITLE} - Lesson 1", "link": LESSON_LINKS[1]}
        ]

    def test_results_never_exceed_max_results(self, temp_vector_store):
        tool = CourseSearchTool(temp_vector_store)
        tool.execute(query="testing")
        assert 0 < len(tool.last_sources) <= temp_vector_store.max_results


# ---------- execute() against the production DB the app is using ----------

class TestExecuteWithProductionStore:

    def test_content_query_returns_results(self, prod_vector_store):
        tool = CourseSearchTool(prod_vector_store)
        out = tool.execute(query="What is computer use?")
        assert not out.startswith("Search error"), out
        assert not out.startswith("No relevant content"), out
        assert out.startswith("["), out
        assert tool.last_sources

    def test_course_filtered_query_returns_results(self, prod_vector_store):
        tool = CourseSearchTool(prod_vector_store)
        out = tool.execute(query="prompt caching", course_name="Computer Use", lesson_number=1)
        assert not out.startswith("Search error"), out
        assert not out.startswith("No course found"), out


# ---------- ToolManager dispatch ----------

class TestToolManager:

    def test_dispatches_to_search_tool(self, temp_vector_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(temp_vector_store))
        out = manager.execute_tool("search_course_content", query="embeddings vector space")
        assert "vector space" in out
        assert manager.get_last_sources()
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_unknown_tool(self):
        assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"

    def test_tool_definition_matches_execute_signature(self, mock_vector_store):
        definition = CourseSearchTool(mock_vector_store).get_tool_definition()
        assert definition["name"] == "search_course_content"
        assert set(definition["input_schema"]["properties"]) == {
            "query", "course_name", "lesson_number"
        }
        assert definition["input_schema"]["required"] == ["query"]

    def test_unexpected_tool_input_does_not_raise(self, temp_vector_store):
        """The model controls tool input; a bad argument must not crash the request"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(temp_vector_store))
        out = manager.execute_tool("search_course_content", query="mocks", unknown_arg=1)
        assert isinstance(out, str)

    def test_string_lesson_number_still_filters(self, temp_vector_store):
        """Tool inputs may arrive as strings; '1' should behave like 1"""
        tool = CourseSearchTool(temp_vector_store)
        out = tool.execute(query="testing", lesson_number="1")
        assert "Lesson 1" in out
