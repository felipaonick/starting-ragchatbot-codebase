"""Tests that AIGenerator calls the CourseSearchTool correctly"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from ai_generator import AIGenerator
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from tests.conftest import make_text_response, make_tool_use_response

MODEL = "claude-sonnet-5"


@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model=MODEL)
    gen.client = MagicMock()
    return gen


@pytest.fixture
def tool_manager(temp_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(temp_vector_store))
    manager.register_tool(CourseOutlineTool(temp_vector_store))
    return manager


def call_kwargs(generator, index):
    return generator.client.messages.create.call_args_list[index].kwargs


# ---------- first request ----------


class TestFirstRequest:

    def test_sends_tools_with_auto_choice(self, generator, tool_manager):
        generator.client.messages.create.return_value = make_text_response("hi")
        tools = tool_manager.get_tool_definitions()
        generator.generate_response("q", tools=tools, tool_manager=tool_manager)

        kwargs = call_kwargs(generator, 0)
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["model"] == MODEL
        assert "temperature" not in kwargs  # rejected by Sonnet 5

    def test_includes_history_in_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_text_response("hi")
        generator.generate_response("q", conversation_history="User: earlier")
        assert "User: earlier" in call_kwargs(generator, 0)["system"]

    def test_direct_answer_skips_tools(self, generator):
        generator.client.messages.create.return_value = make_text_response("Paris")
        manager = MagicMock()
        out = generator.generate_response(
            "capital of France?", tools=[{}], tool_manager=manager
        )
        assert out == "Paris"
        manager.execute_tool.assert_not_called()
        assert generator.client.messages.create.call_count == 1


# ---------- tool round trip ----------


class TestToolExecution:

    def _run(self, generator, manager, tool_input=None, preamble=None):
        tool_input = tool_input or {"query": "mock objects", "course_name": "RAG"}
        generator.client.messages.create.side_effect = [
            make_tool_use_response(
                "search_course_content", tool_input, "toolu_1", preamble
            ),
            make_text_response("Mocks replace dependencies."),
        ]
        return generator.generate_response(
            "What is a mock?",
            tools=[{"name": "search_course_content"}],
            tool_manager=manager,
        )

    def test_executes_search_tool_with_model_input(self, generator):
        manager = MagicMock()
        manager.execute_tool.return_value = "search output"
        out = self._run(generator, manager)

        manager.execute_tool.assert_called_once_with(
            "search_course_content", query="mock objects", course_name="RAG"
        )
        assert out == "Mocks replace dependencies."
        assert generator.client.messages.create.call_count == 2
        assert "tools" in call_kwargs(generator, 1)  # round 2 may still use a tool

    def test_second_request_carries_tool_result(self, generator):
        manager = MagicMock()
        manager.execute_tool.return_value = "search output"
        self._run(generator, manager, preamble="Let me search.")

        messages = call_kwargs(generator, 1)["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[0]["content"] == "What is a mock?"
        assert any(b.type == "tool_use" for b in messages[1]["content"])
        assert messages[2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "search output",
            }
        ]

    def test_final_answer_after_thinking_block(self, generator):
        """Sonnet 5 can put a thinking block before the text; the text must still be returned.
        Regression: `response.content[0].text` raised AttributeError -> HTTP 500 -> 'Query failed'
        """
        manager = MagicMock()
        manager.execute_tool.return_value = "search output"
        final = make_text_response("Mocks replace dependencies.")
        final.content.insert(
            0, SimpleNamespace(type="thinking", thinking="", signature="sig")
        )
        generator.client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "x"}),
            final,
        ]
        out = generator.generate_response("q", tools=[{}], tool_manager=manager)
        assert out == "Mocks replace dependencies."

    def test_thinking_disabled_in_requests(self, generator):
        """Keeps the 800-token budget for the answer instead of reasoning"""
        generator.client.messages.create.return_value = make_text_response("hi")
        generator.generate_response("q")
        assert call_kwargs(generator, 0)["thinking"] == {"type": "disabled"}

    def test_real_search_output_reaches_claude(self, generator, tool_manager):
        self._run(generator, tool_manager, {"query": "What is a mock object?"})
        tool_result = call_kwargs(generator, 1)["messages"][2]["content"][0]["content"]
        assert "mock object replaces a real dependency" in tool_result
        assert tool_manager.get_last_sources()

    def test_tool_exception_does_not_crash(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = RuntimeError("chroma exploded")
        out = self._run(generator, manager)

        assert out == "Mocks replace dependencies."
        assert generator.client.messages.create.call_count == 2
        final = call_kwargs(generator, 1)
        assert "tools" not in final
        result = final["messages"][-1]["content"][0]
        assert result["is_error"] is True
        assert "chroma exploded" in result["content"]

    def test_tool_use_without_manager_returns_text(self, generator):
        generator.client.messages.create.return_value = make_tool_use_response(
            "search_course_content", {"query": "x"}, preamble="Searching"
        )
        assert generator.generate_response("q", tools=[{}]) == "Searching"


# ---------- sequential tool rounds ----------


class TestSequentialTools:

    QUERY = "Find a course on the same topic as lesson 4 of course X"
    TOOLS = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

    def _outline(self):
        return make_tool_use_response(
            "get_course_outline", {"course_title": "X"}, "toolu_1"
        )

    def _search(self):
        return make_tool_use_response(
            "search_course_content", {"query": "Lesson 4 title"}, "toolu_2"
        )

    def _run(self, generator, manager, responses):
        generator.client.messages.create.side_effect = responses
        return generator.generate_response(
            self.QUERY, tools=self.TOOLS, tool_manager=manager
        )

    def test_two_rounds_then_final_answer(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = ["outline output", "search output"]
        out = self._run(
            generator,
            manager,
            [
                self._outline(),
                self._search(),
                make_text_response("Course Y covers it."),
            ],
        )

        assert manager.execute_tool.call_args_list == [
            call("get_course_outline", course_title="X"),
            call("search_course_content", query="Lesson 4 title"),
        ]
        assert generator.client.messages.create.call_count == 3
        assert out == "Course Y covers it."

    def test_second_round_sees_first_result(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = ["outline output", "search output"]
        self._run(
            generator,
            manager,
            [
                self._outline(),
                self._search(),
                make_text_response("done"),
            ],
        )

        second = call_kwargs(generator, 1)
        assert second["tools"] == self.TOOLS
        assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
        assert second["messages"][2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "outline output",
            }
        ]

    def test_final_call_after_max_rounds_omits_tools(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = ["outline output", "search output"]
        self._run(
            generator,
            manager,
            [
                self._outline(),
                self._search(),
                make_text_response("done"),
            ],
        )

        final = call_kwargs(generator, 2)
        assert "tools" not in final
        assert "tool_choice" not in final
        messages = final["messages"]
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert [m["content"][0]["tool_use_id"] for m in messages[2::2]] == [
            "toolu_1",
            "toolu_2",
        ]

    def test_stops_after_one_round_when_claude_answers(self, generator):
        manager = MagicMock()
        manager.execute_tool.return_value = "outline output"
        out = self._run(
            generator, manager, [self._outline(), make_text_response("Lesson 4 is X.")]
        )

        assert manager.execute_tool.call_count == 1
        assert generator.client.messages.create.call_count == 2
        assert out == "Lesson 4 is X."

    def test_error_in_first_round_skips_second_round(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = RuntimeError("boom")
        out = self._run(
            generator,
            manager,
            [
                self._outline(),
                make_text_response("Could not retrieve the outline."),
            ],
        )

        assert manager.execute_tool.call_count == 1
        assert generator.client.messages.create.call_count == 2
        assert "tools" not in call_kwargs(generator, 1)
        assert out == "Could not retrieve the outline."

    def test_error_in_second_round_still_answers(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = ["outline output", RuntimeError("boom")]
        out = self._run(
            generator,
            manager,
            [
                self._outline(),
                self._search(),
                make_text_response("Partial answer."),
            ],
        )

        assert generator.client.messages.create.call_count == 3
        last_result = call_kwargs(generator, 2)["messages"][-1]["content"][0]
        assert last_result["tool_use_id"] == "toolu_2"
        assert last_result["is_error"] is True
        assert out == "Partial answer."

    def test_parallel_tool_uses_in_one_round(self, generator):
        manager = MagicMock()
        manager.execute_tool.side_effect = ["result a", "result b"]
        both = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="search_course_content",
                    input={"query": "a"},
                    id="toolu_a",
                ),
                SimpleNamespace(
                    type="tool_use",
                    name="search_course_content",
                    input={"query": "b"},
                    id="toolu_b",
                ),
            ],
        )
        out = self._run(generator, manager, [both, make_text_response("compared")])

        assert manager.execute_tool.call_count == 2
        results = call_kwargs(generator, 1)["messages"][2]["content"]
        assert [(r["tool_use_id"], r["content"]) for r in results] == [
            ("toolu_a", "result a"),
            ("toolu_b", "result b"),
        ]
        assert out == "compared"


# ---------- live: real Anthropic API ----------


@pytest.mark.live
class TestLive:

    def test_content_question_uses_search_tool(self, api_key, tool_manager):
        gen = AIGenerator(api_key, MODEL)
        calls = []
        original = tool_manager.execute_tool

        def spy(name, **kwargs):
            calls.append((name, kwargs))
            return original(name, **kwargs)

        tool_manager.execute_tool = spy
        out = gen.generate_response(
            "Answer this question about course materials: In the 'Testing RAG Systems "
            "with Pytest' course, how is retrieval quality measured?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )
        assert (
            calls and calls[0][0] == "search_course_content"
        ), f"no search made; answer: {out!r}"
        assert out.strip()
        assert "recall" in out.lower()

    def test_answer_has_no_leaked_tool_call_text(self, api_key, tool_manager):
        gen = AIGenerator(api_key, MODEL)
        out = gen.generate_response(
            "Answer this question about course materials: What does lesson 1 of the "
            "RAG testing course say about fixtures?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )
        for marker in ("search_course_content", "<thinking", "<function", "tool_use"):
            assert marker not in out, f"leaked {marker!r} in answer: {out!r}"

    def test_outline_then_search_in_sequence(self, api_key, tool_manager):
        """Model-dependent: Claude should look up the lesson title before searching"""
        gen = AIGenerator(api_key, MODEL)
        calls = []
        original = tool_manager.execute_tool

        def spy(name, **kwargs):
            calls.append(name)
            return original(name, **kwargs)

        tool_manager.execute_tool = spy
        out = gen.generate_response(
            "Answer this question about course materials: Find content in the course "
            "materials that discusses the same topic as lesson 2 of the 'Testing RAG "
            "Systems with Pytest' course.",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )
        assert calls[:2] == [
            "get_course_outline",
            "search_course_content",
        ], f"tool calls: {calls}; answer: {out!r}"
        assert out.strip()
