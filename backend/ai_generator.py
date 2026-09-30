import anthropic
from typing import List, Optional, Dict, Any, Tuple

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum sequential tool-calling rounds per user query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content or detailed educational information
- **get_course_outline**: Get a course's outline (course title, course link, and complete lesson list)

Tool Usage:
- Use **get_course_outline** for questions about a course's outline, structure, syllabus, or list of lessons
- Use **search_course_content** **only** for questions about specific course content or detailed educational materials
- You may make **up to 2 sequential tool calls** per query when a question needs information from one tool before using another (e.g., get a course outline to find a lesson title, then search course content for that topic)
- Use a second tool call only if the first result is insufficient; otherwise answer immediately
- If a tool returns an error, answer with what you have and state briefly that the information could not be retrieved
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Responses:
- When answering an outline-related query, always include:
 - The course title
 - The course link
 - Every lesson, each with its lesson number and lesson title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use the outline tool, then answer
- **Course-specific content questions**: Search first, then answer
- **Multi-step questions**: Use the tools in sequence, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    @staticmethod
    def _extract_text(response) -> str:
        """Get the text content from a response, skipping thinking blocks"""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        # Thinking disabled: this app needs short, direct answers and the
        # small max_tokens budget would otherwise be consumed by reasoning,
        # leaving no room for the actual response text.
        self.base_params = {
            "model": self.model,
            "max_tokens": 800,
            "thinking": {"type": "disabled"}
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        messages = [{"role": "user", "content": query}]

        # Tools stay available for every tool round
        tool_params = {"tools": tools, "tool_choice": {"type": "auto"}} if tools else {}

        for _ in range(self.MAX_TOOL_ROUNDS):
            # Pass a copy so each request keeps the messages it was sent with
            response = self.client.messages.create(
                **self.base_params,
                messages=list(messages),
                system=system_content,
                **tool_params
            )

            # Claude answered directly, or there is nothing to run tools with
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Keep Claude's tool request and the results in context for the next round
            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            if failed:
                break

        # Max rounds reached or a tool failed: final answer without tools
        final_response = self.client.messages.create(
            **self.base_params,
            messages=list(messages),
            system=system_content
        )
        return self._extract_text(final_response)

    @staticmethod
    def _execute_tools(response, tool_manager) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool call in a response.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            (tool_result blocks, whether any tool call failed)
        """
        tool_results = []
        failed = False
        for content_block in response.content:
            if content_block.type != "tool_use":
                continue
            try:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                })
            except Exception as e:
                # Every tool_use needs a tool_result, so report the error to Claude
                failed = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": f"Tool error: {e}",
                    "is_error": True
                })
        return tool_results, failed