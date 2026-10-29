import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import LLM_URL, MCP_URL
from .models import LOCAL_ID, available_models, build_provider
from .prompts import SYSTEM_PROMPT
from .providers import LLMProvider, assistant_entry

# Chart payloads from the MCP server travel as embedded JSON resources. They go to the browser
# only, never to the LLM. Keep in sync with mcp_server/tools/charts.py.
CHART_MIME = "application/vnd.simple-alpha.chart+json"


def _chart_payloads(content: list[dict]) -> list[dict]:
    charts = []
    for block in content:
        if block.get("type") != "resource":
            continue
        resource = block.get("resource", {})
        if (resource.get("mimeType") or resource.get("mime_type")) != CHART_MIME:
            continue
        try:
            charts.append(json.loads(resource.get("text", "")))
        except json.JSONDecodeError:
            continue
    return charts


class AgentHarness:
    def __init__(self, provider: LLMProvider) -> None:
        self.tools = []
        self.messages = []
        self.conversation = []
        self.last_attempt_count = 0
        self._llm = provider

    async def query(self, client: ClientSession, user_query: str, on_event=None) -> str:
        # on_event: optional async callback receiving progress dicts (tool calls, images)

        async def emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        self.last_attempt_count = 0

        if self.tools == []:
            mcp_tools = (await client.list_tools()).tools
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.input_schema,
                }
                for tool in mcp_tools
            ]

        # self.conversation is provider-neutral (see providers.py); each provider converts it
        # to its own wire format. self.messages is a separate trace for debugging/tests: it also
        # keeps model reasoning and the full tool results.
        self.conversation.append({"role": "user", "content": user_query})

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.conversation]

        for attempt in range(1, 9):
            completion = await self._llm.complete(SYSTEM_PROMPT, self.conversation, self.tools)

            self.last_attempt_count = attempt
            self.messages.append({"role": "assistant", "raw": completion.raw, "model_attempt": attempt})
            self.conversation.append(assistant_entry(completion))

            if not completion.tool_calls:
                return completion.text

            for tool_call in completion.tool_calls:
                arguments = tool_call.arguments
                await emit({
                    "type": "tool_start",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": arguments,
                })
                result = await client.call_tool(
                    tool_call.name,
                    arguments,
                )

                # Separate image content from text content
                tool_payload = result.model_dump(exclude_none=True)
                text_blocks = [
                    block["text"]
                    for block in tool_payload.get("content", [])
                    if block.get("type") == "text"
                ]
                image_count = sum(
                    block.get("type") == "image"
                    for block in tool_payload.get("content", [])
                )
                charts = _chart_payloads(tool_payload.get("content", []))

                await emit({
                    "type": "tool_end",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "text": text_blocks,
                    "is_error": bool(tool_payload.get("isError", False)),
                })
                for block in tool_payload.get("content", []):
                    if block.get("type") == "image":
                        await emit({
                            "type": "image",
                            "tool": tool_call.name,
                            "arguments": arguments,
                            "mime": block.get("mimeType", "image/png"),
                            "data": block.get("data", ""),
                        })

                for chart in charts:
                    await emit({
                        "type": "chart",
                        "tool": tool_call.name,
                        "arguments": arguments,
                        "chart": chart,
                    })

                # Images and charts are not sent to the LLM, only the text and counts.
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps({
                        "text": text_blocks,
                        "images_omitted": image_count,
                        "charts_displayed": len(charts),
                    }),
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.model_dump_json(exclude_none=True),
                    "model_attempt": attempt,
                })

        return "Unable to complete the request after multiple attempts. Please try again later."

    def clear_memory(self) -> None:
        """Start a fresh conversation; the cached tool list is kept."""
        self.messages = []
        self.conversation = []
        self.last_attempt_count = 0


async def chat_loop(agent_harness: AgentHarness, client: ClientSession) -> None:
    print("Welcome to the Agent Harness! Type 'exit' or 'quit' to quit.")
    while True:
        try:
            user_query = (await asyncio.to_thread(input, "\nQuery: ")).strip()
        except EOFError:
            print("\nExiting...")
            break

        if user_query.lower() == "exit" or user_query.lower() == "quit":
            print("Exiting...")
            break

        try:
            response = await agent_harness.query(client, user_query)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"\nError: {e}")


async def main(model_id: str) -> None:
    options = {m.id: m for m in available_models()}
    if model_id not in options:
        raise SystemExit(f"Model '{model_id}' is not available. Choose from: {', '.join(options)}")
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            agent_harness = AgentHarness(build_provider(options[model_id], LLM_URL))

            await chat_loop(agent_harness, mcp_session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the agent harness in the terminal.")
    parser.add_argument("--model", default=LOCAL_ID, help="Model id, e.g. local or anthropic:claude-sonnet-5-5")
    asyncio.run(main(parser.parse_args().model))
