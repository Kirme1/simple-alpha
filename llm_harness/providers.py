"""LLM provider adapters for the agent harness.

The harness keeps its conversation in a provider-neutral form:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}], "raw": <native>}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Each provider converts that history to its own wire format. ``raw`` holds the provider's
own assistant message so it can be replayed unchanged (Anthropic thinking blocks need that).
Tools are passed as ``{"name", "description", "input_schema"}`` (the MCP shape).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall]
    raw: Any
    request: dict
    usage: dict | None = None
    extra: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    async def complete(self, system: str, history: list[dict], tools: list[dict]) -> Completion: ...


def assistant_entry(completion: Completion) -> dict:
    return {
        "role": "assistant",
        "content": completion.text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in completion.tool_calls
        ],
        "raw": completion.raw,
    }


class OpenAIProvider:
    """Any OpenAI-compatible chat completions endpoint: llama-server, OpenAI, and similar."""

    def __init__(self, base_url: str | None, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @staticmethod
    def _convert(system: str, history: list[dict]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for entry in history:
            if entry["role"] == "assistant":
                message: dict = {"role": "assistant", "content": entry["content"] or None}
                if entry["tool_calls"]:
                    message["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in entry["tool_calls"]
                    ]
                messages.append(message)
            elif entry["role"] == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": entry["tool_call_id"],
                    "content": entry["content"],
                })
            else:
                messages.append({"role": entry["role"], "content": entry["content"]})
        return messages

    async def complete(self, system: str, history: list[dict], tools: list[dict]) -> Completion:
        request = {
            "model": self._model,
            "messages": self._convert(system, history),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ],
            "tool_choice": "auto",
        }
        response = await self._client.chat.completions.create(**request)
        message = response.choices[0].message

        tool_calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(call.id, call.function.name, arguments))

        return Completion(
            text=message.content or "",
            tool_calls=tool_calls,
            raw=message.model_dump(exclude_none=True),
            request=request,
            usage=response.usage.model_dump(exclude_none=True) if response.usage else None,
        )


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, max_tokens: int = 16000) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _convert(history: list[dict]) -> list[dict]:
        messages: list[dict] = []
        for entry in history:
            if entry["role"] == "assistant":
                if entry.get("raw"):
                    # Replay the model's own blocks (including thinking) unchanged.
                    content = entry["raw"]
                else:
                    content = []
                    if entry["content"]:
                        content.append({"type": "text", "text": entry["content"]})
                    for call in entry["tool_calls"]:
                        content.append({
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call["arguments"],
                        })
                messages.append({"role": "assistant", "content": content})
            elif entry["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": entry["tool_call_id"],
                    "content": entry["content"],
                }
                # All results for one assistant turn go in a single user message.
                last = messages[-1] if messages else None
                if (
                    last
                    and last["role"] == "user"
                    and isinstance(last["content"], list)
                    and last["content"]
                    and last["content"][0].get("type") == "tool_result"
                ):
                    last["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
            else:
                messages.append({"role": "user", "content": entry["content"]})
        return messages

    async def complete(self, system: str, history: list[dict], tools: list[dict]) -> Completion:
        request = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": self._convert(history),
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in tools
            ],
        }
        response = await self._client.messages.create(**request)

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = [
            ToolCall(block.id, block.name, block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        if response.stop_reason == "refusal" and not text:
            text = "The model declined to answer this request."
        elif response.stop_reason == "max_tokens" and not tool_calls:
            text = (text + "\n\n" if text else "") + "(The response was cut off at the token limit.)"

        return Completion(
            text=text,
            tool_calls=tool_calls,
            raw=[block.model_dump(exclude_none=True) for block in response.content],
            request=request,
            usage=response.usage.model_dump(exclude_none=True) if response.usage else None,
        )
