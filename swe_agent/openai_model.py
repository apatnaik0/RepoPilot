"""OpenAI Responses API adapter for RepoPilot's model interface."""

from __future__ import annotations

import json
from typing import Any

from swe_agent.executor import ToolCall
from swe_agent.model import ModelResponse


class OpenAIModelClient:
    def __init__(self, model: str, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.model = model
        self.client = client

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        response = self.client.responses.create(
            model=self.model,
            input=self._convert_messages(messages),
            tools=[{"type": "function", **tool} for tool in tools],
        )

        tool_calls = tuple(
            self._convert_tool_call(item)
            for item in response.output
            if item.type == "function_call"
        )
        return ModelResponse(content=response.output_text or "", tool_calls=tool_calls)

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message["content"],
                    }
                )
                continue

            content = message.get("content", "")
            if content:
                converted.append({"role": role, "content": content})

            for call in message.get("tool_calls", []):
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    }
                )

        return converted

    @staticmethod
    def _convert_tool_call(item: Any) -> ToolCall:
        try:
            arguments = json.loads(item.arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON arguments for tool call {item.call_id}"
            ) from error

        return ToolCall(id=item.call_id, name=item.name, arguments=arguments)

