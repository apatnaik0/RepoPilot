"""Provider-neutral contract for language-model calls."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from swe_agent.executor import ToolCall


@dataclass(frozen=True)
class ModelResponse:
    """The assistant text and tool requests returned by a model."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ModelClient(Protocol):
    """Interface the agent loop expects from any model provider."""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


class ScriptedModelClient:
    """Return predetermined responses for deterministic agent tests."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        del messages, tools
        if not self._responses:
            raise RuntimeError("Scripted model has no responses remaining")
        return self._responses.popleft()

