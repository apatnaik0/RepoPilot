"""Coordination logic for the software-engineering agent."""

from __future__ import annotations

from typing import Any

from swe_agent.executor import ToolCall, ToolExecutor, ToolResult
from swe_agent.model import ModelClient, ModelResponse
from swe_agent.state import AgentState, AgentStatus
from swe_agent.tools import ToolRegistry


class Agent:
    def __init__(self, model: ModelClient, registry: ToolRegistry) -> None:
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry)

    def run_step(self, state: AgentState) -> None:
        """Perform one model call and execute its requested tools."""
        if state.status is not AgentStatus.RUNNING:
            raise RuntimeError(f"Cannot step an agent with status: {state.status}")

        if state.step_count == 0 and not state.messages:
            state.messages.append({"role": "user", "content": state.task})

        response = self.model.complete(
            messages=state.messages,
            tools=self.registry.model_schemas(),
        )
        state.step_count += 1
        state.messages.append(self._assistant_message(response))

        for call in response.tool_calls:
            result = self.executor.execute(call)
            state.messages.append(self._tool_message(result))

            if call.name == "finish" and result.success:
                state.status = AgentStatus.COMPLETED
                state.final_answer = result.output
                break

    @staticmethod
    def _assistant_message(response: ModelResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [Agent._tool_call_data(call) for call in response.tool_calls],
        }

    @staticmethod
    def _tool_call_data(call: ToolCall) -> dict[str, Any]:
        return {"id": call.id, "name": call.name, "arguments": call.arguments}

    @staticmethod
    def _tool_message(result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.tool_name,
            "content": result.output if result.success else result.error,
            "success": result.success,
        }

