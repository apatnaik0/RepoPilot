"""Validation and execution boundary for model-requested tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swe_agent.tools import Tool, ToolRegistry


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by a model."""

    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class ToolResult:
    """The success or failure produced by executing one tool call."""

    call_id: str
    tool_name: str
    success: bool
    output: str
    error: str | None = None


class ToolExecutor:
    """Validate model-supplied arguments and invoke authorized tools."""

    _JSON_TYPES: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a call, returning failures as data instead of raising them."""
        try:
            tool = self.registry.get(call.name)
            arguments = self._validate_arguments(tool, call.arguments)
            output = tool.function(**arguments)
            if not isinstance(output, str):
                raise TypeError(
                    f"Tool returned {type(output).__name__}; expected str"
                )
        except Exception as error:
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                output="",
                error=f"{type(error).__name__}: {error}",
            )

        return ToolResult(
            call_id=call.id,
            tool_name=call.name,
            success=True,
            output=output,
        )

    def _validate_arguments(self, tool: Tool, arguments: Any) -> dict[str, Any]:
        """Validate the small JSON Schema subset used by the V0 tools."""
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be an object")

        schema = tool.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(
                f"Missing required argument(s): {', '.join(sorted(missing))}"
            )

        if schema.get("additionalProperties") is False:
            unexpected = set(arguments) - set(properties)
            if unexpected:
                raise ValueError(
                    "Unexpected argument(s): " + ", ".join(sorted(unexpected))
                )

        for name, value in arguments.items():
            property_schema = properties.get(name)
            if property_schema is None:
                continue
            expected_json_type = property_schema.get("type")
            if expected_json_type is None:
                continue
            self._validate_type(name, value, expected_json_type)

        return arguments

    def _validate_type(self, name: str, value: Any, expected_json_type: str) -> None:
        expected_python_type = self._JSON_TYPES.get(expected_json_type)
        if expected_python_type is None:
            raise ValueError(f"Unsupported schema type: {expected_json_type}")

        valid = isinstance(value, expected_python_type)
        if expected_json_type in {"integer", "number"} and isinstance(value, bool):
            valid = False

        if not valid:
            raise TypeError(
                f"Argument '{name}' must be {expected_json_type}; "
                f"received {type(value).__name__}"
            )

