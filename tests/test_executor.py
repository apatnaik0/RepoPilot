from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swe_agent.executor import ToolCall, ToolExecutor
from swe_agent.tools import RepositoryTools, Tool, ToolRegistry


@pytest.fixture
def executor(tmp_path: Path) -> ToolExecutor:
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    registry = RepositoryTools(tmp_path).build_registry()
    return ToolExecutor(registry)


def test_executes_known_tool(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
    )

    assert result.call_id == "call-1"
    assert result.tool_name == "read_file"
    assert result.success is True
    assert result.output == "# Example\n"
    assert result.error is None


def test_default_arguments_are_applied_by_tool(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="call-2", name="list_files", arguments={})
    )

    assert result.success is True
    assert result.output == "README.md"


def test_unknown_tool_becomes_failed_result(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="call-3", name="delete_file", arguments={})
    )

    assert result.success is False
    assert result.output == ""
    assert result.error == "KeyError: 'Unknown tool: delete_file'"


def test_arguments_must_be_an_object(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="call-4", name="read_file", arguments='{"path": "README.md"}')
    )

    assert result.success is False
    assert result.error == "TypeError: Tool arguments must be an object"


def test_missing_required_argument_becomes_failed_result(
    executor: ToolExecutor,
) -> None:
    result = executor.execute(ToolCall(id="call-5", name="read_file", arguments={}))

    assert result.success is False
    assert result.error == "ValueError: Missing required argument(s): path"


def test_unexpected_argument_becomes_failed_result(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(
            id="call-6",
            name="read_file",
            arguments={"path": "README.md", "encoding": "utf-16"},
        )
    )

    assert result.success is False
    assert result.error == "ValueError: Unexpected argument(s): encoding"


def test_incorrect_argument_type_becomes_failed_result(
    executor: ToolExecutor,
) -> None:
    result = executor.execute(
        ToolCall(id="call-7", name="read_file", arguments={"path": 42})
    )

    assert result.success is False
    assert result.error == (
        "TypeError: Argument 'path' must be string; received int"
    )


def test_tool_exception_becomes_failed_result(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="call-8", name="read_file", arguments={"path": "missing.py"})
    )

    assert result.success is False
    assert result.error == "FileNotFoundError: File does not exist: missing.py"


def test_tool_call_identity_is_preserved_on_failure(executor: ToolExecutor) -> None:
    result = executor.execute(
        ToolCall(id="model-generated-id", name="read_file", arguments={})
    )

    assert result.call_id == "model-generated-id"
    assert result.tool_name == "read_file"


@pytest.mark.parametrize(
    ("schema_type", "valid_value", "invalid_value"),
    [
        ("integer", 3, True),
        ("number", 2.5, False),
        ("boolean", True, 1),
        ("object", {"key": "value"}, []),
        ("array", [1, 2], {"0": 1}),
        ("null", None, "none"),
    ],
)
def test_basic_json_types_are_validated(
    schema_type: str, valid_value: Any, invalid_value: Any
) -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="typed_tool",
            description="Test type validation.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": schema_type}},
                "required": ["value"],
                "additionalProperties": False,
            },
            function=lambda value: str(value),
        )
    )
    executor = ToolExecutor(registry)

    successful_result = executor.execute(
        ToolCall(id="valid", name="typed_tool", arguments={"value": valid_value})
    )
    failed_result = executor.execute(
        ToolCall(id="invalid", name="typed_tool", arguments={"value": invalid_value})
    )

    assert successful_result.success is True
    assert failed_result.success is False
    assert "TypeError" in failed_result.error


def test_non_string_tool_output_becomes_failed_result() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="bad_output",
            description="Return an invalid output type.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            function=lambda: 123,  # type: ignore[return-value]
        )
    )

    result = ToolExecutor(registry).execute(
        ToolCall(id="call-9", name="bad_output", arguments={})
    )

    assert result.success is False
    assert result.error == "TypeError: Tool returned int; expected str"


def test_unsupported_schema_type_becomes_failed_result() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="unsupported",
            description="Use an unsupported schema type.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "mystery"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            function=lambda value: str(value),
        )
    )

    result = ToolExecutor(registry).execute(
        ToolCall(id="call-10", name="unsupported", arguments={"value": "x"})
    )

    assert result.success is False
    assert result.error == "ValueError: Unsupported schema type: mystery"

