import pytest

from swe_agent.executor import ToolCall
from swe_agent.model import ModelResponse, ScriptedModelClient


def test_scripted_model_returns_responses_in_order() -> None:
    first = ModelResponse(content="I will inspect the repository.")
    second = ModelResponse(
        tool_calls=(ToolCall(id="call-1", name="list_files", arguments={}),)
    )
    model = ScriptedModelClient([first, second])

    assert model.complete(messages=[], tools=[]) == first
    assert model.complete(messages=[], tools=[]) == second


def test_model_response_can_contain_text_and_multiple_tool_calls() -> None:
    response = ModelResponse(
        content="I need both files.",
        tool_calls=(
            ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
            ToolCall(id="call-2", name="read_file", arguments={"path": "b.py"}),
        ),
    )

    assert response.content == "I need both files."
    assert [call.name for call in response.tool_calls] == ["read_file", "read_file"]


def test_scripted_model_fails_clearly_when_responses_are_exhausted() -> None:
    model = ScriptedModelClient([])

    with pytest.raises(RuntimeError, match="no responses remaining"):
        model.complete(messages=[], tools=[])

