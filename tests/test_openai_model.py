from types import SimpleNamespace

import pytest

from swe_agent.openai_model import OpenAIModelClient


class FakeResponses:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.request: dict | None = None

    def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return self.response


def make_client(response: SimpleNamespace) -> tuple[OpenAIModelClient, FakeResponses]:
    responses = FakeResponses(response)
    sdk_client = SimpleNamespace(responses=responses)
    return OpenAIModelClient(model="test-model", client=sdk_client), responses


def test_complete_translates_messages_and_tools() -> None:
    client, responses = make_client(
        SimpleNamespace(output=[], output_text="I am done.")
    )
    messages = [
        {"role": "user", "content": "Inspect the repository"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "name": "list_files", "arguments": {}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "list_files",
            "content": "README.md",
            "success": True,
        },
    ]
    tools = [
        {
            "name": "list_files",
            "description": "List repository files.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    result = client.complete(messages=messages, tools=tools)

    assert result.content == "I am done."
    assert responses.request == {
        "model": "test-model",
        "input": [
            {"role": "user", "content": "Inspect the repository"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "list_files",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "README.md",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "list_files",
                "description": "List repository files.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }


def test_complete_converts_openai_function_calls() -> None:
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-2",
        name="read_file",
        arguments='{"path": "README.md"}',
    )
    client, _ = make_client(
        SimpleNamespace(output=[function_call], output_text="I will read the file.")
    )

    result = client.complete(messages=[], tools=[])

    assert result.content == "I will read the file."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call-2"
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "README.md"}


def test_complete_rejects_malformed_tool_arguments() -> None:
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-3",
        name="read_file",
        arguments="not-json",
    )
    client, _ = make_client(SimpleNamespace(output=[function_call], output_text=""))

    with pytest.raises(ValueError, match="Invalid JSON arguments for tool call call-3"):
        client.complete(messages=[], tools=[])

