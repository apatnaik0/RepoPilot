"""Command-line entry point for RepoPilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from swe_agent.agent import Agent
from swe_agent.openai_model import OpenAIModelClient
from swe_agent.state import AgentState, AgentStatus
from swe_agent.tools import RepositoryTools


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repopilot",
        description="Run RepoPilot on a software repository.",
    )
    parser.add_argument("--repo", required=True, help="Path to the target repository")
    parser.add_argument("--task", required=True, help="Coding task for the agent")
    parser.add_argument("--model", required=True, help="OpenAI model name")
    parser.add_argument("--max-steps", type=positive_integer, default=5)
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="Print model responses, tool calls, and tool results",
    )
    return parser


def print_trajectory(messages: list[dict[str, Any]]) -> None:
    step = 0
    for message in messages:
        if message["role"] == "assistant":
            step += 1
            print(f"\nStep {step}")
            if message["content"]:
                print(f"Assistant: {message['content']}")
            for call in message["tool_calls"]:
                arguments = json.dumps(call["arguments"], sort_keys=True)
                print(f"Tool call: {call['name']} {arguments}")
        elif message["role"] == "tool":
            outcome = "succeeded" if message["success"] else "failed"
            print(f"Tool result ({message['name']}, {outcome}):")
            print(message["content"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repository = Path(args.repo).expanduser().resolve()
    if not repository.is_dir():
        parser.error(f"repository is not a directory: {repository}")

    repository_tools = RepositoryTools(repository)
    registry = repository_tools.build_registry()
    model = OpenAIModelClient(model=args.model)
    state = AgentState(task=args.task)

    Agent(model, registry).run(state, max_steps=args.max_steps)

    if args.show_trajectory:
        print_trajectory(state.messages)
    print(f"Status: {state.status}")
    if state.final_answer:
        print(f"Final answer: {state.final_answer}")

    return 0 if state.status is AgentStatus.COMPLETED else 1
