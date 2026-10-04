"""Command-line entry point for RepoPilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
    return parser


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

    print(f"Status: {state.status}")
    if state.final_answer:
        print(f"Final answer: {state.final_answer}")

    return 0 if state.status is AgentStatus.COMPLETED else 1

