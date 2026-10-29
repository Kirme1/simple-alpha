"""Tool-use test runner: replays the test case prompts against one or more models and grades the tool calls.

Run from the repo root (MCP server and, for the local model, llama-server must be up):
    python -m llm_harness.testing.test [suite] [--models local anthropic:claude-sonnet-5-5 ...]

``suite`` is a module in this package (``test_cases`` or ``test_cases_multi_prompt``). ``--models all``
runs every model whose API key is set. Traces go to ``traces/<model>/`` and the grades to ``results.json``.
"""

import argparse
import asyncio
import importlib
import json
from collections import Counter
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from tqdm import tqdm

from ..config import LLM_URL, MCP_URL
from ..harness import AgentHarness
from ..models import LOCAL_ID, ModelOption, available_models, build_provider

TESTING_DIR = Path(__file__).resolve().parent
TRACES_DIR = TESTING_DIR / "traces"
RESULTS_PATH = TESTING_DIR / "results.json"


def load_test_cases(suite: str) -> dict:
    return importlib.import_module(f"{__package__}.{suite}").test_cases


def grade_args(expected_args: dict, actual_args: dict) -> tuple[bool, dict[str, bool]]:
    per_field = {}
    for field, spec in expected_args.items():
        try:
            per_field[field] = bool(spec.validator(actual_args.get(field)))
        except Exception:  # a missing argument can make a validator raise
            per_field[field] = False
    return all(per_field.values()), per_field


def new_tool_calls(harness: AgentHarness, start: int) -> list[dict]:
    """Tool calls the model made in the conversation entries added since ``start``."""
    return [call for entry in harness.conversation[start:] for call in entry.get("tool_calls", [])]


def grade_prompt(case_id: str, prompt: dict, actual: list[dict], stats: Counter, error_log: list[str]) -> None:
    for expected in prompt["expected_tool_calls"]:
        matching = [call for call in actual if call["name"] == expected["name"]]
        if not matching:
            stats["incorrect_tool_calls"] += 1
            error_log.append(f"model did not call {expected['name']} in test case {case_id}")
            continue
        stats["correct_tool_calls"] += 1
        for call in matching:
            passed, per_field = grade_args(expected["args"], call["arguments"])
            if passed:
                stats["correct_args"] += 1
            else:
                stats["incorrect_args"] += 1
                failed = [field for field, ok in per_field.items() if not ok]
                error_log.append(
                    f"test case {case_id} used incorrect arguments for {failed} of {expected['name']}\n"
                    f" Arguments used: {call['arguments']}"
                )

    category = prompt["category"]
    expected_count = {"no_tool": "0", "single_tool": "1", "multi_tool": "multiple"}[category]
    count_ok = {
        "no_tool": len(actual) == 0,
        "single_tool": len(actual) == 1,
        "multi_tool": len(actual) > 1,
    }[category]
    stats[f"{'correct' if count_ok else 'incorrect'}_{category}_call"] += 1
    if not count_ok:
        error_log.append(f"model used {len(actual)} tool calls when {expected_count} were expected for test case {case_id}")


async def run_model(option: ModelOption, test_cases: dict, mcp_session: ClientSession) -> dict:
    harness = AgentHarness(build_provider(option, LLM_URL))
    stats: Counter = Counter()
    error_log: list[str] = []
    trace_dir = TRACES_DIR / option.id.replace(":", "_").replace("/", "_")
    trace_dir.mkdir(parents=True, exist_ok=True)

    for case in tqdm(test_cases["tests"], desc=option.id):
        harness.clear_memory()
        failed = False
        for prompt in case["prompts"]:
            start = len(harness.conversation)
            try:
                await harness.query(mcp_session, prompt["prompt"])
            except Exception as exc:
                error_log.append(f"test case {case['id']} raised {type(exc).__name__}: {exc}")
                failed = True
                break
            grade_prompt(case["id"], prompt, new_tool_calls(harness, start), stats, error_log)

        (trace_dir / f"{case['id']}-traces.json").write_text(json.dumps(harness.messages, indent=4, default=str))
        if failed:
            continue
        if harness.last_attempt_count == case["ideal_attempts"]:
            stats["correct_ideal_attempts"] += 1
        else:
            stats["incorrect_ideal_attempts"] += 1
            error_log.append(
                f"model used {harness.last_attempt_count} attempts when {case['ideal_attempts']} "
                f"should have been enough for test case {case['id']}"
            )

    return {"stats": dict(stats), "error_log": error_log}


def success_rate(stats: dict, correct: str, incorrect: str) -> str:
    total = stats.get(correct, 0) + stats.get(incorrect, 0)
    return f"{stats.get(correct, 0) / total * 100:.1f}%" if total else "n/a"


def select_models(requested: list[str]) -> list[ModelOption]:
    options = {m.id: m for m in available_models()}
    if requested == ["all"]:
        return list(options.values())
    unknown = [model_id for model_id in requested if model_id not in options]
    if unknown:
        raise SystemExit(f"Unavailable models: {unknown}. Choose from: {', '.join(options)}")
    return [options[model_id] for model_id in requested]


async def main(suite: str, model_ids: list[str]) -> None:
    models = select_models(model_ids)
    test_cases = load_test_cases(suite)
    results = {}
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            for option in models:
                results[option.id] = await run_model(option, test_cases, mcp_session)

    RESULTS_PATH.write_text(json.dumps(results, indent=4))
    for model_id, result in results.items():
        stats = result["stats"]
        print(
            f"{model_id}: {success_rate(stats, 'correct_tool_calls', 'incorrect_tool_calls')} tool calls, "
            f"{success_rate(stats, 'correct_args', 'incorrect_args')} arguments"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade a model's tool use against a test case suite.")
    parser.add_argument("suite", nargs="?", default="test_cases", choices=["test_cases", "test_cases_multi_prompt"])
    parser.add_argument("--models", nargs="+", default=[LOCAL_ID], help="Model ids, or 'all' for every available model")
    args = parser.parse_args()
    asyncio.run(main(args.suite, args.models))
