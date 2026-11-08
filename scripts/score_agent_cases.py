"""Check each agent run against its case in data/agent_cases.json.

usage: python -m scripts.score_agent_cases [run_file]     (default: the newest docs/phase-notes/phase4-agent-run-*.json)

A run passes when its final route, reasons, mismatches and tool use all match what the case expects. This script lists the
problems per run and a few averages. Aggregate tables, the invariants over many runs and the attack scoring come later.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "data" / "agent_cases.json"
NOTES = ROOT / "docs" / "phase-notes"


def load_cases() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(CASES_FILE.read_text(encoding="utf-8"))}


def check_run(case: dict, run: dict) -> list[str]:
    """The ways this run differs from what the case expects. An empty list means it passed."""
    if "error" in run:
        return [f"run failed: {run['error']}"]
    problems = []
    if run["final_route"] != case["expected_route"]:
        problems.append(f"route {run['final_route']}, expected {case['expected_route']}")
    for reason in case["reasons_include"]:
        if reason not in run["reasons"]:
            problems.append(f"missing reason {reason}")
    for reason in case["reasons_exclude"]:
        if reason in run["reasons"]:
            problems.append(f"unexpected reason {reason}")

    expected = sorted((m["document"], m["field"]) for m in case["expected_mismatches"])
    found = sorted((m["document"], m["field"]) for m in run["mismatches"])
    if found != expected:
        problems.append(f"mismatches {found}, expected {expected}")

    if case["agent_runs"]:
        agent = run["agent"]
        if agent is None:
            problems.append("the agent did not run")
        else:
            if agent["status"] != "finished":
                problems.append(f"agent status {agent['status']}")
            if agent["steps"] > case["max_steps"]:
                problems.append(f"{agent['steps']} steps, limit {case['max_steps']}")
    elif run["agent"] is not None:
        problems.append("the agent ran although no documents were attached")

    called = set(run["tools_called"])
    for tool in case["must_call"]:
        if tool not in called:
            problems.append(f"did not call {tool}")
    for tool in case["must_not_call"]:
        if tool in called:
            problems.append(f"called {tool}")
    return problems


def newest_run_file() -> Path:
    files = sorted(NOTES.glob("phase4-agent-run-*.json"))
    if not files:
        sys.exit("No agent run found. Run: python -m scripts.run_agent_cases")
    return files[-1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_run_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = load_cases()
    results = data["results"]
    print(f"{path.name}: {len(results)} runs, config {data['config']}")

    passed = 0
    for run in results:
        problems = check_run(cases[run["case"]], run)
        passed += not problems
        print(f"  {run['case']} run {run['run']}: {'PASS' if not problems else 'FAIL'}" + "".join(f"\n      - {p}" for p in problems))
    print(f"{passed}/{len(results)} runs passed")

    done = [r for r in results if r.get("agent")]
    if done:
        n = len(done)
        print(f"agent runs: {n}; average {sum(r['agent']['steps'] for r in done) / n:.1f} steps, "
              f"{sum(r['agent']['tool_calls'] for r in done) / n:.1f} tool calls, "
              f"{sum(r['agent']['input_tokens'] + r['agent']['output_tokens'] for r in done) / n:.0f} tokens, "
              f"{sum(r['agent']['elapsed_ms'] for r in done) / n / 1000:.1f} s")


if __name__ == "__main__":
    main()
