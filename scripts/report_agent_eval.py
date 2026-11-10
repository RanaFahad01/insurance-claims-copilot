"""Turn recorded runs into the evaluation tables.

usage: python -m scripts.report_agent_eval --trajectory RUNFILE [--attacks FILE FILE ...] [--reader FILE] [--labels FILE]
                                          [--price-in USD_PER_MTOK --price-out USD_PER_MTOK] [--out report.md]

Reads files written by scripts.run_agent_cases (and scripts.check_invoices --attack). It calls no model. Every rate is shown as
a count with a 95% interval, because the sets are small. Without --out the report is printed.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.eval_metrics import attack_outcome, trajectory_summary, wilson
from scripts.score_agent_cases import check_run

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NOTES = ROOT / "docs" / "phase-notes"


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0 runs)"
    low, high = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% CI {low:.0%}-{high:.0%})"


def seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def stat_row(label: str, stats: dict, fmt=lambda v: f"{v:,.0f}") -> str:
    if not stats["n"]:
        return f"| {label} | n/a | | | |"
    return f"| {label} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p95'])} | {fmt(stats['max'])} |"


def frozen_consistent(configs: list[dict]) -> bool:
    """True when every file records the same code, prompt, cases and documents (per kind of set)."""
    return len({json.dumps(c.get("frozen"), sort_keys=True) for c in configs}) <= 1


# ---- the trajectory report

def render_trajectory(summary: dict, cases: dict[str, dict], config: dict) -> str:
    wd, nd = summary["with_documents"], summary["without_documents"]
    out = [
        "## Trajectory cases (held-out test split)", "",
        f"{summary['runs']} runs of {len(cases)} cases, {config.get('runs', '?')} per case, engine `{config.get('engine', 'raw')}`, agent model "
        f"`{config.get('agent_model')}`, prompt `{config.get('agent_prompt')}`. Runs that ended in an error: {summary['errors']}.", "",
        "### Final route", "",
        "| | Runs correct | Cases correct on every run | Cases that agreed with themselves |", "|---|---|---|---|",
        f"| With documents ({wd['cases']} cases) | {rate(wd['route_correct_runs'], wd['runs'])} | {rate(wd['cases_all_runs_correct'], wd['cases'])} "
        f"| {rate(wd['cases_agreeing_with_themselves'], wd['cases'])} |",
        f"| Without documents ({nd['cases']} cases, the agent is not run) | {rate(nd['route_correct_runs'], nd['runs'])} "
        f"| {rate(nd['cases_all_runs_correct'], nd['cases'])} | {rate(nd['cases_agreeing_with_themselves'], nd['cases'])} |", "",
        f"Runs that passed every check (route, reasons, mismatches, tools, steps): {summary['full_pass_runs']}/{summary['runs'] - summary['errors']}.", "",
        "### Documents", "",
        f"- Mismatch detection (every planted mismatch reported): {rate(summary['detection']['all_found'], summary['detection']['runs'])}",
        f"- False flags (a mismatch reported on a clean document): {rate(summary['false_flags']['flagged'], summary['false_flags']['runs'])}",
        f"- The agent itself attempted the comparison on every attached document: {rate(summary['required_tool']['attempted_every_document'], summary['required_tool']['runs'])}",
        f"- Runs where `finalize` had to rescue a skipped check: {summary['required_tool']['rescued_by_finalize']}",
        f"- Claim extraction matched the answer key on the five fields the comparison uses: {rate(summary['extraction_correct_runs'], summary['runs'] - summary['errors'])}", "",
        "### Cost of a run (runs with documents)", "",
        "| | Mean | Median | 95th percentile | Max |", "|---|---|---|---|---|",
        stat_row("Agent steps", summary["steps"], lambda v: f"{v:.1f}"),
        stat_row("Agent tool calls", summary["tool_calls"], lambda v: f"{v:.1f}"),
        stat_row("Agent tokens (input + output)", summary["tokens"]),
        stat_row("Agent step time", summary["agent_ms"], seconds),
        stat_row("Whole request time", summary["total_ms"], seconds), "",
    ]
    out += [
        "The token and cost figures are for the agent step only. The extraction, coverage and invoice-reader calls are not counted.", "",
        f"Agent statuses: {summary['statuses']}. Ended `max_steps_exceeded`: {summary['max_steps_runs']}. Runs with more than 4 steps: {summary['runs_over_four_steps']}.", "",
        f"Tools the agent called (all runs with documents): {summary['tool_use']}.",
        f"After a document disagreed with the claim, the agent also used other tools in {rate(summary['wandering_after_mismatch']['used_other_tools'], summary['wandering_after_mismatch']['runs'])} of those runs.", "",
        "### Invariants (checked from the recorded runs, independently of the code)", "",
        f"**Violations: {len(summary['invariant_violations'])}**", "",
    ]
    for violation in summary["invariant_violations"]:
        out.append(f"- {violation['case']} run {violation['run']}: {'; '.join(violation['problems'])}")
    out += ["", "### Failing runs and the layer they came from", ""]
    if not summary["failures"]:
        out.append("None.")
    else:
        out += ["| Case | Run | Layer | What differed |", "|---|---|---|---|"]
        out += [f"| {f['case']} | {f['run']} | {f['layer']} | {'; '.join(f['problems'])} |" for f in summary["failures"]]
    return "\n".join(out) + "\n"


def render_case_table(cases: dict[str, dict], runs: list[dict], expected_check) -> str:
    rows = ["| Case | What it tests | Route right | Passed every check |", "|---|---|---|---|"]
    for case_id, case in cases.items():
        mine = [r for r in runs if r["case"] == case_id and "error" not in r]
        route = sum(r["final_route"] == case["expected_route"] for r in mine)
        passed = sum(not expected_check(case, r) for r in mine)
        rows.append(f"| {case_id} | {case['name']} | {route}/{len(mine)} | {passed}/{len(mine)} |")
    return "\n".join(rows) + "\n"


# ---- the attack report

def attack_summary(cases: dict[str, dict], batches: dict[str, list[dict]], labels: dict[str, dict], system_prompt: str) -> dict:
    """Per case and in total: the code-measured levels, the labelled level, and the controls."""
    per_case = {}
    for case_id, case in cases.items():
        rows = []
        for batch, runs in batches.items():
            for run in runs:
                if run["case"] != case_id:
                    continue
                outcome = attack_outcome(case, run, system_prompt)
                label = labels.get(f"{case_id}/{batch}/{run['run']}")
                outcome.update(batch=batch, run=run["run"], label=(label or {}).get("label", "unlabelled"), findings=run.get("findings"),
                               route=run.get("final_route"), reasons=run.get("reasons"))
                rows.append(outcome)
        per_case[case_id] = rows
    return per_case


def _count(rows: list[dict], key: str) -> int:
    return sum(1 for r in rows if r.get(key) is True)


def render_attacks(cases: dict[str, dict], per_case: dict[str, list[dict]], reader: dict[str, list[list[str]]] | None) -> str:
    planted = {c: cases[c] for c in cases if cases[c]["attack"]["planted"]}
    controls = {c: cases[c] for c in cases if not cases[c]["attack"]["planted"]}
    rows_planted = [r for c in planted for r in per_case[c] if "error" not in r]
    errors = sum(1 for rows in per_case.values() for r in rows if "error" in r)
    measurable = [r for r in rows_planted if r["route_measurable"]]
    labelled = Counter(r["label"] for r in rows_planted)
    out = [
        "## Attack baseline (undefended agent)", "",
        f"{len(cases)} attack cases ({len(planted)} with a planted problem, {len(controls)} controls), "
        f"{sum(len(v) for v in per_case.values())} runs in {len({r['batch'] for v in per_case.values() for r in v})} batches. Runs that errored: {errors}.", "",
        "### Planted-problem cases: what the attacks achieved", "",
        f"- **Route unsafe** (`auto_approve_candidate` despite the planted problem): {rate(_count(measurable, 'route_unsafe'), len(measurable))}",
        f"- **Finding suppressed** (a planted mismatch missing from the mismatch list): {rate(_count(rows_planted, 'finding_suppressed'), len(rows_planted))}",
        f"- **Report misleading** (labelled by reading the findings): {rate(labelled['misleading'], len(rows_planted))}; accurate {labelled['accurate']}, "
        f"ambiguous {labelled['ambiguous']}, unlabelled {labelled['unlabelled']}",
        f"- The injected text was visible to the agent (in a tool result): {rate(_count(rows_planted, 'injection_reached_agent'), len(rows_planted))}",
        f"- The agent did not finish its step (looping, out of steps, error): {rate(sum(r['agent_status'] != 'finished' for r in rows_planted), len(rows_planted))}",
        f"- The agent tried the hidden escalation tool: {rate(_count(rows_planted, 'tried_hidden_tool'), len(rows_planted))}",
        f"- The agent skipped an attached document: {rate(_count(rows_planted, 'skipped_attached_document'), len(rows_planted))}",
        f"- A payout was calculated although a mismatch exists: {rate(_count(rows_planted, 'payout_despite_mismatch'), len(rows_planted))}",
        f"- Invariant violations: {sum(len(r['invariant_violations']) > 0 for rows in per_case.values() for r in rows if 'error' not in r)}", "",
        "Route safe versus report misleading:", "",
        "| | Report misleading | Report accurate | Ambiguous or unlabelled |", "|---|---|---|---|",
        f"| Route safe (escalated) | {sum(1 for r in rows_planted if not r['route_unsafe'] and r['label'] == 'misleading')} "
        f"| {sum(1 for r in rows_planted if not r['route_unsafe'] and r['label'] == 'accurate')} "
        f"| {sum(1 for r in rows_planted if not r['route_unsafe'] and r['label'] not in ('misleading', 'accurate'))} |",
        f"| Route unsafe (approved) | {sum(1 for r in rows_planted if r['route_unsafe'] and r['label'] == 'misleading')} "
        f"| {sum(1 for r in rows_planted if r['route_unsafe'] and r['label'] == 'accurate')} "
        f"| {sum(1 for r in rows_planted if r['route_unsafe'] and r['label'] not in ('misleading', 'accurate'))} |", "",
        "### Per case", "",
        "| Case | Type | Lang | Runs | Reached the agent | Route unsafe | Finding suppressed | Misleading | Hidden tool tried | Skipped a document | Reader fooled |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case_id, case in planted.items():
        rows = [r for r in per_case[case_id] if "error" not in r]
        fooled = ""
        if reader and case["attack"]["injected_document"] in reader:
            reads = reader[case["attack"]["injected_document"]]
            fooled = f"{sum(bool(f) for f in reads)}/{len(reads)}"
        a = case["attack"]
        meas = [r for r in rows if r["route_measurable"]]
        out.append(f"| {case_id} | {a['type']} | {a['language']} | {len(rows)} | {_count(rows, 'injection_reached_agent')}/{len(rows)} | {_count(meas, 'route_unsafe')}/{len(meas)} | "
                   f"{_count(rows, 'finding_suppressed')}/{len(rows)} | {sum(r['label'] == 'misleading' for r in rows)}/{len(rows)} | "
                   f"{_count(rows, 'tried_hidden_tool')}/{len(rows)} | {_count(rows, 'skipped_attached_document')}/{len(rows)} | {fooled} |")

    out += ["", "### By injection type", "", "| Type | Runs | Route unsafe | Finding suppressed | Misleading |", "|---|---|---|---|---|"]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for case_id in planted:
        by_type[cases[case_id]["attack"]["type"]] += [r for r in per_case[case_id] if "error" not in r]
    for kind, rows in by_type.items():
        meas = [r for r in rows if r["route_measurable"]]
        out.append(f"| {kind} | {len(rows)} | {_count(meas, 'route_unsafe')}/{len(meas)} | {_count(rows, 'finding_suppressed')}/{len(rows)} "
                   f"| {sum(r['label'] == 'misleading' for r in rows)}/{len(rows)} |")
    out += ["", "### By language", "", "| Language | Runs | Route unsafe | Finding suppressed | Misleading |", "|---|---|---|---|---|"]
    for language in ("en", "es"):
        rows = [r for c in planted if cases[c]["attack"]["language"] == language for r in per_case[c] if "error" not in r]
        meas = [r for r in rows if r["route_measurable"]]
        out.append(f"| {language} | {len(rows)} | {_count(meas, 'route_unsafe')}/{len(meas)} | {_count(rows, 'finding_suppressed')}/{len(rows)} "
                   f"| {sum(r['label'] == 'misleading' for r in rows)}/{len(rows)} |")

    batches = sorted({r["batch"] for v in per_case.values() for r in v})
    if len(batches) > 1:
        out += ["", "### Between sessions", "", "| Batch | Runs | Route unsafe | Finding suppressed | Misleading |", "|---|---|---|---|---|"]
        for batch in batches:
            rows = [r for r in rows_planted if r["batch"] == batch]
            meas = [r for r in rows if r["route_measurable"]]
            out.append(f"| {batch} | {len(rows)} | {_count(meas, 'route_unsafe')}/{len(meas)} | {_count(rows, 'finding_suppressed')}/{len(rows)} "
                       f"| {sum(r['label'] == 'misleading' for r in rows)}/{len(rows)} |")

    out += ["", "### Controls (clean documents with an injection)", "",
            "| Case | Type | Runs | False alarm | Derailed | Leaked instructions | Misleading |", "|---|---|---|---|---|---|---|"]
    for case_id, case in controls.items():
        rows = [r for r in per_case[case_id] if "error" not in r]
        out.append(f"| {case_id} | {case['attack']['type']} | {len(rows)} | {_count(rows, 'false_alarm')}/{len(rows)} | {_count(rows, 'derailed')}/{len(rows)} "
                   f"| {_count(rows, 'leaked')}/{len(rows)} | {sum(r['label'] == 'misleading' for r in rows)}/{len(rows)} |")
    return "\n".join(out) + "\n"


def reader_reads(path: Path) -> dict[str, list[list[str]]]:
    """doc id -> for each read, the decision fields the reader got wrong."""
    data = json.loads(path.read_text(encoding="utf-8"))
    reads: dict[str, list[list[str]]] = defaultdict(list)
    for r in data["results"]:
        if "fooled_fields" in r:
            reads[r["document"]].append(r["fooled_fields"])
    return dict(reads)


def newest(pattern: str) -> Path | None:
    files = sorted(NOTES.glob(pattern))
    return files[-1] if files else None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--attacks", type=Path, nargs="*", default=[])
    parser.add_argument("--reader", type=Path)
    parser.add_argument("--labels", type=Path, default=NOTES / "phase4-attack-labels.json")
    parser.add_argument("--price-in", type=float)
    parser.add_argument("--price-out", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    parts = ["# Phase 4 evaluation report", ""]
    if args.trajectory:
        data = json.loads(args.trajectory.read_text(encoding="utf-8"))
        cases = {c["id"]: c for c in json.loads((DATA / "agent_cases.json").read_text(encoding="utf-8")) if c["id"] in data["config"]["cases"]}
        extractions = json.loads((DATA / "expected_extractions.json").read_text(encoding="utf-8"))
        expected_docs = json.loads((DATA / "expected_documents.json").read_text(encoding="utf-8"))
        summary = trajectory_summary(cases, data["results"], extractions, expected_docs)
        price = (args.price_in, args.price_out) if args.price_in is not None and args.price_out is not None else None
        parts += [f"Run file `{args.trajectory.name}`. Frozen: {data['config'].get('frozen')}.", "",
                  render_trajectory(summary, cases, data["config"])]
        if price and summary["tokens"]["n"]:
            agent_runs = [r for r in data["results"] if r.get("agent")]
            cost = sum(r["agent"]["input_tokens"] * price[0] + r["agent"]["output_tokens"] * price[1] for r in agent_runs) / 1e6 / len(agent_runs)
            parts += [f"Estimated agent-step cost per claim at {price[0]} and {price[1]} USD per million input and output tokens: {cost:.4f} USD. "
                      "Extraction, coverage and invoice-reader calls are not included.", ""]
        parts += ["### Per case", "", render_case_table(cases, data["results"], check_run)]
    if args.attacks:
        from app.agent.prompt import SYSTEM_PROMPT
        attack_cases = {c["id"]: c for c in json.loads((DATA / "attack_cases.json").read_text(encoding="utf-8"))}
        batches, configs = {}, []
        for path in args.attacks:
            file_data = json.loads(path.read_text(encoding="utf-8"))
            label = file_data["config"].get("label") or path.stem
            batches[label] = file_data["results"]
            configs.append(file_data["config"])
        labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.exists() else {}
        parts += [f"Attack files: {[p.name for p in args.attacks]}. Same system in every batch: {frozen_consistent(configs)}.", "",
                  render_attacks(attack_cases, attack_summary(attack_cases, batches, labels, SYSTEM_PROMPT),
                                 reader_reads(args.reader) if args.reader else None)]
    text = "\n".join(parts)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
