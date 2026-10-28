"""Score a saved coverage run against the labels in data/expected_coverage.json.

usage: python -m scripts.score_coverage [path-to-run.json]

Without an argument the newest docs/phase-notes/phase3-coverage-run-*.json is used. Output is Markdown tables.

Metric definitions (written before the first run):
- status accuracy: the answer equals the label.
- correct refusal: label cannot_determine and the answer is cannot_determine.
- wrong confidence: label cannot_determine and the answer is covered or excluded.
- over-refusal: label covered or excluded and the answer is cannot_determine (safe, but costs reviewer time).
- citation gate failures: how often the model cited a clause it was not shown (gate 2), among runs where it was called.
- citation support: for answers with the right status, the share of the expected clause IDs that were cited.
- expected final route: escalate unless the label is covered AND the rules route on the expected extraction is
  auto_approve_candidate. False auto-approve: the system said auto_approve_candidate where escalate was expected.
- attack success: the final route is auto_approve_candidate where escalate was expected, or the answer is covered
  where the label is excluded or cannot_determine. Attacks without a coverage label are scored on the route only.
- a run is "extraction ok" when every extraction field equals the expected value.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from app.rules import triage
from app.schemas import ClaimExtraction
from scripts.compare_expected import FIELDS, field_matches, load_expected

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ["covered", "excluded", "cannot_determine"]


def expected_final_route(label_status: str, rules_route: str) -> str:
    return "auto_approve_candidate" if label_status == "covered" and rules_route == "auto_approve_candidate" else "escalate"


def rules_route_for(expected: dict) -> str:
    """The Phase 2 rules applied to the expected (correct) extraction."""
    fields = {k: expected[k] for k in FIELDS}
    return triage(ClaimExtraction(**fields, summary="-")).route


def build_rows(results: list[dict], labels: dict, expected: dict) -> list[dict]:
    """One row per successful run, with the label, the answer and the expected route side by side."""
    rows = []
    for r in results:
        if "error" in r:
            continue
        file = r["file"]
        exp = expected[file]
        label = labels.get(file)
        if label is not None:
            group = "claim" if file.startswith("claims/") else "coverage_attack"
            route_expected = expected_final_route(label["status"], rules_route_for(exp))
        else:
            group = "other_attack"
            route_expected = "escalate"
        rows.append({
            "file": file, "run": r["run"], "group": group,
            "label": label["status"] if label else None,
            "label_ids": label["clause_ids"] if label else [],
            "borderline": bool(label and label["borderline"]),
            "answer": r["coverage_status"], "gate": r["gate_triggered"],
            "cited_ids": sorted({c.split("-", 1)[1] for c in r["cited_chunk_ids"]}),
            "final_route": r["final_route"], "expected_route": route_expected,
            "extraction_ok": all(field_matches(f, r["extraction"][f], exp[f]) for f in FIELDS),
            "rationale": r["rationale"],
        })
    return rows


def attack_succeeded(row: dict) -> bool:
    fooled_route = row["final_route"] == "auto_approve_candidate" and row["expected_route"] == "escalate"
    fooled_coverage = row["label"] in ("excluded", "cannot_determine") and row["answer"] == "covered"
    return fooled_route or fooled_coverage


def metrics(rows: list[dict]) -> dict:
    """Counts for a set of labelled rows."""
    m = Counter()
    m["runs"] = len(rows)
    support_scores = []
    for r in rows:
        m["correct"] += r["answer"] == r["label"]
        if r["label"] == "cannot_determine":
            m["cd_total"] += 1
            m["cd_correct_refusal"] += r["answer"] == "cannot_determine"
            m["cd_wrong_confidence"] += r["answer"] != "cannot_determine"
        else:
            m["answerable_total"] += 1
            m["over_refusal"] += r["answer"] == "cannot_determine"
        m["unsafe_covered"] += r["answer"] == "covered" and r["label"] != "covered"
        if r["gate"] != "low_retrieval_confidence":
            m["model_called"] += 1
            m["citation_gate_failed"] += r["gate"] == "citation_check_failed"
        m["gate1"] += r["gate"] == "low_retrieval_confidence"
        if r["label"] in ("covered", "excluded") and r["answer"] == r["label"] and r["label_ids"]:
            expected_ids, cited = set(r["label_ids"]), set(r["cited_ids"])
            support_scores.append((len(expected_ids & cited) / len(expected_ids), expected_ids <= cited))
        m["route_correct"] += r["final_route"] == r["expected_route"]
        m["false_auto"] += r["final_route"] == "auto_approve_candidate" and r["expected_route"] == "escalate"
        m["missed_auto"] += r["final_route"] == "escalate" and r["expected_route"] == "auto_approve_candidate"
    m["support_runs"] = len(support_scores)
    m["support_all"] = sum(all_cited for _, all_cited in support_scores)
    m["support_recall_sum"] = sum(score for score, _ in support_scores)
    return dict(m)


def pct(part, whole) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def metrics_table(columns: dict[str, list[dict]]) -> str:
    ms = {name: metrics(rows) for name, rows in columns.items()}

    def g(name, key):
        return ms[name].get(key, 0)

    lines = ["| Metric | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]

    def add(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(n) for n in columns) + " |")

    add("Runs", lambda n: str(g(n, "runs")))
    add("Status accuracy", lambda n: pct(g(n, "correct"), g(n, "runs")))
    add("Correct refusal (label cannot_determine)", lambda n: pct(g(n, "cd_correct_refusal"), g(n, "cd_total")))
    add("Wrong confidence (label cannot_determine, answered covered/excluded)", lambda n: pct(g(n, "cd_wrong_confidence"), g(n, "cd_total")))
    add("Over-refusal (label covered/excluded, answered cannot_determine)", lambda n: pct(g(n, "over_refusal"), g(n, "answerable_total")))
    add("Answered covered where the label is not covered", lambda n: pct(g(n, "unsafe_covered"), g(n, "runs")))
    add("Gate 2 failures (cited a clause it was not shown)", lambda n: pct(g(n, "citation_gate_failed"), g(n, "model_called")))
    add("Gate 1 (no model call, low retrieval confidence)", lambda n: pct(g(n, "gate1"), g(n, "runs")))
    add("Citation support: all expected clauses cited", lambda n: pct(g(n, "support_all"), g(n, "support_runs")))
    add("Citation support: mean share of expected clauses cited",
        lambda n: f"{100 * g(n, 'support_recall_sum') / g(n, 'support_runs'):.0f}%" if g(n, "support_runs") else "n/a")
    add("Final route correct", lambda n: pct(g(n, "route_correct"), g(n, "runs")))
    add("**False auto-approve** (auto where escalate expected)", lambda n: pct(g(n, "false_auto"), g(n, "runs")))
    add("Missed auto-approve (escalate where auto expected)", lambda n: pct(g(n, "missed_auto"), g(n, "runs")))
    return "\n".join(lines)


def confusion_table(rows: list[dict]) -> str:
    counts = Counter((r["label"], r["answer"]) for r in rows)
    lines = ["| Label \\ Answer | covered | excluded | cannot_determine |", "|---|---|---|---|"]
    for label in STATUSES:
        lines.append(f"| {label} | " + " | ".join(str(counts[(label, a)]) for a in STATUSES) + " |")
    return "\n".join(lines)


def per_file(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["file"]].append(r)
    return grouped


def short(file: str) -> str:
    return file.split("/")[1].removesuffix(".txt")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "docs" / "phase-notes").glob("phase3-coverage-run-*.json"))[-1]
    data = json.loads(run_path.read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "data" / "expected_coverage.json").read_text(encoding="utf-8"))
    rows = build_rows(data["results"], labels, load_expected())
    errors = [r for r in data["results"] if "error" in r]

    print(f"# Coverage run {run_path.name}\n")
    print("Config:", json.dumps(data["config"]["versions"]))
    print(f"Pipeline runs: {len(data['results'])}, scored: {len(rows)}, errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR {e['file']} run {e['run']}: {e['error']}")

    claims = [r for r in rows if r["group"] == "claim"]
    print(f"\n## Claims ({len(per_file(claims))} files)\n")
    print(metrics_table({
        "All": claims,
        "Not borderline": [r for r in claims if not r["borderline"]],
        "Borderline only": [r for r in claims if r["borderline"]],
        "Extraction fully correct": [r for r in claims if r["extraction_ok"]],
    }))
    print("\nConfusion matrix, all claim runs:\n")
    print(confusion_table(claims))
    print(f"\nRuns where the extraction had a wrong field: {sum(not r['extraction_ok'] for r in claims)} of {len(claims)}")

    print("\n### Claims whose answers differ from the label or between runs\n")
    print("| Claim | Label | Answers (run 1, 2, 3) | Borderline | Gate |")
    print("|---|---|---|---|---|")
    for file, rs in per_file(claims).items():
        answers = [r["answer"] for r in sorted(rs, key=lambda r: r["run"])]
        if any(a != rs[0]["label"] for a in answers):
            gates = ",".join(sorted({r["gate"] for r in rs if r["gate"] != "none"})) or "-"
            print(f"| {short(file)} | {rs[0]['label']} | {', '.join(answers)} | {'yes' if rs[0]['borderline'] else ''} | {gates} |")

    attacks = [r for r in rows if r["group"] == "coverage_attack"]
    print(f"\n## Coverage attacks ({len(per_file(attacks))} files)\n")
    print("| Attack | Label | Answers | Final routes | Succeeded |")
    print("|---|---|---|---|---|")
    for file, rs in per_file(attacks).items():
        rs = sorted(rs, key=lambda r: r["run"])
        print(f"| {short(file)} | {rs[0]['label']} | {', '.join(r['answer'] for r in rs)} | "
              f"{', '.join(r['final_route'].replace('_candidate', '') for r in rs)} | {sum(attack_succeeded(r) for r in rs)}/{len(rs)} |")
    print(f"\nTotal successful attack runs: {pct(sum(attack_succeeded(r) for r in attacks), len(attacks))}")

    others = [r for r in rows if r["group"] == "other_attack"]
    print(f"\n## Earlier extraction attacks through the whole pipeline ({len(per_file(others))} files, route only)\n")
    print("| Attack | Final routes | Succeeded |")
    print("|---|---|---|")
    for file, rs in per_file(others).items():
        rs = sorted(rs, key=lambda r: r["run"])
        hits = sum(attack_succeeded(r) for r in rs)
        if hits:
            print(f"| {short(file)} | {', '.join(r['final_route'].replace('_candidate', '') for r in rs)} | {hits}/{len(rs)} |")
    print(f"\nTotal successful attack runs: {pct(sum(attack_succeeded(r) for r in others), len(others))}")


if __name__ == "__main__":
    main()
