"""Run the agent test cases against the real system and save everything that happened.

usage: python -m scripts.run_agent_cases [--split dev|test|all] [--runs N] [--workers N] [--cases case_01,case_05] [--show]

For each case it extracts the claim, runs the whole flow (rules, policy search, coverage, the agent with its documents, the
final route) and records the route, the reasons, every tool call with its arguments and result, tokens and time. It calls the
real OpenAI API and uses the real policy index. Escalations are written to a throwaway SQLite file, not runs/reviews.sqlite.
Nothing is tuned here: it records what the system does. Score it with scripts.score_agent_cases.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.agent.documents import DocumentStore, InvoiceError, OpenAIInvoiceModel
from app.agent.investigate import Investigation, investigate_claim
from app.agent.loop import OpenAIAgentModel
from app.agent.policy_terms import load_policy_terms
from app.agent.prompt import PROMPT_VERSION
from app.agent.reviews import ReviewStore
from app.agent.toolset import ToolDeps
from app.extractor import ExtractionError, extract_claim
from app.rag.coverage import CoverageError, OpenAICoverageModel
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingError, OpenAIEmbedder
from app.rag.index import PolicyIndexError, check_index, open_client

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "data" / "agent_cases.json"


def compact(inv: Investigation) -> dict:
    """What the scoring needs, plus the full tool trace for reading."""
    agent = inv.agent
    return {
        "claim_id": inv.claim_id,
        "final_route": inv.final_route,
        "reasons": inv.reasons,
        "phase3_route": inv.dossier.final_route,
        "phase3_reasons": inv.dossier.reasons,
        "coverage_status": inv.dossier.coverage.status,
        "agent": None if agent is None else {
            "status": agent.status, "error_kind": agent.error_kind, "steps": agent.steps, "tool_calls": agent.tool_calls,
            "input_tokens": agent.input_tokens, "output_tokens": agent.output_tokens, "elapsed_ms": agent.elapsed_ms,
            "turns": [t.model_dump() for t in agent.turns],
        },
        "tools_called": [c.tool for c in inv.tool_calls if c.step and c.tool != "escalate_to_human"],
        "documents_compared": inv.documents_compared,
        "mismatches": [{"document": m.document_id, "field": m.field, "severity": m.severity,
                        "claim_value": m.claim_value, "document_value": m.document_value} for m in inv.mismatches],
        "payout": None if inv.payout is None else inv.payout.model_dump(),
        "findings": inv.findings,
        "review_created": inv.review_created,
        "review_error": inv.review_error,
        "tool_trace": [c.model_dump() for c in inv.tool_calls],
        "versions": inv.agent_versions.model_dump(),
    }


def run_one(case: dict, run: int, deps: ToolDeps, coverage_model: OpenAICoverageModel, agent_model: OpenAIAgentModel) -> dict:
    entry = {"case": case["id"], "run": run, "claim": case["claim"], "documents": case["documents"]}
    try:
        text = (ROOT / "data" / case["claim"]).read_text(encoding="utf-8")
        extraction = extract_claim(text)
        entry["extraction"] = extraction.model_dump()
        entry.update(compact(investigate_claim(extraction, text, case["documents"], deps, coverage_model, agent_model)))
    except (ExtractionError, EmbeddingError, CoverageError, InvoiceError) as e:
        entry["error"] = f"{e.kind}: {e}"
    except Exception as e:  # keep the batch going if something else goes wrong
        entry["error"] = f"unexpected: {type(e).__name__}: {e}"
    return entry


def transcript(entry: dict) -> str:
    """The run as a person would read it."""
    lines = [f"=== {entry['case']} run {entry['run']}  {entry['claim']}  documents {entry['documents']}"]
    if "error" in entry:
        return "\n".join(lines + [f"  ERROR {entry['error']}"])
    lines.append(f"  Phase 3: {entry['phase3_route']} {entry['phase3_reasons']} (coverage {entry['coverage_status']})")
    for record in entry["tool_trace"]:
        lines.append(f"  step {record['step']}  {record['tool']}  {record['arguments']}")
        lines.append(f"      -> [{record['outcome']}] {record['result'][:400]}  ({record['latency_ms']} ms)")
    agent = entry["agent"]
    if agent:
        lines.append(f"  agent: {agent['status']}, {agent['steps']} steps, {agent['tool_calls']} tool calls, "
                     f"{agent['input_tokens']}+{agent['output_tokens']} tokens, {agent['elapsed_ms']} ms")
        lines.append(f"  findings: {entry['findings']}")
    else:
        lines.append("  agent: not run (no documents)")
    lines.append(f"  FINAL: {entry['final_route']} {entry['reasons']}  review_created={entry['review_created']}")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cases", help="comma-separated case ids, for example case_01,case_09")
    parser.add_argument("--show", action="store_true", help="print each run as a readable transcript")
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    elif args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]
    if not cases:
        sys.exit("No cases selected.")

    try:
        collection = check_index(open_client(), EMBEDDING_MODEL)
    except PolicyIndexError as e:
        sys.exit(f"{e}\nBuild the policy index first: python -m scripts.ingest_policy")

    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    reviews = ReviewStore(ROOT / "runs" / f"eval-reviews-{stamp}.sqlite")
    deps = ToolDeps(collection=collection, embedder=OpenAIEmbedder(), documents=DocumentStore(),
                    invoice_model=OpenAIInvoiceModel(), terms=load_policy_terms(), reviews=reviews)
    coverage_model, agent_model = OpenAICoverageModel(), OpenAIAgentModel()

    jobs = [(case, run) for case in cases for run in range(1, args.runs + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(job[0], job[1], deps, coverage_model, agent_model), jobs))

    if args.show:
        for entry in results:
            print(transcript(entry) + "\n")

    out = ROOT / "docs" / "phase-notes" / f"phase4-agent-run-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"split": args.split, "runs": args.runs, "cases": [c["id"] for c in cases],
              "agent_model": agent_model.model, "agent_prompt": PROMPT_VERSION}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    errors = sum("error" in r for r in results)
    print(f"{len(results)} runs ({len(cases)} cases x {args.runs}), {errors} errors. Saved {out.relative_to(ROOT)}")
    print("Score it with: python -m scripts.score_agent_cases")


if __name__ == "__main__":
    main()
