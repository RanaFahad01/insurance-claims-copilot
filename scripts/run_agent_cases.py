"""Run the agent test cases against the real system and save everything that happened.

usage: python -m scripts.run_agent_cases [--split dev|test|all] [--runs N] [--workers N] [--cases case_01,case_05] [--show] [--engine raw|graph]
                                      [--set agent|attack] [--label batch1]

For each case it extracts the claim, runs the whole flow (rules, policy search, coverage, the agent with its documents, the
final route) and records the route, the reasons, every tool call with its arguments and result, tokens and time. It calls the
real OpenAI API and uses the real policy index. Escalations are written to a throwaway SQLite file, not runs/reviews.sqlite.
Nothing is tuned here: it records what the system does. Score it with scripts.score_agent_cases.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.agent.documents import DocumentStore, InvoiceError, OpenAIInvoiceModel
from app.agent.graph import FlowResources, investigate_with_graph
from app.agent.investigate import Investigation, investigate_claim
from app.agent.loop import OpenAIAgentModel
from app.agent.policy_terms import load_policy_terms
from app.agent.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from app.agent.reviews import ReviewStore
from app.agent.toolset import ToolDeps
from app.extractor import ExtractionError, extract_claim
from app.rag.coverage import CoverageError, OpenAICoverageModel
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingError, OpenAIEmbedder
from app.rag.index import PolicyIndexError, check_index, open_client

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "data" / "agent_cases.json"
ATTACK_CASES_FILE = ROOT / "data" / "attack_cases.json"
ATTACK_DOCUMENTS = ROOT / "data" / "attack_documents"


def hash_files(paths: list[Path], root: Path = ROOT) -> str:
    """A short fingerprint of some files: their names and contents, with line endings normalised."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


def freeze_info(cases_file: Path, documents_dir: Path | None, root: Path = ROOT) -> dict:
    """What was being measured. Two runs with the same values ran the same system on the same data."""
    docs = sorted(p for p in (documents_dir or root / "data" / "documents").glob("*") if p.is_file())
    return {"code": hash_files(list((root / "app").rglob("*.py")), root), "prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
            "cases": hash_files([cases_file], root), "documents": hash_files(docs, root)}


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


def run_one(case: dict, run: int, deps: ToolDeps, coverage_model: OpenAICoverageModel, agent_model: OpenAIAgentModel,
            engine: str = "raw") -> dict:
    entry = {"case": case["id"], "run": run, "claim": case["claim"], "documents": case["documents"]}
    started = time.perf_counter()
    try:
        text = (ROOT / "data" / case["claim"]).read_text(encoding="utf-8")
        if engine == "graph":
            extracted = {}

            def extract(claim_text):     # the real extraction, run inside the graph's first node; kept so it can be recorded
                extracted["value"] = extract_claim(claim_text)
                return extracted["value"]

            resources = FlowResources(deps=deps, coverage_model=coverage_model, agent_model=agent_model, extract=extract)
            investigation = investigate_with_graph(text, case["documents"], resources)
            entry["extraction"] = extracted["value"].model_dump()
        else:
            extraction = extract_claim(text)
            entry["extraction"] = extraction.model_dump()
            investigation = investigate_claim(extraction, text, case["documents"], deps, coverage_model, agent_model)
        entry.update(compact(investigation))
    except (ExtractionError, EmbeddingError, CoverageError, InvoiceError) as e:
        entry["error"] = f"{e.kind}: {e}"
    except Exception as e:  # keep the batch going if something else goes wrong
        entry["error"] = f"unexpected: {type(e).__name__}: {e}"
    entry["total_ms"] = round((time.perf_counter() - started) * 1000)    # the whole request: extraction, search, coverage, agent
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
    parser.add_argument("--engine", choices=["raw", "graph"], default="raw", help="the hand-written flow or the LangGraph version")
    parser.add_argument("--set", choices=["agent", "attack"], default="agent", help="the agent cases, or the attack cases and documents")
    parser.add_argument("--label", default="", help="a name for this batch, kept in the file (for example batch1)")
    args = parser.parse_args()

    attack = args.set == "attack"
    cases_file, documents_dir = (ATTACK_CASES_FILE, ATTACK_DOCUMENTS) if attack else (CASES_FILE, None)
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    elif not attack and args.split != "all":    # every attack case is run unless --cases says otherwise
        cases = [c for c in cases if c["split"] == args.split]
    if not cases:
        sys.exit("No cases selected.")

    try:
        collection = check_index(open_client(), EMBEDDING_MODEL)
    except PolicyIndexError as e:
        sys.exit(f"{e}\nBuild the policy index first: python -m scripts.ingest_policy")

    stamp = f"{datetime.now():%Y%m%d-%H%M}" + ("-graph" if args.engine == "graph" else "") + (f"-{args.label}" if args.label else "")
    reviews = ReviewStore(ROOT / "runs" / f"eval-reviews-{stamp}.sqlite")
    deps = ToolDeps(collection=collection, embedder=OpenAIEmbedder(), documents=DocumentStore(documents_dir) if documents_dir else DocumentStore(),
                    invoice_model=OpenAIInvoiceModel(), terms=load_policy_terms(), reviews=reviews)
    coverage_model, agent_model = OpenAICoverageModel(), OpenAIAgentModel()

    started_at = datetime.now().isoformat(timespec="seconds")
    jobs = [(case, run) for case in cases for run in range(1, args.runs + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(job[0], job[1], deps, coverage_model, agent_model, args.engine), jobs))

    if args.show:
        for entry in results:
            print(transcript(entry) + "\n")

    out = ROOT / "docs" / "phase-notes" / f"phase4-{'attack' if attack else 'agent'}-run-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"set": args.set, "split": "attack" if attack else args.split, "runs": args.runs, "cases": [c["id"] for c in cases],
              "engine": args.engine, "label": args.label, "started_at": started_at, "workers": args.workers,
              "agent_model": agent_model.model, "agent_prompt": PROMPT_VERSION, "frozen": freeze_info(cases_file, documents_dir)}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    errors = sum("error" in r for r in results)
    print(f"{len(results)} runs ({len(cases)} cases x {args.runs}), {errors} errors. Saved {out.relative_to(ROOT)}")
    print("Report it with: python -m scripts.report_agent_eval")


if __name__ == "__main__":
    main()
