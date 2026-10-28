"""Run every claim and attack through the whole pipeline and save what came out.

usage: python -m scripts.run_coverage [runs] [workers]     (defaults: 3 runs, 4 workers)

It reads data/claims and data/adversarial, calls the real OpenAI API (extraction, embeddings, coverage) and the
policy index, and writes docs/phase-notes/phase3-coverage-run-<timestamp>.json. Score it with scripts.score_coverage.
Nothing is tuned or fixed here: it only records what the system does.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.extractor import ExtractionError, extract_claim
from app.rag.chunker import policy_hash
from app.rag.coverage import CoverageError, OpenAICoverageModel
from app.rag.dossier import Dossier, collect_versions, review_claim
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingError, OpenAIEmbedder
from app.rag.index import PolicyIndexError, check_index, index_info, open_client

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["claims", "adversarial"]


def compact(dossier: Dossier) -> dict:
    """What the scoring needs, without the clause texts (they are in the policy files)."""
    c = dossier.coverage
    return {
        "coverage_status": c.status,
        "gate_triggered": c.gate_triggered,
        "cited_chunk_ids": [x.chunk_id for x in c.cited_clauses],
        "retrieved": [{"chunk_id": x.chunk_id, "distance": round(x.distance, 4), "added_by_rule": x.added_by_rule}
                      for x in c.retrieved],
        "rationale": c.rationale,
        "rules_route": dossier.rules_decision.route,
        "rules_reasons": dossier.rules_decision.reasons,
        "final_route": dossier.final_route,
        "reasons": dossier.reasons,
    }


def run_one(file: str, run: int, collection, embedder, coverage_model) -> dict:
    entry = {"file": file, "run": run}
    try:
        text = (ROOT / "data" / file).read_text(encoding="utf-8")
        extraction = extract_claim(text)
        entry["extraction"] = extraction.model_dump()
        entry.update(compact(review_claim(extraction, text, collection, embedder, coverage_model)))
    except (ExtractionError, EmbeddingError, CoverageError) as e:
        entry["error"] = f"{e.kind}: {e}"
    except Exception as e:  # keep the batch going if something else goes wrong
        entry["error"] = f"unexpected: {type(e).__name__}: {e}"
    return entry


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    embedder = OpenAIEmbedder()
    try:
        collection = check_index(open_client(), EMBEDDING_MODEL)
    except PolicyIndexError as e:
        sys.exit(f"{e}")
    details = index_info(collection)
    if details["policy_hash"] != policy_hash():
        sys.exit("The index was built from a different policy than the files on disk. Run: python -m scripts.ingest_policy")
    coverage_model = OpenAICoverageModel()

    files = [f"{folder}/{p.name}" for folder in FOLDERS for p in sorted((ROOT / "data" / folder).glob("*.txt"))]
    jobs = [(file, run) for file in files for run in range(1, runs + 1)]
    print(f"{len(files)} files x {runs} runs = {len(jobs)} pipeline runs, {workers} at a time")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, file, run, collection, embedder, coverage_model) for file, run in jobs]
        for i, future in enumerate(futures, 1):
            results.append(future.result())
            if i % 20 == 0 or i == len(futures):
                errors = sum("error" in r for r in results)
                print(f"  {i}/{len(futures)} done, {errors} errors")

    versions = collect_versions(coverage_model, details)
    out = ROOT / "docs" / "phase-notes" / f"phase3-coverage-run-{datetime.now():%Y%m%d-%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"versions": versions.model_dump(), "runs": runs, "files": len(files), "workers": workers,
              "started": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
