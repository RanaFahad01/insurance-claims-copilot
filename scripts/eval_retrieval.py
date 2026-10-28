"""Measure retrieval on its own, before any language model judges coverage.

usage: python -m scripts.eval_retrieval

Reads data/expected_retrieval.json, searches the real policy index for every case with two query
variants, and saves a dated result file in docs/phase-notes/.

Variant "claim_language": search the policy in the claim's language (the design from WP4).
Variant "english_clauses": always search the English clauses, whatever the claim's language.

Reports recall@k for the in-scope cases, whether the exclusion ("companion") clauses arrive with the
covering clause, and how far the best match is for in-scope, gap and off-topic cases, which is what
the distance cutoff is calibrated from.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from app.rag.embeddings import OpenAIEmbedder
from app.rag.index import DEFAULT_INDEX_DIR, check_index, index_info, open_client
from app.rag.retriever import TOP_K, available_languages, build_query, retrieve_clauses, search_language
from app.schemas import ClaimExtraction

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "data" / "expected_retrieval.json"
K_MAX = 8
RECALL_KS = (1, 2, 3, 4, 5, 6)
CUTOFFS = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 .. 0.80
VARIANTS = ("claim_language", "english_clauses")


class CachingEmbedder:
    """Embeds each distinct text once, so both variants share the same query vectors."""

    def __init__(self, inner):
        self.model = inner.model
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


def to_extraction(case: dict) -> ClaimExtraction:
    return ClaimExtraction(
        incident_date=None, incident_location=case.get("incident_location"), claimant_name=None, policy_number=None,
        damage_type=case["damage_type"],
        estimated_amount=None, currency=None, language=case["language"], summary=case["summary"],
    )


def run_case(collection, embedder, case: dict, variant: str, available: set[str]) -> dict:
    query = build_query(to_extraction(case))
    language = search_language(case["language"], available) if variant == "claim_language" else "en"
    clauses = retrieve_clauses(collection, embedder, query, language=language, k=K_MAX)
    ranks = {c.clause_id: i for i, c in enumerate(clauses, start=1)}
    distances = {c.clause_id: round(c.distance, 4) for c in clauses}
    return {
        "id": case["id"], "category": case["category"], "language": case["language"], "variant": variant,
        "searched_language": language,
        "best_distance": round(clauses[0].distance, 4) if clauses else None,
        "retrieved": [{"clause_id": c.clause_id, "distance": round(c.distance, 4)} for c in clauses],
        "expected_ranks": {cid: ranks.get(cid) for cid in case["expected_clause_ids"]},
        "expected_distances": {cid: distances.get(cid) for cid in case["expected_clause_ids"]},
        "companion_ranks": {cid: ranks.get(cid) for cid in case["companion_clause_ids"]},
    }


def recall_at(results: list[dict], k: int) -> tuple[float, float]:
    """(mean share of expected clauses in the top k, share of cases with every expected clause in the top k)"""
    shares, all_found = [], []
    for r in results:
        ranks = list(r["expected_ranks"].values())
        found = [rank is not None and rank <= k for rank in ranks]
        shares.append(sum(found) / len(found))
        all_found.append(all(found))
    return sum(shares) / len(shares), sum(all_found) / len(all_found)


def companion_rate(results: list[dict], k: int) -> tuple[int, int]:
    """(cases whose companion clauses are all in the top k, cases that have companion clauses)"""
    with_companions = [r for r in results if r["companion_ranks"]]
    ok = sum(all(rank is not None and rank <= k for rank in r["companion_ranks"].values()) for r in with_companions)
    return ok, len(with_companions)


def stats(values: list[float]) -> dict:
    return {"n": len(values), "min": round(min(values), 3), "mean": round(sum(values) / len(values), 3),
            "max": round(max(values), 3)} if values else {"n": 0}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    embedder = CachingEmbedder(OpenAIEmbedder())
    collection = check_index(open_client(DEFAULT_INDEX_DIR), embedder.model)
    available = available_languages(collection)
    info = index_info(collection)

    all_results = {v: [run_case(collection, embedder, c, v, available) for c in cases] for v in VARIANTS}
    summary: dict = {}

    for variant in VARIANTS:
        in_scope = [r for r in all_results[variant] if r["category"] == "in_scope"]
        print(f"\n=== Variant: {variant} ({len(in_scope)} in-scope cases)")
        print(f"{'k':>3} {'mean recall':>12} {'all expected found':>20}")
        recall_table = {}
        for k in RECALL_KS:
            mean_recall, all_found = recall_at(in_scope, k)
            recall_table[k] = {"mean_recall": round(mean_recall, 3), "cases_fully_found": round(all_found, 3)}
            print(f"{k:>3} {mean_recall:>12.3f} {all_found:>20.3f}")
        ok, total = companion_rate(in_scope, TOP_K)
        print(f"Companion (exclusion) clauses all in top {TOP_K}: {ok} of {total} cases")

        by_language = {}
        for lang in sorted({r["language"] for r in in_scope}):
            subset = [r for r in in_scope if r["language"] == lang]
            mean_recall, all_found = recall_at(subset, TOP_K)
            by_language[lang] = {"n": len(subset), "mean_recall_at_4": round(mean_recall, 3),
                                 "cases_fully_found_at_4": round(all_found, 3)}
        print("Recall@4 by claim language:", by_language)

        misses = [(r["id"], {c: rank for c, rank in r["expected_ranks"].items() if rank is None or rank > TOP_K})
                  for r in in_scope]
        misses = [(i, m) for i, m in misses if m]
        print(f"Expected clauses outside the top {TOP_K}:")
        for case_id, missing in misses:
            print(f"  {case_id}: {missing}")
        if not misses:
            print("  none")

        best = {cat: [r["best_distance"] for r in all_results[variant] if r["category"] == cat and r["best_distance"] is not None]
                for cat in ("in_scope", "gap", "off_topic")}
        best_by_lang = {lang: [r["best_distance"] for r in in_scope if r["language"] == lang] for lang in sorted({r["language"] for r in in_scope})}
        print("Best distance by category:", {k: stats(v) for k, v in best.items()})
        print("In-scope best distance by language:", {k: stats(v) for k, v in best_by_lang.items()})

        cutoff_table = []
        print(f"{'cutoff':>7} {'in-scope refused':>17} {'gap flagged':>12} {'off-topic flagged':>18}")
        for t in CUTOFFS:
            row = {
                "cutoff": t,
                "in_scope_refused": sum(d > t for d in best["in_scope"]),
                "gap_flagged": sum(d > t for d in best["gap"]),
                "off_topic_flagged": sum(d > t for d in best["off_topic"]),
            }
            cutoff_table.append(row)
            print(f"{t:>7.2f} {row['in_scope_refused']:>12}/{len(best['in_scope'])} {row['gap_flagged']:>9}/{len(best['gap'])} "
                  f"{row['off_topic_flagged']:>14}/{len(best['off_topic'])}")

        summary[variant] = {
            "recall": recall_table,
            "companions_all_in_top_k": {"ok": ok, "of": total, "k": TOP_K},
            "recall_at_4_by_language": by_language,
            "expected_outside_top_k": dict(misses),
            "best_distance_by_category": {k: stats(v) for k, v in best.items()},
            "in_scope_best_distance_by_language": {k: stats(v) for k, v in best_by_lang.items()},
            "cutoff_table": cutoff_table,
        }

    print("\nGap and off-topic cases, variant claim_language (best distance, top 3):")
    for r in all_results["claim_language"]:
        if r["category"] != "in_scope":
            top = ", ".join(f"{c['clause_id']} {c['distance']:.3f}" for c in r["retrieved"][:3])
            print(f"  {r['id']:18s} {r['best_distance']:.3f}  {top}")

    out = ROOT / "docs" / "phase-notes" / f"phase3-retrieval-{datetime.now():%Y%m%d-%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"embedding_model": info["embedding_model"], "policy_hash": info["policy_hash"], "clauses": info["clauses"],
                   "top_k": TOP_K, "k_max": K_MAX, "cases_file": CASES_FILE.name},
        "summary": summary,
        "results": all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
