"""Read every supporting document with the real model and score the facts against data/expected_documents.json.

usage: python -m scripts.check_invoices [runs] [workers] [--attack]     (defaults: 3 runs, 4 workers)

--attack reads the attack documents (data/attack_documents) instead and records which decision fields the reader got wrong:
the 'reader fooled' level of the attack baseline.

It calls the real OpenAI API (one call per document and run) and writes docs/phase-notes/phase4-invoice-run-<timestamp>.json.
Nothing is fixed here: it records what the invoice reader does, per field.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.agent.documents import DocumentStore, InvoiceError, OpenAIInvoiceModel
from scripts.eval_metrics import reader_fooled

ROOT = Path(__file__).resolve().parent.parent
FIELD_NAMES = {
    "vendor": "vendor", "document_date": "document_date", "total": "total", "currency": "currency",
    "customer_name": "customer_name", "policy_number": "policy_number", "remarks": "remarks",
}


def same(field: str, got, want) -> bool:
    if field == "total":
        return got is not None and abs(got - want) < 0.005
    if isinstance(want, str) and isinstance(got, str):
        return " ".join(got.split()) == " ".join(want.split())
    return got == want


def run_one(doc_id: str, run: int, store: DocumentStore, model: OpenAIInvoiceModel, expected: dict) -> dict:
    entry = {"document": doc_id, "run": run}
    try:
        invoice = model.extract(store.load(doc_id))
    except InvoiceError as e:
        entry["error"] = f"{e.kind}: {e}"
        return entry
    entry["extracted"] = invoice.model_dump()
    facts = expected["facts"]
    if facts is not None:
        entry["fooled_fields"] = reader_fooled(facts, entry["extracted"])
    if facts is None:
        entry["unreadable_detected"] = invoice.is_unreadable
    else:
        entry["wrong_fields"] = [f for f in FIELD_NAMES if not same(f, getattr(invoice, f), facts[f])]
    return entry


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    attack = "--attack" in sys.argv
    numbers = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = int(numbers[0]) if numbers else 3
    workers = int(numbers[1]) if len(numbers) > 1 else 4
    expected_file = ROOT / "data" / "attack_documents" / "expected.json" if attack else ROOT / "data" / "expected_documents.json"
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    store, model = (DocumentStore(ROOT / "data" / "attack_documents") if attack else DocumentStore()), OpenAIInvoiceModel()

    jobs = [(doc_id, run) for doc_id in sorted(expected) for run in range(1, runs + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: run_one(job[0], job[1], store, model, expected[job[0]]), jobs))

    readable = [r for r in results if "wrong_fields" in r]
    scan = [r for r in results if "unreadable_detected" in r]
    errors = [r for r in results if "error" in r]
    print(f"{len(results)} reads ({len(expected)} documents x {runs} runs), prompt {model.prompt_version}, model {model.model}")
    print(f"errors: {len(errors)}")
    print(f"readable documents: {sum(not r['wrong_fields'] for r in readable)}/{len(readable)} reads had every field right")
    print("per field:")
    for field in FIELD_NAMES:
        wrong = sum(field in r["wrong_fields"] for r in readable)
        print(f"  {field:<14} {len(readable) - wrong}/{len(readable)}")
    print(f"unreadable scan detected: {sum(r['unreadable_detected'] for r in scan)}/{len(scan)}")
    for r in readable:
        if r["wrong_fields"]:
            facts = expected[r["document"]]["facts"]
            for f in r["wrong_fields"]:
                print(f"  MISS {r['document']} run {r['run']} {f}: got {r['extracted'][f]!r}, expected {facts[f]!r}")
    for r in errors:
        print(f"  ERROR {r['document']} run {r['run']}: {r['error']}")

    if attack:
        print("reader fooled (a field the comparison uses differs from the document):")
        for doc_id in sorted(expected):
            reads = [r for r in results if r["document"] == doc_id and "fooled_fields" in r]
            fooled = [r for r in reads if r["fooled_fields"]]
            detail = "; ".join(f"run {r['run']}: {r['fooled_fields']} -> {[r['extracted'][f] for f in r['fooled_fields']]}" for r in fooled)
            print(f"  {doc_id}: {len(fooled)}/{len(reads)} reads fooled  {detail}")
    prefix = "attack-reader" if attack else "invoice"
    out = ROOT / "docs" / "phase-notes" / f"phase4-{prefix}-run-{datetime.now():%Y%m%d-%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"runs": runs, "model": model.model, "prompt": model.prompt_version, "set": "attack" if attack else "main"},
                               "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
