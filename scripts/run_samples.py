import json
import sys
from datetime import datetime
from pathlib import Path

from app.extractor import extract_claim, ExtractionError, MODEL
from app.rules import triage

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["claims", "adversarial"]
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    results = []
    for folder in FOLDERS:
        for path in sorted((ROOT / "data" / folder).glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for run in range(REPEATS):
                entry = {"file": f"{folder}/{path.name}", "run": run + 1}
                try:
                    extraction = extract_claim(text)
                    decision = triage(extraction)
                    entry["extraction"] = extraction.model_dump()
                    entry["decision"] = decision.model_dump()
                except ExtractionError as e:
                    entry["error"] = f"{e.kind}: {e}"
                except Exception as e:
                    # Keep the batch going if the SDK raises something other than ExtractionError.
                    entry["error"] = f"unexpected: {type(e).__name__}: {e}"
                results.append(entry)
                route = entry.get("decision", {}).get("route", entry.get("error"))
                print(f"{entry['file']:50s} run {run + 1}: {route}")

    out = ROOT / "docs" / "phase-notes" / f"phase2-run-{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({"model": MODEL, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
