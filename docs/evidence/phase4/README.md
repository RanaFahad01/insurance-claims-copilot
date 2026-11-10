# Evidence behind the Phase 4 numbers

Raw output of the runs quoted in the README and in [`docs/phase4-evaluation.md`](../../phase4-evaluation.md). Every file was
written by a script in `scripts/` from real calls to the OpenAI API. Nothing in them was edited by hand, except
`attacks-labels.json`, which is my reading of the agent's findings. All claims, documents and policy text are invented.

| File | What it is | Made with |
|---|---|---|
| `heldout-test-cases-3x.json` | The 23 held-out test cases, 3 runs each (69 runs): route, reasons, mismatches, every tool call with its result, tokens and time | `python -m scripts.run_agent_cases --split test --runs 3 --label test3` |
| `evaluation-report.md` | The tables computed from the files below | `python -m scripts.report_agent_eval --trajectory ... --attacks ... --reader ... --out ...` |
| `attacks-batch1.json`, `attacks-batch2.json` | The 15 attack cases, 3 runs each, in two batches run back to back (90 runs) | `python -m scripts.run_agent_cases --set attack --runs 3 --label batch1` (and `batch2`) |
| `attacks-reader-alone.json` | The invoice reader on the 17 attack documents, 3 reads each, scored field by field | `python -m scripts.check_invoices 3 4 --attack` |
| `attacks-labels.json` | One label per attack run (accurate, misleading or ambiguous) with a reason, written by reading the findings against the rule in `data/ATTACKS.md` | by hand |
| `invoice-reader-3x.json` | The invoice reader on the 22 main documents, 3 reads each: all 63 reads of the 21 readable documents had every field right, and the unreadable scan was recognised 3 times out of 3 | `python -m scripts.check_invoices 3 4` |
| `dev-pass1.json`, `dev-pass2-3x.json` | The 5 development cases before and after the tool descriptions were tightened. These cases were used for tuning, so they are not evidence of how well the system works | `python -m scripts.run_agent_cases --split dev` |
| `dev-raw-engine.json`, `dev-graph-engine.json` | The same 5 development cases through the hand-written flow and through the LangGraph version, one run each, one after the other | `... --engine raw` and `--engine graph` |

Each run file starts with a `config` block. The `frozen` entry holds short fingerprints of the code, the system prompt, the cases
and the documents, so two files with the same fingerprints came from the same system on the same data.

To rebuild the tables from these files:

```bash
python -m scripts.report_agent_eval --trajectory docs/evidence/phase4/heldout-test-cases-3x.json \
  --attacks docs/evidence/phase4/attacks-batch1.json docs/evidence/phase4/attacks-batch2.json \
  --reader docs/evidence/phase4/attacks-reader-alone.json --labels docs/evidence/phase4/attacks-labels.json
```
