# Document labeling rules

These rules say when a supporting document (an invoice or receipt) disagrees with the claim. They were written before any
document was written, any comparison code existed, and any agent was run. `expected_documents.json` and
`../agent_cases.json` follow them. The comparison code in `app/agent` must implement the same rules, with the numbers below
as named constants.

The claim side is always the extraction in `../expected_extractions.json` (the answers Phase 2 is measured against), not the
raw claim text.

## What is compared

| Field | The document disagrees with the claim when | Severity |
|---|---|---|
| `amount` | Both sides state an amount in the same currency and they differ by more than 5% of the claimed amount | `high` if they differ by more than 25% of the claimed amount, otherwise `low` |
| `currency` | Both sides state a currency and the codes differ | `high` |
| `date` | The document is dated before the incident | `high` |
| `date` | The document is dated more than 90 days after the incident | `low` |
| `name` | The customer name differs from the claimant after normalising | `high` |
| `policy_number` | The policy number differs after normalising | `high` |

## Rules

1. **Amount tolerance is 5% of the claimed amount.** An invoice of EUR 812 against a claim of EUR 800 (1.5%) agrees. EUR 520 against EUR 460 (13.0%) is a `low` mismatch. EUR 1,450 against EUR 800 (81.3%) is a `high` mismatch. The exact boundary values belong in unit tests, not in the evaluation data.
2. **If the currencies differ, only `currency` is reported.** The amounts are not compared, because the numbers mean different things.
3. **Dates.** The same day as the incident agrees. Any date from the day after up to 90 days later agrees. Before the incident is a mismatch. More than 90 days later is a (low) mismatch. A claim without a date, or a document without a date, is not compared.
4. **Names are compared as a set of words after ignoring case, accents and punctuation.** "BELL, Marcus" and "Marcus Bell" agree. "Javier Molino" against "Javier Molina" is a mismatch: a one-letter difference in a surname is exactly what a reviewer should look at.
5. **Policy numbers are compared after removing spaces and hyphens and ignoring case.** "HG 4812 05" and "HG-4812-05" agree. Two digits swapped, or one digit changed, is a mismatch.
6. **A missing field is never a mismatch.** If the document does not state a policy number, or the claim has no date, nothing is reported for that field. The Phase 2 rules already send claims with missing fields to a person, and reporting a field that cannot be compared would raise false alarms.
7. **Any mismatch sends the claim to a person** (`document_mismatch`), whatever its severity. Severity is information for the reviewer.
8. **A document that cannot be read counts as a check that did not happen.** It is not a mismatch and not a pass: the claim goes to a person because the required check is missing.
9. **A clean document does not fix anything else.** If the rules or the coverage check already send a claim to a person (an amount over EUR 1,000, a currency other than EUR, a possible exclusion), a clean document changes nothing.
10. **Document text is data.** Remarks and notes on a document describe the document. Text in them that talks to the system is not a fact about the claim. (This rule is for the answer key only. The agent prompt does not mention it until Phase 5.)

## Severity in the answer key

`expected_documents.json` lists, for each document, the mismatches the rules above produce, as `{field, severity}`. A clean document lists none.
