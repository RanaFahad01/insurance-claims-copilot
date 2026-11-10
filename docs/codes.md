# Codes the API returns

Every value below is language-neutral: a client can show its own label for it (for example in Spanish). The free text in
a response is different. The agent's `findings`, the review `summary`, the extraction `summary` and the coverage `rationale`
are written in English, also for Spanish claims. A test (`tests/test_codes_reference.py`) fails if this page and the code
disagree.

## Routes

`final_route` in a dossier or an investigation.

| Code | Meaning |
|---|---|
| `auto_approve_candidate` | Nothing asked for a person. The best the system says: it never says "approved". |
| `escalate` | At least one reason applies, so a person looks at the claim. |

## Reasons

`reasons` lists every trigger that applied. One reason is enough to escalate.

| Code | Meaning |
|---|---|
| `missing_incident_date` | The claim does not state when it happened. |
| `missing_claimant_name` | The claim does not state who is claiming. |
| `missing_policy_number` | The claim does not state a policy number. |
| `missing_estimated_amount` | The claim does not state an amount. |
| `damage_type_unclear` | The cause of the damage could not be classified. |
| `non_positive_amount` | The stated amount is zero or negative. |
| `currency_not_eur` | The claim is in a currency other than euro. |
| `amount_above_auto_threshold` | The amount is above EUR 1,000, the fast-track limit, and not above EUR 5,000. |
| `amount_above_high_threshold` | The amount is above EUR 5,000. |
| `retrieval_low_confidence` | No policy clause was close enough to the claim, so the coverage model was not asked. |
| `citation_check_failed` | A clause the coverage answer cited could not be matched to the clauses it was shown. |
| `coverage_possible_exclusion` | The coverage check found the claim excluded. A person confirms. |
| `coverage_undetermined` | The coverage check could not determine whether the claim is covered. |
| `document_mismatch` | A supporting document disagrees with the claim (see the mismatch list). |
| `required_check_missing` | An attached document was not successfully compared: it was unreadable, not found or skipped. |
| `agent_incomplete` | The agent step did not finish (out of steps, tokens or time, looping, or a model error). |

## Coverage status

`coverage.status` in a dossier.

| Code | Meaning |
|---|---|
| `covered` | The cited clauses cover the claim. |
| `excluded` | The cited clauses exclude the claim. |
| `cannot_determine` | The policy is silent or unclear, or a check failed. |

## Coverage gate

`coverage.gate_triggered`: a check in code that stopped the coverage answer.

| Code | Meaning |
|---|---|
| `none` | No gate fired. |
| `low_retrieval_confidence` | The closest clause was too far from the claim. |
| `citation_check_failed` | A cited clause was not one the model was shown. |

## Mismatch fields

`mismatches[].field`: what a supporting document gets wrong, compared with the claim.

| Code | Meaning |
|---|---|
| `amount` | The document total differs from the claimed amount by more than 5% (`high` above 25%). |
| `currency` | The document names a different currency. Only this is reported then, not the amount. |
| `date` | The document is dated before the incident (`high`), or more than 90 days after it (`low`). |
| `name` | The customer name on the document differs from the claimant (compared as a set of words). |
| `policy_number` | The policy number on the document differs from the claim's (spaces and hyphens ignored). |

## Mismatch severity

| Code | Meaning |
|---|---|
| `low` | Information for the reviewer. Any mismatch still escalates the claim. |
| `high` | The same, and worth looking at first. |

## Agent status

`agent.status` in an investigation.

| Code | Meaning |
|---|---|
| `not_run` | No documents were attached, so the agent step did not run. |
| `finished` | The model stopped asking for tools within its limits. |
| `max_steps_exceeded` | The model used more than 8 turns. |
| `budget_exceeded` | The run used more than 30,000 tokens. |
| `deadline_exceeded` | The run took longer than 60 seconds. |
| `looping` | The model repeated the same tool call more than twice. |
| `error` | A model call failed (see the error kind). |

## Agent error kinds

`agent.error_kind` when the status is `error`.

| Code | Meaning |
|---|---|
| `rate_limited` | The model service asked us to slow down. |
| `timeout` | A model call took too long. |
| `connection` | The model service could not be reached. |
| `api_error` | The model service returned an error. |
| `bad_request` | The request was refused as malformed, usually our own mistake. |
| `incomplete` | The model's answer was cut off. |
| `refusal` | The model refused to answer. |

## Review status

`status` of a record in `GET /reviews`.

| Code | Meaning |
|---|---|
| `pending` | Waiting for a person. |
| `resolved` | A person has recorded a decision. |

## Review decisions

`resolution` in `POST /reviews/{id}/resolve`. It records the person's decision and changes nothing else.

| Code | Meaning |
|---|---|
| `approved` | The person approves the claim. |
| `rejected` | The person rejects the claim. |
| `needs_more_information` | The person asks for more information. |

## Run status

`run.status` in `GET /runs/{id}`.

| Code | Meaning |
|---|---|
| `completed` | The investigation finished and produced a route. |
| `failed` | It stopped before producing one (see the error stage). |

## Run error stages

`run.error_stage` when the run status is `failed`.

| Code | Meaning |
|---|---|
| `extraction` | Reading the claim text failed. |
| `embedding` | The policy search failed. |
| `coverage` | The coverage check failed. |
| `unexpected` | A bug: nothing else explains it. |

## Tool outcomes

`outcome` of a `tool_call` event in a run's trace.

| Code | Meaning |
|---|---|
| `ok` | The tool ran and returned a result. |
| `unknown_tool` | The model asked for a tool that does not exist or that it may not use. Nothing ran. |
| `invalid_arguments` | The arguments did not match the tool's schema. Nothing ran. |
| `arguments_too_long` | The arguments were longer than the limit. Nothing ran. |
| `tool_failed` | The tool raised an error. The model is told it failed. |
| `result_too_large` | The result was larger than the limit and was not shown to the model. |
| `duplicate_call` | The same call was made again. It was answered without running. |
| `skipped_limit_reached` | More calls in one turn than allowed. The extra ones were answered without running. |
| `document_not_attached` | The model asked about a document that is not attached to this claim. |
| `document_invalid_id` | The document ID is not a valid ID. |
| `document_not_found` | There is no such document. |
| `document_too_large` | The document is too long to read. |
| `document_reader_unavailable` | The invoice reader failed. |
| `document_unreadable` | Nothing could be read from the document. The check was not done. |
| `clause_not_found` | There is no policy clause with that ID. |
| `search_unavailable` | The policy search failed. |
| `currency_not_supported` | The payout calculation only supports euro. |
