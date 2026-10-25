"""Decide, from the retrieved clauses only, whether a claim looks covered. Code gates sit around the language model.

Gate 1 runs before the model: if nothing in the policy is close enough, the model is not called.
Gate 2 runs after it: a covered or excluded answer needs citations, and every citation must be a clause
that was actually retrieved. Anything that fails a gate becomes cannot_determine, which sends the claim to a person.
The system never says "approved" or "denied".
"""
from typing import Literal, Protocol

from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)
from pydantic import BaseModel, Field

from ..schemas import ClaimExtraction
from .retriever import DISTANCE_CUTOFF, RetrievedClause

load_dotenv()

COVERAGE_MODEL = "gpt-6-luna"
PROMPT_VERSION = "coverage-v1"

CoverageStatus = Literal["covered", "excluded", "cannot_determine"]
GateName = Literal["none", "low_retrieval_confidence", "citation_check_failed"]

LOW_CONFIDENCE_RATIONALE = "No policy clause is close enough to this claim to assess coverage."

SYSTEM_PROMPT = """You assess whether an insurance claim is covered by a policy, using only the policy clauses provided.

You are given a short summary of the claim and a list of policy clauses. Each clause is labeled with an ID in square brackets.
Answer with one of three statuses:
- covered: a clause covers the event, and no exclusion or condition in the clauses provided removes the cover on the facts given.
- excluded: a clause clearly removes cover on the facts given, for example the property or event is outside the territory, or a stated exclusion applies.
- cannot_determine: none of the clauses provided addresses the situation, or the answer depends on a fact the summary does not state while the summary hints that it may matter, or the clauses conflict.

Rules:
- Use only the clauses provided. Never use outside knowledge about insurance and never invent a clause.
- Cite the clause IDs you rely on in cited_clause_ids, exactly as written in the square brackets. A covered or excluded answer must cite at least one clause.
- A related clause is not enough. If no clause actually addresses what happened, answer cannot_determine.
- Do not assume an exclusion applies when the summary gives no fact pointing to it. If the summary contains a detail that hints an exclusion or condition might apply but does not settle it, answer cannot_determine.
- Assume the claimant is the policyholder, the property or vehicle is insured, and the event happened during the policy period, unless the summary says otherwise.
- Where the event happened matters: compare it with the territory clause when one is provided.
- Do not calculate payouts, limits or deductibles. Amounts do not change the status. A missed notification deadline is only relevant if the summary says it was missed, and then the answer is cannot_determine.
- If part of the claim is outside the policy, judge the main loss the claimant describes.
- Write the rationale in two or three sentences in English.
"""


class CoverageAnswer(BaseModel):
    """What the language model returns."""
    coverage_status: CoverageStatus = Field(description="covered, excluded or cannot_determine")
    cited_clause_ids: list[str] = Field(
        description="IDs of the clauses the answer relies on, exactly as written in square brackets in the input"
    )
    rationale: str = Field(description="Two or three sentences explaining the conclusion")


class CoverageResult(BaseModel):
    """The answer after the gates."""
    status: CoverageStatus
    cited_clauses: list[RetrievedClause]
    rationale: str
    retrieved: list[RetrievedClause]     # everything the model could see, for transparency
    gate_triggered: GateName


class CoverageError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class CoverageModel(Protocol):
    model: str
    prompt_version: str

    def assess(self, extraction: ClaimExtraction, clauses: list[RetrievedClause]) -> CoverageAnswer:
        ...


def build_user_message(extraction: ClaimExtraction, clauses: list[RetrievedClause]) -> str:
    blocks = [f"[{c.chunk_id}] {c.title}\n{c.text}" for c in clauses]
    return (
        f"Claim summary: {extraction.summary}\n"
        f"Damage type (from the extraction): {extraction.damage_type}\n\n"
        "Policy clauses:\n\n" + "\n\n".join(blocks)
    )


class OpenAICoverageModel:
    def __init__(self, model: str = COVERAGE_MODEL, client: OpenAI | None = None):
        self.model = model
        self.prompt_version = PROMPT_VERSION
        self._client = client if client is not None else OpenAI(timeout=30.0, max_retries=3)

    def assess(self, extraction: ClaimExtraction, clauses: list[RetrievedClause]) -> CoverageAnswer:
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(extraction, clauses)},
                ],
                text_format=CoverageAnswer,
                temperature=0,
                reasoning={"effort": "none"},
            )
        except RateLimitError as e:
            raise CoverageError("rate_limited", "Rate limit hit after retries") from e
        except APITimeoutError as e:
            raise CoverageError("timeout", "Request timed out after retries") from e
        except APIConnectionError as e:
            raise CoverageError("connection", "Could not reach the API") from e
        except APIStatusError as e:
            raise CoverageError("api_error", f"API returned status {e.status_code}") from e

        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
            raise CoverageError("incomplete", f"Response was cut off: {reason}")

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise CoverageError("refusal", f"Model refused: {part.refusal}")

        if response.output_parsed is None:
            raise CoverageError("no_parsed_output", "Model returned no parsed result")
        return response.output_parsed


def _resolve_citation(cited: str, retrieved: list[RetrievedClause]) -> RetrievedClause | None:
    """Match a cited ID to a retrieved clause. 'en-C2.1' must match exactly. A plain 'C2.1' is accepted
    only when exactly one retrieved clause has that clause ID, so it cannot point at the wrong language."""
    cited = cited.strip()
    for clause in retrieved:
        if clause.chunk_id == cited:
            return clause
    same_clause = [c for c in retrieved if c.clause_id == cited]
    return same_clause[0] if len(same_clause) == 1 else None


def verify_citations(answer: CoverageAnswer, retrieved: list[RetrievedClause]) -> CoverageResult:
    """Gate 2. This proves the cited clauses were retrieved, not that they support the conclusion."""
    resolved: list[RetrievedClause] = []
    invalid: list[str] = []
    for cited in answer.cited_clause_ids:
        clause = _resolve_citation(cited, retrieved)
        if clause is None:
            invalid.append(cited)
        elif clause.chunk_id not in {c.chunk_id for c in resolved}:
            resolved.append(clause)

    if answer.coverage_status == "cannot_determine":
        return CoverageResult(status="cannot_determine", cited_clauses=resolved, rationale=answer.rationale,
                              retrieved=retrieved, gate_triggered="none")

    if invalid:
        problem = f"cited clauses that were not retrieved: {', '.join(invalid)}"
    elif not resolved:
        problem = "no clause was cited"
    else:
        return CoverageResult(status=answer.coverage_status, cited_clauses=resolved, rationale=answer.rationale,
                              retrieved=retrieved, gate_triggered="none")

    return CoverageResult(
        status="cannot_determine",
        cited_clauses=[],
        rationale=f"Citation check failed ({problem}). The model had answered '{answer.coverage_status}': {answer.rationale}",
        retrieved=retrieved,
        gate_triggered="citation_check_failed",
    )


def assess_coverage(
    extraction: ClaimExtraction,
    retrieved: list[RetrievedClause],
    model: CoverageModel,
    cutoff: float = DISTANCE_CUTOFF,
) -> CoverageResult:
    """Gate 1, then the model, then gate 2.

    Clauses added by rule (such as the territory clause) are shown to the model but do not count toward
    gate 1, because they were not found by the search and say nothing about how relevant the policy is.
    """
    searched = [c for c in retrieved if not c.added_by_rule]
    if not searched or min(c.distance for c in searched) > cutoff:
        return CoverageResult(status="cannot_determine", cited_clauses=[], rationale=LOW_CONFIDENCE_RATIONALE,
                              retrieved=retrieved, gate_triggered="low_retrieval_confidence")

    answer = model.assess(extraction, retrieved)
    return verify_citations(answer, retrieved)
