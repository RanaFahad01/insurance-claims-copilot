"""The agent's tools: what the model may call, and the one that only the surrounding code may call.

Each tool is plain code over something that already exists (the policy index, the document store, the comparison, the
payout arithmetic, the review store). The descriptions are prompts: the model chooses tools from them.
Arguments are untrusted. The claim itself is never an argument: it comes from the run context.
"""
from dataclasses import dataclass
from typing import Literal

from chromadb.api.models.Collection import Collection
from pydantic import Field

from ..rag.embeddings import Embedder, EmbeddingError
from ..rag.retriever import TOP_K, get_clause, retrieve_clauses
from .compare import clip, compare_invoice
from .documents import DocumentError, DocumentStore, InvoiceError, InvoiceModel
from .policy_terms import InsuredObject, calculate_payout, terms_for
from .reviews import ReviewStore
from .state import RunContext
from .tools import Tool, ToolArgs, build_registry

REMARKS_MAX_CHARS = 300


@dataclass
class ToolDeps:
    """Everything the tools use, injected so the tests need no network and no API key."""
    collection: Collection
    embedder: Embedder
    documents: DocumentStore
    invoice_model: InvoiceModel
    terms: dict
    reviews: ReviewStore


# --- arguments -------------------------------------------------------------------------------------------------

class SearchPolicyArgs(ToolArgs):
    query: str = Field(min_length=3, max_length=300, description="What to look for, described in plain words.")
    language: Literal["en", "es"] | None = Field(description="Search only this language's clauses, or null for both.")


class GetClauseArgs(ToolArgs):
    clause_id: str = Field(min_length=1, max_length=40, description="The full clause ID, such as en-C2.1.")


class CompareDocumentsArgs(ToolArgs):
    document_id: str = Field(min_length=1, max_length=20, description="One of the attached document IDs, such as doc_01.")


class PolicyTermsArgs(ToolArgs):
    insured_object: InsuredObject = Field(description="home, contents or vehicle.")


class PayoutArgs(ToolArgs):
    insured_object: InsuredObject = Field(description="home, contents or vehicle.")
    amount: float = Field(ge=0, le=10_000_000, allow_inf_nan=False, description="The claimed amount.")
    currency: Literal["EUR", "USD", "GBP", "other"] = Field(description="The currency of the amount.")


class EscalateArgs(ToolArgs):
    summary: str = Field(max_length=500, description="Why a person needs to look at this claim.")
    reasons: list[str] = Field(max_length=20, description="The routing reasons.")


# --- tools -----------------------------------------------------------------------------------------------------

def search_policy_clauses(ctx: RunContext, args: SearchPolicyArgs) -> dict:
    deps = ctx.deps
    try:
        clauses = retrieve_clauses(deps.collection, deps.embedder, args.query, language=args.language, k=TOP_K)
    except EmbeddingError:
        return {"error": "search_unavailable"}
    for clause in clauses:
        ctx.state.note_clause(clause.chunk_id)
    return {"clauses": [
        {"clause_id": c.chunk_id, "title": c.title, "text": c.text, "distance": round(c.distance, 3)} for c in clauses
    ]}


def get_policy_clause(ctx: RunContext, args: GetClauseArgs) -> dict:
    clause = get_clause(ctx.deps.collection, args.clause_id)
    if clause is None:
        return {"error": "clause_not_found", "hint": "Use a full ID such as en-C2.1 or es-C7.2."}
    ctx.state.note_clause(clause.chunk_id)
    return {"clause_id": clause.chunk_id, "title": clause.title, "text": clause.text}


def compare_documents(ctx: RunContext, args: CompareDocumentsArgs) -> dict:
    deps = ctx.deps
    if args.document_id not in ctx.document_ids:
        return {"error": "document_not_attached", "attached": list(ctx.document_ids)}
    try:
        text = deps.documents.load(args.document_id)
    except DocumentError as e:
        return {"error": f"document_{e.kind}"}
    try:
        invoice = deps.invoice_model.extract(text)
    except InvoiceError:
        return {"error": "document_reader_unavailable"}
    if invoice.is_unreadable:
        return {"error": "document_unreadable"}

    mismatches = compare_invoice(args.document_id, ctx.extraction, invoice)
    ctx.state.record_comparison(args.document_id, mismatches)    # last, so a failure above leaves the state unchanged
    return {
        "document_id": args.document_id,
        "agrees_with_claim": not mismatches,
        "mismatches": [m.model_dump(exclude={"document_id"}) for m in mismatches],
        "document": {
            "vendor": clip(invoice.vendor),
            "date": invoice.document_date,
            "total": invoice.total,
            "currency": invoice.currency,
        },
        "remarks": clip(invoice.remarks, REMARKS_MAX_CHARS),
    }


def get_policy_terms(ctx: RunContext, args: PolicyTermsArgs) -> dict:
    return terms_for(args.insured_object, ctx.deps.terms)


def calculate_policy_payout(ctx: RunContext, args: PayoutArgs) -> dict:
    if args.currency != "EUR":
        return {"error": "currency_not_supported", "supported": "EUR"}
    result = calculate_payout(args.insured_object, args.amount, ctx.deps.terms)
    ctx.state.payout = result
    return result.model_dump()


def escalate_to_human(ctx: RunContext, args: EscalateArgs) -> dict:
    review, created = ctx.deps.reviews.create(ctx.claim_id, args.reasons, args.summary)
    return {"review_id": review.claim_id, "created": created}


# --- the registry ----------------------------------------------------------------------------------------------

def default_registry() -> dict[str, Tool]:
    return build_registry([
        Tool(
            "search_policy_clauses",
            "Search the policy for clauses about the damage or event in the claim: what is covered or excluded, time limits "
            "and limits of cover. Use it when you need policy wording you have not seen yet. The policy says nothing about "
            "how to treat invoices or other documents, so do not search for that. Returns up to 4 clauses with their full "
            "IDs (such as en-C2.1), titles, text and a distance (lower is closer; above about 0.65 is a weak match). For a "
            "clause whose ID you already know, use get_clause instead.",
            SearchPolicyArgs, search_policy_clauses,
        ),
        Tool(
            "get_clause",
            "Fetch one policy clause by its exact full ID, for example 'en-C2.1' or 'es-C7.2'. Use it for an ID you "
            "already have, for instance one named inside another clause. Returns the clause text, or clause_not_found.",
            GetClauseArgs, get_policy_clause,
        ),
        Tool(
            "compare_documents",
            "Compare one attached supporting document (an invoice or receipt) with the claim. Use it for every document "
            "ID you were given. Returns the fields that disagree, or an empty list if they agree, plus the document's "
            "remarks. If it returns document_unreadable, the document could not be checked: report that.",
            CompareDocumentsArgs, compare_documents,
        ),
        Tool(
            "get_policy_terms",
            "Get the deductible, limits and notification periods for one kind of insured property: home, contents or "
            "vehicle. Use it only when you need those figures to answer a question about the claim, for example whether the "
            "amount is near a limit or the claim was reported in time. These are the policy's own figures: quote them, do "
            "not guess.",
            PolicyTermsArgs, get_policy_terms,
        ),
        Tool(
            "calculate_payout",
            "Calculate what the policy would pay for a claimed amount on one kind of insured property: home, contents or "
            "vehicle. The deductible is taken off and the limit applied. Only EUR is supported. Call it only when every "
            "attached document agrees with the claim: it works from the claimed amount, so it means nothing while a "
            "document disagrees. Use it instead of doing the arithmetic yourself. The result is information for the "
            "reviewer and does not decide the claim.",
            PayoutArgs, calculate_policy_payout,
        ),
        Tool(
            "escalate_to_human",
            "Create a pending review for a person. Only the surrounding system calls this.",
            EscalateArgs, escalate_to_human,
            side_effect=True, model_visible=False,
        ),
    ])
