"""The pieces the flow tests share: a sample claim, an invoice reading, fake coverage, and a factory for the tool dependencies."""
import json
import uuid

import chromadb
import pytest
from fakes import FakeEmbedder, FakeInvoiceModel, fn_call

from app.agent.documents import DocumentStore, InvoiceExtraction
from app.agent.policy_terms import load_policy_terms
from app.agent.reviews import ReviewStore
from app.agent.toolset import ToolDeps
from app.rag.chunker import load_policy_chunks, policy_hash
from app.rag.coverage import CoverageAnswer
from app.rag.index import ingest
from app.schemas import ClaimExtraction

CLAIM_TEXT = "Hi, this is Marcus Bell, policy HG-2231-88. A pipe burst under the kitchen sink. The plumber quoted EUR 800."
EXTRACTION = ClaimExtraction(
    incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell", policy_number="HG-2231-88",
    damage_type="water", estimated_amount=800.0, currency="EUR", language="en", summary="A pipe burst under the kitchen sink.",
)
CUTOFF = 2.0     # the fake embedder's distances are not meaningful, so the retrieval gate is switched off


class FakeCoverage:
    model = "fake-coverage"
    prompt_version = "fake-prompt"

    def __init__(self, status="covered"):
        self.status = status
        self.calls = 0

    def assess(self, extraction, claim_text, clauses):
        self.calls += 1
        return CoverageAnswer(coverage_status=self.status, cited_clause_ids=[clauses[0].chunk_id], rationale="Fits the clause.")


def reading(**overrides) -> InvoiceExtraction:
    base = dict(vendor="Hartwell & Pike Plumbing Ltd", document_date="2026-03-14", total=800.0, currency="EUR",
                customer_name="Marcus Bell", policy_number="HG-2231-88", remarks="Paid in full.")
    base.update(overrides)
    return InvoiceExtraction(**base)


@pytest.fixture
def deps_factory(tmp_path):
    embedder = FakeEmbedder()
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex}"
    ingest(client, load_policy_chunks(), embedder, policy_hash(), collection_name=name)
    collection = client.get_collection(name, embedding_function=None)

    def build(invoice_result=None, reviews_file="reviews.sqlite"):
        return ToolDeps(
            collection=collection, embedder=embedder, documents=DocumentStore(),
            invoice_model=FakeInvoiceModel(invoice_result or reading()), terms=load_policy_terms(),
            reviews=ReviewStore(tmp_path / reviews_file),
        )
    yield build
    client.delete_collection(name)


def compare(doc_id: str, call_id: str):
    return fn_call("compare_documents", json.dumps({"document_id": doc_id}), call_id)
