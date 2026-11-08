"""The agent's tools against the real OpenAI API: do the strict schemas get accepted, and does the invoice reader work.

Skipped by default. Run it with: pytest -m live tests/test_agent_tools_live.py
It needs OPENAI_API_KEY and costs a fraction of a cent.
"""
import pytest

from app.agent.documents import DocumentStore, OpenAIInvoiceModel
from app.agent.compare import compare_invoice
from app.agent.loop import OpenAIAgentModel
from app.agent.toolset import default_registry
from app.agent.tools import model_schemas
from app.schemas import ClaimExtraction

CLAIM = ClaimExtraction(
    incident_date="2026-03-12", incident_location=None, claimant_name="Marcus Bell", policy_number="HG-2231-88",
    damage_type="water", estimated_amount=800.0, currency="EUR", language="en", summary="A pipe burst under the kitchen sink.",
)


@pytest.mark.live
def test_the_api_accepts_the_tool_schemas():
    response = OpenAIAgentModel().create(
        [{"role": "user", "content": "Say hi."}], model_schemas(default_registry()), 30.0,
    )
    assert response.status == "completed"


@pytest.mark.live
def test_the_invoice_reader_reads_a_clean_and_an_overcharged_document():
    store, reader = DocumentStore(), OpenAIInvoiceModel()
    clean = reader.extract(store.load("doc_14"))
    assert (clean.customer_name, clean.total, clean.currency, clean.document_date) == ("Marcus Bell", 800.0, "EUR", "2026-03-14")
    assert compare_invoice("doc_14", CLAIM, clean) == []

    overcharged = reader.extract(store.load("doc_22"))
    assert [(m.field, m.severity) for m in compare_invoice("doc_22", CLAIM, overcharged)] == [("amount", "high")]

    scan = reader.extract(store.load("doc_07"))
    assert scan.is_unreadable
