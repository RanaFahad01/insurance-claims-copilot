"""The whole flow against the real OpenAI API and the real policy index: extraction, policy search, coverage, the agent
with its documents, the final route and the review record.

Skipped by default. Run it with: pytest -m live tests/test_investigate_live.py
It needs OPENAI_API_KEY and a built index (python -m scripts.ingest_policy), and costs a few cents.
"""
from pathlib import Path

import pytest

from app.agent.documents import DocumentStore, OpenAIInvoiceModel
from app.agent.investigate import investigate_claim
from app.agent.loop import OpenAIAgentModel
from app.agent.policy_terms import load_policy_terms
from app.agent.reviews import ReviewStore
from app.agent.toolset import ToolDeps
from app.extractor import extract_claim
from app.rag.coverage import OpenAICoverageModel
from app.rag.embeddings import EMBEDDING_MODEL, OpenAIEmbedder
from app.rag.index import PolicyIndexError, check_index, open_client

CLAIM = (Path(__file__).resolve().parent.parent / "data" / "claims" / "claim_01_water_en.txt").read_text(encoding="utf-8")


@pytest.mark.live
def test_a_clean_document_and_an_overcharged_one_go_through_the_whole_flow(tmp_path):
    try:
        collection = check_index(open_client(), EMBEDDING_MODEL)
    except PolicyIndexError:
        pytest.skip("No policy index. Build it with: python -m scripts.ingest_policy")

    deps = ToolDeps(collection=collection, embedder=OpenAIEmbedder(), documents=DocumentStore(),
                    invoice_model=OpenAIInvoiceModel(), terms=load_policy_terms(), reviews=ReviewStore(tmp_path / "reviews.sqlite"))
    extraction = extract_claim(CLAIM)
    coverage_model, agent_model = OpenAICoverageModel(), OpenAIAgentModel()

    clean = investigate_claim(extraction, CLAIM, ["doc_14"], deps, coverage_model, agent_model)
    assert clean.agent.status == "finished" and clean.documents_compared == ["doc_14"] and clean.mismatches == []
    assert "compare_documents" in [c.tool for c in clean.tool_calls]
    assert clean.final_route == "auto_approve_candidate" and clean.reasons == [] and clean.review_created is None

    overcharged = investigate_claim(extraction, CLAIM, ["doc_22"], deps, coverage_model, agent_model)
    assert [(m.document_id, m.field) for m in overcharged.mismatches] == [("doc_22", "amount")]
    assert overcharged.final_route == "escalate" and "document_mismatch" in overcharged.reasons
    assert overcharged.review_created is True
    (review,) = deps.reviews.list()
    assert review.claim_id == overcharged.claim_id and "document_mismatch" in review.reasons
