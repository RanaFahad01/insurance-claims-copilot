import logging
import uuid

from chromadb.api.models.Collection import Collection
from fastapi import Depends, FastAPI, HTTPException, Path, Response
from pydantic import BaseModel, Field

from .dependencies import get_collection, get_coverage_model, get_embedder, http_error
from .extractor import extract_claim, ExtractionError
from .rag.coverage import CoverageError, CoverageModel
from .rag.dossier import Dossier, review_claim
from .rag.embeddings import Embedder, EmbeddingError
from .rag.retriever import Clause, get_clause
from .rules import triage
from .schemas import TriageResult

app = FastAPI(title="Insurance Claims Copilot")

# One line per review. It holds IDs, distances and decisions, never the claim text. Without this handler the
# line would not show under uvicorn, which only configures its own loggers.
log = logging.getLogger("claims.review")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


class ClaimIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


@app.post("/claims/triage", response_model=TriageResult)
def triage_claim(req: ClaimIn):
    try:
        extraction = extract_claim(req.text)
    except ExtractionError as e:
        raise http_error(e.kind, e)
    return TriageResult(extraction=extraction, decision=triage(extraction))


@app.post("/claims/review", response_model=Dossier)
def review(
    req: ClaimIn,
    response: Response,
    # The index is checked first, so a missing index fails before any call to the language model.
    collection: Collection = Depends(get_collection),
    embedder: Embedder = Depends(get_embedder),
    coverage_model: CoverageModel = Depends(get_coverage_model),
):
    try:
        extraction = extract_claim(req.text)
        dossier = review_claim(extraction, req.text, collection, embedder, coverage_model)
    except (ExtractionError, EmbeddingError, CoverageError) as e:
        raise http_error(e.kind, e)

    request_id = uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = request_id
    retrieved = ",".join(f"{c.chunk_id}:{c.distance:.2f}" for c in dossier.coverage.retrieved)
    log.info(
        "request_id=%s route=%s coverage=%s gate=%s retrieved=%s reasons=%s",
        request_id, dossier.final_route, dossier.coverage.status, dossier.coverage.gate_triggered,
        retrieved, ",".join(dossier.reasons) or "none",
    )
    return dossier


@app.get("/policy/clauses/{clause_id}", response_model=Clause)
def policy_clause(
    clause_id: str = Path(min_length=1, max_length=40),
    collection: Collection = Depends(get_collection),
):
    clause = get_clause(collection, clause_id)
    if clause is None:
        raise HTTPException(status_code=404, detail="No clause with that ID. Use a full ID such as en-C2.1.")
    return clause
