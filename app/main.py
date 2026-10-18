from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .extractor import extract_claim, ExtractionError
from .rules import triage
from .schemas import TriageResult

app = FastAPI(title="Insurance Claims Copilot")


class ClaimIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


@app.post("/claims/triage", response_model=TriageResult)
def triage_claim(req: ClaimIn):
    try:
        extraction = extract_claim(req.text)
    except ExtractionError as e:
        status = 503 if e.kind in ("rate_limited", "timeout", "connection") else 502
        raise HTTPException(status_code=status, detail=f"{e.kind}: {e}")
    return TriageResult(extraction=extraction, decision=triage(extraction))
