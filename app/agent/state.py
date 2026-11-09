"""What the investigation found, as typed data.

The final route is decided from this state and never from the agent's text. Only tool code changes it, so a check that
did not run, or ran and failed, leaves the state saying so.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

from ..schemas import ClaimExtraction
from .policy_terms import PayoutResult

if TYPE_CHECKING:
    from .toolset import ToolDeps

AgentStatus = Literal["not_run", "finished", "max_steps_exceeded", "budget_exceeded", "deadline_exceeded", "looping", "error"]


class Mismatch(BaseModel):
    document_id: str
    field: Literal["amount", "currency", "date", "name", "policy_number"]
    claim_value: str | None
    document_value: str | None
    severity: Literal["low", "high"]


class InvestigationState(BaseModel):
    claim_id: str
    document_ids: list[str] = Field(default_factory=list)       # the documents attached to this claim
    compared_document_ids: list[str] = Field(default_factory=list)
    mismatches: list[Mismatch] = Field(default_factory=list)
    clause_ids_seen: list[str] = Field(default_factory=list)
    payout: PayoutResult | None = None                          # information for the reviewer, not used when routing
    agent_status: AgentStatus = "not_run"
    steps: int = 0
    tool_calls: int = 0
    agent_text: str | None = None                               # kept for the dossier, never read when routing

    @field_validator("document_ids")
    @classmethod
    def _no_duplicates(cls, ids: list[str]) -> list[str]:
        return list(dict.fromkeys(ids))

    @property
    def documents_attached(self) -> bool:
        return bool(self.document_ids)

    @property
    def required_checks_done(self) -> bool:
        """True when nothing is attached, or every attached document was compared successfully."""
        return all(d in self.compared_document_ids for d in self.document_ids)

    def mark_compared(self, document_id: str) -> None:
        if document_id not in self.document_ids:
            raise ValueError(f"{document_id} is not attached to this claim")
        if document_id not in self.compared_document_ids:
            self.compared_document_ids.append(document_id)

    def add_mismatches(self, mismatches: list[Mismatch]) -> None:
        self.mismatches.extend(mismatches)

    def record_comparison(self, document_id: str, mismatches: list[Mismatch]) -> None:
        """Replaces anything recorded earlier for this document, so comparing it twice does not count its mismatches twice.
        It raises before changing anything if the document is not attached."""
        if document_id not in self.document_ids:
            raise ValueError(f"{document_id} is not attached to this claim")
        self.mismatches = [m for m in self.mismatches if m.document_id != document_id] + mismatches
        self.mark_compared(document_id)

    def note_clause(self, clause_id: str) -> None:
        if clause_id not in self.clause_ids_seen:
            self.clause_ids_seen.append(clause_id)


@dataclass
class RunContext:
    """What the code knows about this run. The model does not control any of it."""
    extraction: ClaimExtraction
    state: InvestigationState
    deps: "ToolDeps | None" = None    # the injected index, embedder, document store, invoice model, terms and review store
    run_id: str | None = None         # links a review record back to the trace of the run that created it

    @property
    def claim_id(self) -> str:
        return self.state.claim_id

    @property
    def document_ids(self) -> list[str]:
        return self.state.document_ids
