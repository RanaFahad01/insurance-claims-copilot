"""What the investigation found, as typed data.

The final route is decided from this state and never from the agent's text. Only tool code changes it, so a check that
did not run, or ran and failed, leaves the state saying so.
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..schemas import ClaimExtraction

AgentStatus = Literal["not_run", "finished", "max_steps_exceeded", "error"]


class Mismatch(BaseModel):
    document_id: str
    field: str
    claim_value: str | None
    document_value: str | None
    severity: Literal["low", "high"]


class InvestigationState(BaseModel):
    claim_id: str
    document_ids: list[str] = Field(default_factory=list)       # the documents attached to this claim
    compared_document_ids: list[str] = Field(default_factory=list)
    mismatches: list[Mismatch] = Field(default_factory=list)
    clause_ids_seen: list[str] = Field(default_factory=list)
    payout: dict | None = None                                  # gets a real model when the payout tool exists
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

    def note_clause(self, clause_id: str) -> None:
        if clause_id not in self.clause_ids_seen:
            self.clause_ids_seen.append(clause_id)


@dataclass
class RunContext:
    """What the code knows about this run. The model does not control any of it."""
    extraction: ClaimExtraction
    state: InvestigationState

    @property
    def claim_id(self) -> str:
        return self.state.claim_id

    @property
    def document_ids(self) -> list[str]:
        return self.state.document_ids
