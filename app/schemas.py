from typing import Literal
from pydantic import BaseModel, Field

DamageType = Literal["water", "fire", "theft", "collision", "other", "unclear"]


class ClaimExtraction(BaseModel):
    incident_date: str | None = Field(
        description="Incident date as YYYY-MM-DD. Null if the text gives no explicit full date including year."
    )
    incident_location: str | None = Field(
        description="Place where the incident happened, as written in the claim (a city, region or country). Null if the text does not state one. Not a room, not the place a letter was written, not the claimant's address."
    )
    claimant_name: str | None = Field(
        description="Full name of the person filing the claim, or null."
    )
    policy_number: str | None = Field(
        description="Policy number exactly as written, or null."
    )
    damage_type: DamageType = Field(
        description="Type of damage. Use 'unclear' if it cannot be determined or several incidents are described."
    )
    estimated_amount: float | None = Field(
        description="Total claimed amount as a plain number, or null if no amount is stated."
    )
    currency: Literal["EUR", "USD", "GBP", "other"] | None = Field(
        description="Currency of the amount, or null if not stated."
    )
    language: Literal["en", "es", "other"] = Field(
        description="Language of the claim text."
    )
    summary: str = Field(
        description="One neutral sentence summarizing the incident."
    )


class RouteDecision(BaseModel):
    route: Literal["auto_approve_candidate", "escalate"]
    severity_score: int
    reasons: list[str]


class TriageResult(BaseModel):
    extraction: ClaimExtraction
    decision: RouteDecision
