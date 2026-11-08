"""Deductibles and limits from the policy, and the payout arithmetic. Plain code: the model never does the math.

The numbers live in data/policy_terms.json, which a test checks against the clause text in both languages.
The payout is information for the reviewer. It is not an input to the final route.
"""
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

DEFAULT_TERMS_PATH = Path(__file__).resolve().parents[2] / "data" / "policy_terms.json"
CENT = Decimal("0.01")

InsuredObject = Literal["home", "contents", "vehicle"]


class PayoutResult(BaseModel):
    insured_object: InsuredObject
    currency: str
    claimed_amount: float
    deductible: float
    payable_before_limit: float        # the claimed amount minus the deductible, never below zero
    limit: float | None                # None when the policy gives no figure (the vehicle: market value)
    over_limit: bool | None            # None when there is no figure to compare with
    payable: float                     # payable_before_limit, capped at the limit when there is one
    notes: list[str]


def load_policy_terms(path: Path = DEFAULT_TERMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def calculate_payout(insured_object: InsuredObject, amount: float, terms: dict) -> PayoutResult:
    """Apply the deductible, then the limit. Raises ValueError for an amount that is not a finite number of zero or more."""
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount) or amount < 0:
        raise ValueError("amount must be a finite number, zero or more")
    entry = terms["insured_objects"][insured_object]
    claimed = _money(Decimal(str(amount)))
    deductible = Decimal(str(entry["deductible"]))
    before_limit = max(claimed - deductible, Decimal(0))

    notes: list[str] = []
    limit = Decimal(str(entry["limit"])) if entry["limit"] is not None else None
    if limit is None:
        over_limit, payable = None, before_limit
        notes.append("limit_is_market_value_not_checked")
    else:
        over_limit, payable = before_limit > limit, min(before_limit, limit)
    if entry["sub_limits"]:
        notes.append("sub_limits_not_applied")

    return PayoutResult(
        insured_object=insured_object,
        currency=terms["currency"],
        claimed_amount=float(claimed),
        deductible=float(deductible),
        payable_before_limit=float(_money(before_limit)),
        limit=float(limit) if limit is not None else None,
        over_limit=over_limit,
        payable=float(_money(payable)),
        notes=notes,
    )


def terms_for(insured_object: InsuredObject, terms: dict) -> dict:
    """One insured object's figures plus the notification periods, as the get_policy_terms tool returns them."""
    entry = terms["insured_objects"][insured_object]
    return {
        "insured_object": insured_object,
        "currency": terms["currency"],
        "deductible": entry["deductible"],
        "limit": entry["limit"],
        "limit_basis": entry["limit_basis"],
        "sub_limits": entry["sub_limits"],
        "notification": {
            "report_damage_within_days": terms["notification"]["report_damage_within_days"],
            "police_report_within_hours": terms["notification"]["police_report_within_hours"],
            "police_report_applies_to": terms["notification"]["police_report_applies_to"],
        },
        "clauses": entry["clauses"] + terms["notification"]["clauses"],
    }
