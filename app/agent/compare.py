"""Does a supporting document agree with the claim? Plain code that follows data/documents/DOCUMENT_LABELING.md.

The claim side is the extraction of the claim text. A field missing on either side is never a mismatch.
"""
import unicodedata
from datetime import date, timedelta
from decimal import Decimal

from ..schemas import ClaimExtraction
from .documents import InvoiceExtraction
from .state import Mismatch

AMOUNT_TOLERANCE = Decimal("0.05")        # of the claimed amount
AMOUNT_HIGH_SEVERITY = Decimal("0.25")    # above this the mismatch is "high", otherwise "low"
LATE_DAYS = 90                            # a document dated later than this after the incident is a (low) mismatch
VALUE_MAX_CHARS = 80                      # document text is clipped before it is shown to the agent


def clip(value: str | None, limit: int = VALUE_MAX_CHARS) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit] + "..."


def name_words(name: str) -> list[str]:
    """Case, accents, punctuation and word order do not matter: "BELL, Marcus" is "Marcus Bell"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return sorted("".join(c if c.isalnum() else " " for c in ascii_name).split())


def policy_key(number: str) -> str:
    """Spaces, hyphens and case do not matter: "HG 4812 05" is "HG-4812-05"."""
    return "".join(c for c in number if c.isalnum()).lower()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None       # not a date we can use, so it is treated like a date that was not stated


def compare_invoice(document_id: str, claim: ClaimExtraction, invoice: InvoiceExtraction) -> list[Mismatch]:
    found: list[Mismatch] = []

    def add(field: str, claim_value: object, document_value: object, severity: str) -> None:
        found.append(Mismatch(
            document_id=document_id, field=field, severity=severity,
            claim_value=clip(None if claim_value is None else str(claim_value)),
            document_value=clip(None if document_value is None else str(document_value)),
        ))

    currencies_differ = bool(claim.currency and invoice.currency and claim.currency != invoice.currency)
    if currencies_differ:
        # The amounts are not compared: the numbers mean different things.
        add("currency", claim.currency, invoice.currency, "high")
    elif claim.estimated_amount is not None and invoice.total is not None:
        claimed, written = Decimal(str(claim.estimated_amount)), Decimal(str(invoice.total))
        difference = abs(written - claimed)
        if difference > AMOUNT_TOLERANCE * claimed:
            severity = "high" if difference > AMOUNT_HIGH_SEVERITY * claimed else "low"
            add("amount", claim.estimated_amount, invoice.total, severity)

    incident, written_on = _parse_date(claim.incident_date), _parse_date(invoice.document_date)
    if incident and written_on:
        if written_on < incident:
            add("date", claim.incident_date, invoice.document_date, "high")
        elif written_on > incident + timedelta(days=LATE_DAYS):
            add("date", claim.incident_date, invoice.document_date, "low")

    if claim.claimant_name and invoice.customer_name and name_words(claim.claimant_name) != name_words(invoice.customer_name):
        add("name", claim.claimant_name, invoice.customer_name, "high")

    if claim.policy_number and invoice.policy_number and policy_key(claim.policy_number) != policy_key(invoice.policy_number):
        add("policy_number", claim.policy_number, invoice.policy_number, "high")

    return found
