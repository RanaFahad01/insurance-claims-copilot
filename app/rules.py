from .schemas import ClaimExtraction, RouteDecision

AUTO_APPROVE_MAX_EUR = 1000.0
HIGH_AMOUNT_EUR = 5000.0
REQUIRED_FIELDS = ("incident_date", "claimant_name", "policy_number", "estimated_amount")
TYPE_WEIGHT = {"fire": 20, "collision": 15, "theft": 10, "water": 10, "other": 5, "unclear": 15}


def triage(claim: ClaimExtraction) -> RouteDecision:
    reasons: list[str] = []
    score = TYPE_WEIGHT[claim.damage_type]

    for field in REQUIRED_FIELDS:
        if getattr(claim, field) is None:
            reasons.append(f"missing_{field}")
            score += 10

    if claim.damage_type == "unclear":
        reasons.append("damage_type_unclear")

    amount = claim.estimated_amount
    if amount is not None:
        if amount <= 0:
            reasons.append("non_positive_amount")
        if claim.currency != "EUR":
            reasons.append("currency_not_eur")
            score += 20
        elif amount > HIGH_AMOUNT_EUR:
            reasons.append("amount_above_high_threshold")
            score += 40
        elif amount > AUTO_APPROVE_MAX_EUR:
            reasons.append("amount_above_auto_threshold")
            score += 20

    route = "escalate" if reasons else "auto_approve_candidate"
    return RouteDecision(route=route, severity_score=min(score, 100), reasons=reasons)
