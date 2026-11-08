import pytest

from app.agent.policy_terms import DEFAULT_TERMS_PATH, calculate_payout, load_policy_terms, terms_for

TERMS = load_policy_terms(DEFAULT_TERMS_PATH)


def payout(insured_object, amount):
    return calculate_payout(insured_object, amount, TERMS)


def test_the_deductible_is_taken_off_the_claimed_amount():
    result = payout("home", 800)
    assert (result.claimed_amount, result.deductible, result.payable_before_limit, result.payable) == (800.0, 250.0, 550.0, 550.0)
    assert result.limit == 300000.0 and result.over_limit is False and result.currency == "EUR"


def test_the_vehicle_deductible_is_higher():
    assert payout("vehicle", 3000).deductible == 500.0
    assert payout("contents", 3000).deductible == 250.0


@pytest.mark.parametrize("amount", [0, 100, 249.99, 250])
def test_a_claim_at_or_below_the_deductible_pays_nothing(amount):
    result = payout("home", amount)
    assert result.payable == 0.0 and result.payable_before_limit == 0.0 and result.over_limit is False


def test_the_limit_is_applied_after_the_deductible():
    at_limit = payout("home", 300250)
    assert at_limit.payable == 300000.0 and at_limit.over_limit is False
    over = payout("home", 300250.01)
    assert over.payable == 300000.0 and over.over_limit is True and over.payable_before_limit == 300000.01


def test_contents_have_a_lower_limit_and_say_the_item_limits_were_not_applied():
    result = payout("contents", 60000)
    assert result.limit == 50000.0 and result.over_limit is True and result.payable == 50000.0
    assert "sub_limits_not_applied" in result.notes
    assert "sub_limits_not_applied" not in payout("home", 1000).notes


def test_the_vehicle_has_no_figure_to_compare_with():
    result = payout("vehicle", 9000)
    assert result.limit is None and result.over_limit is None
    assert result.payable == result.payable_before_limit == 8500.0
    assert result.notes == ["limit_is_market_value_not_checked"]


def test_amounts_are_rounded_to_cents_half_up_without_binary_float_error():
    assert payout("home", 1000.005).claimed_amount == 1000.01
    assert payout("home", 250.1 + 0.2).claimed_amount == 250.3
    assert payout("home", 0.1 + 0.2).claimed_amount == 0.3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1, -0.01, True, "5", None])
def test_an_amount_that_is_not_a_finite_number_of_zero_or_more_is_refused(bad):
    with pytest.raises(ValueError):
        payout("home", bad)


def test_an_unknown_insured_object_is_a_key_error_so_it_cannot_pass_silently():
    with pytest.raises(KeyError):
        payout("boat", 100)


def test_terms_for_returns_one_object_with_the_notification_periods():
    terms = terms_for("contents", TERMS)
    assert terms["deductible"] == 250 and terms["limit"] == 50000 and terms["limit_basis"] == "sum_insured"
    assert terms["sub_limits"]["single_item"] == 2500
    assert terms["notification"]["report_damage_within_days"] == 30 and terms["notification"]["police_report_within_hours"] == 48
    assert "C7.1" in terms["clauses"] and "C7.3" in terms["clauses"]
    assert terms_for("vehicle", TERMS)["limit"] is None


def test_the_calculation_reads_the_terms_it_is_given_not_a_copy_of_the_numbers():
    changed = {**TERMS, "insured_objects": {**TERMS["insured_objects"], "home": {**TERMS["insured_objects"]["home"], "deductible": 1000}}}
    assert calculate_payout("home", 3000, changed).payable == 2000.0
