import math

import pytest

from expplat import power
from expplat.common import UserError


def test_sample_size_matches_published_value():
    # 10% -> 11% conversion, 80% power, alpha 5%, two-sided: widely quoted as about 14,750 per group
    r = power.plan("proportions", "sample_size", baseline_rate=0.10, mde=0.10, mde_type="relative")
    assert abs(r["details"]["n_control"] - 14750) / 14750 < 0.01
    assert r["details"]["n_treatment"] == r["details"]["n_control"]


def test_means_matches_textbook_formula():
    # n = 2 * (z_a/2 + z_b)^2 * sd^2 / delta^2 -> about 63 per group for d = 0.5
    r = power.plan("means", "sample_size", sd=10, mde=5)
    assert 62 <= r["details"]["n_control"] <= 66


def test_round_trip_power_and_mde():
    n = power.plan("proportions", "sample_size", baseline_rate=0.2, mde=0.02, mde_type="absolute")["details"]["n_control"]
    p = power.plan("proportions", "power", baseline_rate=0.2, mde=0.02, mde_type="absolute", n_per_group=n)["details"]["power"]
    assert 0.80 <= p < 0.81
    m = power.plan("proportions", "mde", baseline_rate=0.2, n_per_group=n)["details"]["effect_absolute"]
    assert abs(m - 0.02) < 0.0005


def test_unequal_groups_and_duration():
    r = power.plan("means", "sample_size", sd=10, mde=3, ratio=2, daily_users=500)["details"]
    assert abs(r["n_treatment"] - 2 * r["n_control"]) <= 1
    assert r["days_needed"] == math.ceil(r["n_total"] / 500)


def test_more_power_needs_more_users():
    a = power.plan("proportions", "sample_size", power=0.8, baseline_rate=0.1, mde=0.1)["details"]["n_total"]
    b = power.plan("proportions", "sample_size", power=0.9, baseline_rate=0.1, mde=0.1)["details"]["n_total"]
    assert b > a


@pytest.mark.parametrize("kwargs", [
    dict(kind="proportions", solve_for="sample_size", baseline_rate=1.2, mde=0.1),
    dict(kind="means", solve_for="sample_size", sd=-1, mde=1),
    dict(kind="proportions", solve_for="power", baseline_rate=0.1, mde=0.1),   # missing n
    dict(kind="proportions", solve_for="sample_size", baseline_rate=0.95, mde=0.5, mde_type="relative"),  # >100%
])
def test_bad_input_is_a_user_error(kwargs):
    with pytest.raises(UserError):
        power.plan(**kwargs)
