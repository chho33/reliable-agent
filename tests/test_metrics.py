import pytest

from evals.metrics import (
    ScenarioStats, bootstrap_ci, handoff_accuracy, mean_pass_hat_k, paired_delta_ci, pass_hat_k, violation_rate,
)


def S(sid, n, c, viol=0, hand=None):
    return ScenarioStats(sid, "normal", n, c, viol, n if hand is None else hand)


def test_pass_hat_k_known_values():
    assert pass_hat_k(3, 4, 2) == pytest.approx(3 / 6)          # C(3,2)/C(4,2)
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(3, 4, 4) == 0.0
    assert pass_hat_k(2, 5, 1) == pytest.approx(0.4)             # pass^1 = c/n
    assert pass_hat_k(5, 8, 3) == pytest.approx(10 / 56)         # C(5,3)/C(8,3)
    assert pass_hat_k(0, 4, 1) == 0.0


def test_pass_hat_k_validates():
    with pytest.raises(ValueError):
        pass_hat_k(1, 3, 4)
    with pytest.raises(ValueError):
        pass_hat_k(5, 4, 2)
    with pytest.raises(ValueError):
        pass_hat_k(1, 4, 0)


def test_mean_over_tasks_and_pass1():
    stats = [S("a", 4, 4), S("b", 4, 3), S("c", 4, 0), S("d", 4, 1)]
    assert mean_pass_hat_k(stats, 1) == pytest.approx((1 + .75 + 0 + .25) / 4)
    assert mean_pass_hat_k(stats, 4) == pytest.approx(0.25)   # only task "a" succeeded in all 4
    assert mean_pass_hat_k(stats, 2) == pytest.approx((1 + 0.5 + 0 + 0) / 4)


def test_pass_k_is_monotone_non_increasing_in_k():
    stats = [S("a", 6, 5), S("b", 6, 3), S("c", 6, 6)]
    vals = [mean_pass_hat_k(stats, k) for k in range(1, 7)]
    assert all(x >= y for x, y in zip(vals, vals[1:]))


def test_rates():
    stats = [S("a", 4, 4, viol=1, hand=3), S("b", 4, 2, viol=0, hand=4)]
    assert violation_rate(stats) == pytest.approx(1 / 8)
    assert handoff_accuracy(stats) == pytest.approx(7 / 8)


def test_bootstrap_known_two_scenario_case():
    # scenario means are 1 and 0: resampled means are 0 (p=.25), .5 (p=.5), 1 (p=.25)
    lo, hi = bootstrap_ci([S("a", 1, 1), S("b", 1, 0)], lambda s: mean_pass_hat_k(s, 1), n_boot=4000, seed=1)
    assert (lo, hi) == (0.0, 1.0)


def test_bootstrap_degenerate_and_deterministic():
    same = [S("a", 4, 2), S("b", 4, 2), S("c", 4, 2)]
    f = lambda s: mean_pass_hat_k(s, 1)  # noqa: E731
    assert bootstrap_ci(same, f, seed=3) == (0.5, 0.5)
    mixed = [S("a", 4, 4), S("b", 4, 1), S("c", 4, 3), S("d", 4, 0), S("e", 4, 2)]
    assert bootstrap_ci(mixed, f, seed=7) == bootstrap_ci(mixed, f, seed=7)
    lo, hi = bootstrap_ci(mixed, f, seed=7)
    assert lo <= f(mixed) <= hi and 0 <= lo and hi <= 1


def test_paired_delta():
    g = [S("a", 4, 4), S("b", 4, 4), S("c", 4, 4)]
    c = [S("a", 4, 0), S("b", 4, 0), S("c", 4, 0)]
    pt, lo, hi = paired_delta_ci(g, c, lambda s: mean_pass_hat_k(s, 1), seed=0)
    assert (pt, lo, hi) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_delta_ci(g, [S("x", 4, 0)] * 3, lambda s: 0.0)
