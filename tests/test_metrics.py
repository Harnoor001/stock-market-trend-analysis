"""Metric functions checked against values worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from src import metrics


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def from_prices(prices):
    return series(prices).pct_change().iloc[1:]


def test_total_return_compounds():
    assert metrics.total_return(series([0.10, -0.10])) == pytest.approx(1.1 * 0.9 - 1)


def test_cagr_of_constant_daily_return_over_one_year():
    r = 0.001
    assert metrics.cagr(series([r] * 252)) == pytest.approx((1 + r) ** 252 - 1)


def test_cagr_over_two_years_is_annualised():
    returns = series([0.001] * 504)
    assert metrics.cagr(returns) == pytest.approx((1.001 ** 504) ** 0.5 - 1)


def test_annualized_volatility():
    returns = series([0.01, -0.01] * 50)
    assert metrics.annualized_volatility(returns) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


def test_sharpe_with_zero_risk_free():
    returns = series([0.02, -0.01, 0.03, -0.02])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns, risk_free=0.0) == pytest.approx(expected)


def test_daily_risk_free_compounds_back_to_annual():
    assert (1 + metrics.daily_risk_free(0.06)) ** 252 == pytest.approx(1.06)


def test_sortino_matches_hand_calculation():
    returns = series([0.02, -0.01, 0.03, -0.02])
    downside = np.sqrt((0.01 ** 2 + 0.02 ** 2) / 4)  # all four days in the denominator
    expected = 0.005 * np.sqrt(252) / downside
    assert metrics.sortino_ratio(returns, risk_free=0.0) == pytest.approx(expected)


def test_max_drawdown_with_peak_trough_and_recovery():
    returns = from_prices([100, 120, 90, 108, 120, 130])
    dd = metrics.max_drawdown(returns)
    dates = returns.index  # dates of prices 2..6
    assert dd.max_drawdown == pytest.approx(90 / 120 - 1)
    assert dd.peak_date == dates[0]  # price 120
    assert dd.trough_date == dates[1]  # price 90
    assert dd.recovery_date == dates[3]  # back to 120
    assert dd.days_to_recover == 2
    assert dd.current_drawdown == pytest.approx(0.0)


def test_drawdown_from_the_first_price_and_unrecovered():
    returns = from_prices([100, 80, 90])
    dd = metrics.max_drawdown(returns)
    assert dd.max_drawdown == pytest.approx(-0.2)
    assert dd.peak_date is None
    assert dd.recovery_date is None and dd.days_to_recover is None
    assert dd.current_drawdown == pytest.approx(-0.1)


def test_no_drawdown_when_prices_only_rise():
    dd = metrics.max_drawdown(from_prices([100, 101, 102]))
    assert dd.max_drawdown == 0.0
    assert np.isnan(metrics.calmar_ratio(from_prices([100, 101, 102])))


def test_var_and_cvar_on_known_distribution():
    returns = series(np.round(np.arange(-0.10, 0.10, 0.01), 2))  # 20 evenly spaced values
    assert metrics.value_at_risk(returns, 0.95) == pytest.approx(0.0905)  # linear interpolation
    assert metrics.conditional_value_at_risk(returns, 0.95) == pytest.approx(0.10)


def test_beta_and_alpha_of_leveraged_benchmark():
    rng = np.random.default_rng(1)
    bench = series(rng.normal(0.0005, 0.01, 500))
    asset = 2 * bench
    rf_daily = metrics.daily_risk_free(0.06)
    assert metrics.beta(asset, bench) == pytest.approx(2.0)
    # Excess returns: (2b - rf) - 2(b - rf) = rf, so alpha is the risk-free rate per day.
    assert metrics.jensens_alpha(asset, bench) == pytest.approx(rf_daily * 252)
    assert metrics.correlation(asset, bench) == pytest.approx(1.0)


def test_beta_uses_only_overlapping_dates():
    bench = series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01])
    asset = 1.5 * bench.iloc[2:]
    assert metrics.beta(asset, bench, risk_free=0.0) == pytest.approx(1.5)


def test_tracking_error_and_information_ratio():
    bench = series([0.01, -0.01, 0.02, -0.02])
    active = series([0.002, -0.001, 0.003, 0.0])
    asset = bench + active
    assert metrics.tracking_error(asset, bench) == pytest.approx(active.std(ddof=1) * np.sqrt(252))
    assert metrics.information_ratio(asset, bench) == pytest.approx(active.mean() / active.std(ddof=1) * np.sqrt(252))


def test_summarize_includes_benchmark_fields_and_excess():
    rng = np.random.default_rng(3)
    bench = series(rng.normal(0.0004, 0.01, 300))
    asset = series(rng.normal(0.0008, 0.015, 300))
    row = metrics.summarize(asset, bench)
    assert row["Observations"] == 300
    assert row["Excess CAGR"] == pytest.approx(row["CAGR"] - row["Benchmark CAGR (same dates)"])
    assert row["VaR 95% (1-day)"] <= row["CVaR 95% (1-day)"]


def test_infinite_returns_are_rejected():
    with pytest.raises(ValueError):
        metrics.total_return(series([0.01, np.inf]))
