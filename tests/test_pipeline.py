"""Cleaning, trends and an end-to-end pipeline run on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src import config, data_cleaning, metrics, trends
from tests.conftest import LATE_LISTING, TOO_SHORT


def test_cleaning_drops_bad_rows_without_imputing():
    raw = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03", "2024-01-03", "not a date", "2024-01-05", "2024-01-08"],
        "Open": [1, 1, 1, 1, 1, 1], "High": [1, 1, 1, 1, 1, 1], "Low": [1, 1, 1, 1, 1, 1],
        "Close": [10, 11, 99, 12, None, 13], "Adj Close": [9, 10, 98, 11, 12, -1],
        "Volume": [100, 100, 100, 100, 100, 100],
    })
    clean = data_cleaning.clean_prices(raw, "TEST")
    assert list(clean.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert clean.loc["2024-01-03", "Close"] == 11  # first duplicate kept
    assert not clean.isna().any().any()


def test_crossovers_use_previous_day_only():
    close = pd.Series(np.r_[np.linspace(100, 50, 80), np.linspace(50, 150, 80)], index=pd.bdate_range("2024-01-01", periods=160))
    data = trends.crossovers(trends.moving_averages(close))
    bullish = data.index[data["Bullish Crossover"]]
    assert len(bullish) == 1
    day = data.index.get_loc(bullish[0])
    assert data["SMA20"].iloc[day - 1] <= data["SMA50"].iloc[day - 1]
    assert data["SMA20"].iloc[day] > data["SMA50"].iloc[day]
    assert not (data["Bullish Crossover"] & data["Bearish Crossover"]).any()


def test_trend_classification():
    assert trends.classify(pd.Series({"Close": 4, "SMA20": 3, "SMA50": 2, "SMA200": 1})) == "Bullish"
    assert trends.classify(pd.Series({"Close": 1, "SMA20": 2, "SMA50": 3, "SMA200": 4})) == "Bearish"
    assert trends.classify(pd.Series({"Close": 3, "SMA20": 4, "SMA50": 2, "SMA200": 1})) == "Mixed"
    assert trends.classify(pd.Series({"Close": 3, "SMA20": 4, "SMA50": 2, "SMA200": np.nan})) == "Insufficient data"


def test_pipeline_writes_every_output(pipeline_results):
    paths, _ = pipeline_results
    expected = [
        "prices.parquet", "returns.parquet", "trends.parquet", "data_quality.csv",
        "stock_summary.csv", "benchmark_summary.csv", "sector_summary.csv", "trend_summary.csv",
        "correlation_matrix.csv", "correlation_pairs.csv", "rolling_correlation_benchmark.parquet",
        "average_pairwise_correlation.parquet", "final_insights.txt",
    ]
    for name in expected:
        assert (paths.processed / name).exists(), name
    for name in ["risk_vs_return.png", "sector_median_cagr.png", "average_pairwise_correlation.png"]:
        assert (paths.figures / name).exists(), name


def test_quality_report_flags_partial_and_excluded(pipeline_results):
    paths, _ = pipeline_results
    quality = pd.read_csv(paths.processed / "data_quality.csv").set_index("Symbol")
    assert quality.loc[LATE_LISTING, "Status"] == "Partial history"
    assert quality.loc[TOO_SHORT, "Status"] == "Excluded"
    assert quality.loc[config.BENCHMARK, "Status"] == "Benchmark"
    first = next(iter(config.UNIVERSE))
    assert quality.loc[first, "Rows Dropped"] == 2  # duplicate date + missing Close


def test_summary_covers_universe_and_ranks_only_full_history(pipeline_results):
    _, results = pipeline_results
    summary = results["stock_summary"].set_index("Ticker")
    assert config.short_name(TOO_SHORT) not in summary.index
    assert len(summary) == len(config.UNIVERSE) - 1
    late = config.short_name(LATE_LISTING)
    assert pd.isna(summary.loc[late, "CAGR Rank"])
    full = summary[summary["History"] == "Full history"]
    assert sorted(full["CAGR Rank"].astype(int)) == list(range(1, len(full) + 1))


def test_returns_use_adjusted_close(pipeline_results):
    paths, results = pipeline_results
    prices = pd.read_parquet(paths.processed / "prices.parquet")
    ticker = "RELIANCE"
    frame = prices[prices["Ticker"] == ticker].sort_values("Date")
    expected = frame["Adj Close"].iloc[-1] / frame["Adj Close"].iloc[0] - 1
    assert metrics.total_return(results["returns"][ticker]) == pytest.approx(expected)
    close_based = frame["Close"].iloc[-1] / frame["Close"].iloc[0] - 1
    assert expected != pytest.approx(close_based)


def test_partial_history_benchmark_comparison_uses_same_dates(pipeline_results):
    _, results = pipeline_results
    row = results["stock_summary"].set_index("Ticker").loc[config.short_name(LATE_LISTING)]
    bench = results["returns"][config.short_name(config.BENCHMARK)]
    expected = metrics.cagr(bench.loc[row["Start Date"]:row["End Date"]])
    assert row["Benchmark CAGR (same dates)"] == pytest.approx(expected)


def test_correlation_outputs(pipeline_results):
    _, results = pipeline_results
    matrix = results["matrix"]
    assert np.allclose(np.diag(matrix), 1.0)
    pairs = results["pairs"]
    n = len(matrix)
    assert len(pairs) == n * (n - 1) // 2
    # Synthetic data has a sector factor, so same-sector pairs must correlate more.
    means = pairs.groupby("Same Sector")["Correlation"].mean()
    assert means[True] > means[False]
    market = results["market"]["Average Pairwise Correlation"].dropna()
    assert ((market > -1) & (market < 1)).all()


def test_insights_mention_key_sections(pipeline_results):
    paths, _ = pipeline_results
    text = (paths.processed / "final_insights.txt").read_text()
    for heading in ["BENCHMARK", "PERFORMANCE", "RISK", "SECTORS", "CORRELATION", "DATA NOTES"]:
        assert heading in text
    assert config.short_name(LATE_LISTING) in text


def test_ist_dates_are_not_shifted_to_previous_day():
    from src.data_collection import _extract_symbol, to_trading_date

    ist = pd.Series(pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("Asia/Kolkata"))
    assert list(to_trading_date(ist).dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]

    # Shape of a real yfinance batch result: (symbol, field) columns, tz-aware index.
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date").tz_localize("Asia/Kolkata")
    columns = pd.MultiIndex.from_product([["TCS.NS"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    downloaded = pd.DataFrame(1.0, index=index, columns=columns)
    extracted = _extract_symbol(downloaded, "TCS.NS")
    assert list(extracted["Date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert list(extracted.columns) == ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert _extract_symbol(downloaded, "INFY.NS").empty


def test_zero_volume_rows_are_dropped():
    raw = pd.DataFrame({
        "Date": ["2026-04-30", "2026-05-01", "2026-05-04"],
        "Open": [10, 10, 11], "High": [10, 10, 11], "Low": [10, 10, 11],
        "Close": [10, 10, 11], "Adj Close": [10, 10, 11], "Volume": [500, 0, 700],
    })
    clean = data_cleaning.clean_prices(raw, "TEST")
    assert list(clean.index.strftime("%Y-%m-%d")) == ["2026-04-30", "2026-05-04"]


def test_spin_off_is_back_adjusted(monkeypatch):
    monkeypatch.setitem(config.CORPORATE_ACTIONS, "DEMO.NS", [{"ex_date": "2025-10-14", "factor": 400 / 660.75, "note": "demo"}])
    raw = pd.DataFrame({
        "Date": ["2025-10-10", "2025-10-13", "2025-10-14", "2025-10-15"],
        "Open": [670, 665, 400, 395], "High": [680, 670, 401, 397], "Low": [660, 655, 390, 388],
        "Close": [678.95, 660.75, 395.45, 390.85], "Adj Close": [678.95, 660.75, 395.45, 390.85],
        "Volume": [1, 1, 1, 1],
    })
    clean = data_cleaning.clean_prices(raw, "DEMO.NS")
    returns = clean["Adj Close"].pct_change()
    assert clean.loc["2025-10-13", "Close"] == pytest.approx(400.0)
    assert returns.loc["2025-10-14"] == pytest.approx(395.45 / 400 - 1)  # genuine move, not -40%
    assert returns.loc["2025-10-13"] == pytest.approx(660.75 / 678.95 - 1)  # earlier returns unchanged
    assert clean.loc["2025-10-15", "Close"] == pytest.approx(390.85)  # post ex-date untouched
    row = data_cleaning.quality_row("DEMO.NS", 4, clean, clean.index)
    assert "adjusted for demo" in row["Notes"]
