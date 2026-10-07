"""Daily returns from adjusted closes, and per-stock / benchmark / sector summaries."""

import numpy as np
import pandas as pd

from src import config, metrics
from src.config import Paths


def metadata() -> pd.DataFrame:
    """Ticker, Yahoo symbol, company name and sector for the configured universe."""
    return pd.DataFrame(
        [(config.short_name(s), s, name, sector) for s, (name, sector) in config.UNIVERSE.items()],
        columns=["Ticker", "Symbol", "Name", "Sector"],
    )


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide Date x Ticker table of simple daily returns from adjusted closes.

    Each ticker's returns are computed on its own trading dates *before*
    aligning, so a missing day (e.g. a suspension) produces one multi-day return
    rather than being silently treated as zero.
    """
    series = {}
    for ticker, frame in prices.groupby("Ticker"):
        adjusted = frame.set_index("Date")["Adj Close"].sort_index()
        series[ticker] = adjusted.pct_change().iloc[1:]
    returns = pd.DataFrame(series).sort_index()
    returns.index.name = "Date"
    # NaN is expected (dates before a late listing); infinity never is.
    if np.isinf(returns.to_numpy()).any():
        raise ValueError("returns contain infinite values")
    return returns


def validate_returns(prices: pd.DataFrame, returns: pd.DataFrame, tolerance: float = 1e-9) -> float:
    """Independent check: compounded daily returns must equal last/first adjusted close - 1."""
    worst = 0.0
    for ticker, frame in prices.groupby("Ticker"):
        adjusted = frame.sort_values("Date")["Adj Close"]
        direct = adjusted.iloc[-1] / adjusted.iloc[0] - 1
        compounded = metrics.total_return(returns[ticker])
        worst = max(worst, abs(direct - compounded) / max(1.0, abs(direct)))
    if worst > tolerance:
        raise ValueError(f"return reconciliation failed: max relative difference {worst:.3e}")
    return worst


def build_stock_summary(returns: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    benchmark = returns[config.short_name(config.BENCHMARK)]
    status = quality.set_index("Ticker")["Status"]
    rows = []
    for row in metadata().itertuples(index=False):
        if row.Ticker not in returns.columns:
            continue
        summary = metrics.summarize(returns[row.Ticker], benchmark)
        rows.append({
            "Ticker": row.Ticker, "Name": row.Name, "Sector": row.Sector,
            "History": status.get(row.Ticker, ""), **summary,
        })
    table = pd.DataFrame(rows)
    full = table["History"] == "Full history"
    # Ranks only among full-history stocks, so different time windows are never ranked against each other.
    table["CAGR Rank"] = table["CAGR"].where(full).rank(ascending=False, method="min").astype("Int64")
    table["Sharpe Rank"] = table["Sharpe Ratio"].where(full).rank(ascending=False, method="min").astype("Int64")
    table["Volatility Rank"] = table["Annualized Volatility"].where(full).rank(ascending=False, method="min").astype("Int64")
    return table


def build_benchmark_summary(returns: pd.DataFrame) -> pd.DataFrame:
    ticker = config.short_name(config.BENCHMARK)
    return pd.DataFrame([{"Ticker": ticker, "Name": config.BENCHMARK_NAME, **metrics.summarize(returns[ticker])}])


def build_sector_summary(stock_summary: pd.DataFrame) -> pd.DataFrame:
    """Median metrics per sector over full-history stocks (medians resist one outlier stock)."""
    full = stock_summary[stock_summary["History"] == "Full history"]
    grouped = full.groupby("Sector")
    table = grouped.agg(
        Stocks=("Ticker", "count"),
        **{
            "Median CAGR": ("CAGR", "median"),
            "Median Volatility": ("Annualized Volatility", "median"),
            "Median Sharpe": ("Sharpe Ratio", "median"),
            "Median Max Drawdown": ("Max Drawdown", "median"),
            "Median Beta": ("Beta", "median"),
            "Share Beating Benchmark": ("Excess CAGR", lambda values: float((values > 0).mean())),
        },
    )
    return table.sort_values("Median CAGR", ascending=False).reset_index()


def run(prices: pd.DataFrame, quality: pd.DataFrame, paths: Paths = Paths()) -> dict[str, pd.DataFrame]:
    returns = build_returns(prices)
    difference = validate_returns(prices, returns)
    stock_summary = build_stock_summary(returns, quality)
    benchmark_summary = build_benchmark_summary(returns)
    sector_summary = build_sector_summary(stock_summary)

    returns.to_parquet(paths.processed / "returns.parquet")
    stock_summary.to_csv(paths.processed / "stock_summary.csv", index=False)
    benchmark_summary.to_csv(paths.processed / "benchmark_summary.csv", index=False)
    sector_summary.to_csv(paths.processed / "sector_summary.csv", index=False)
    print(f"Performance: {len(stock_summary)} stocks summarised; return reconciliation PASS (max diff {difference:.1e})")
    return {"returns": returns, "stock_summary": stock_summary, "benchmark_summary": benchmark_summary, "sector_summary": sector_summary}
