"""Return correlations: full matrix, ranked pairs, rolling correlation with the
benchmark, and the market-wide average pairwise correlation over time."""

from itertools import combinations

import numpy as np
import pandas as pd

from src import config
from src.config import Paths

MIN_OVERLAP = config.MIN_OBSERVATIONS  # pairs need ~1 year of shared days


def correlation_matrix(stock_returns: pd.DataFrame, sector_order: list[str] | None = None) -> pd.DataFrame:
    """Pairwise Pearson correlation on overlapping dates, ordered by sector so blocks are visible."""
    enough = stock_returns.columns[stock_returns.notna().sum() >= MIN_OVERLAP]
    matrix = stock_returns[enough].corr(method="pearson", min_periods=MIN_OVERLAP)
    if sector_order:
        order = [ticker for ticker in sector_order if ticker in matrix.columns]
        matrix = matrix.loc[order, order]
    return matrix


def ranked_pairs(matrix: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"Stock A": a, "Stock B": b, "Sector A": sectors.get(a), "Sector B": sectors.get(b),
         "Same Sector": sectors.get(a) == sectors.get(b), "Correlation": matrix.loc[a, b]}
        for a, b in combinations(matrix.columns, 2)
        if pd.notna(matrix.loc[a, b])
    ]
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False).reset_index(drop=True)


def rolling_with_benchmark(returns: pd.DataFrame, window: int = config.ROLLING_CORRELATION_WINDOW) -> pd.DataFrame:
    benchmark = returns[config.short_name(config.BENCHMARK)]
    stocks = returns.drop(columns=config.short_name(config.BENCHMARK))
    return stocks.rolling(window, min_periods=window).corr(benchmark)


def average_pairwise_correlation(stock_returns: pd.DataFrame, window: int = config.ROLLING_CORRELATION_WINDOW) -> pd.Series:
    """Mean off-diagonal correlation across all stocks in each trailing window.

    A market-wide 'how much is everything moving together' gauge. Windows are
    computed on stocks with complete data inside the window.
    """
    values = stock_returns.to_numpy()
    out = np.full(len(stock_returns), np.nan)
    for end in range(window, len(stock_returns) + 1):
        block = values[end - window:end]
        block = block[:, ~np.isnan(block).any(axis=0)]
        if block.shape[1] < 2:
            continue
        corr = np.corrcoef(block, rowvar=False)
        n = corr.shape[0]
        out[end - 1] = (corr.sum() - n) / (n * (n - 1))
    return pd.Series(out, index=stock_returns.index, name="Average Pairwise Correlation")


def validate(matrix: pd.DataFrame, stock_returns: pd.DataFrame) -> float:
    values = matrix.to_numpy()
    if not np.allclose(np.diag(values), 1.0):
        raise ValueError("correlation diagonal is not 1")
    if not np.allclose(values, values.T, equal_nan=True):
        raise ValueError("correlation matrix is not symmetric")
    finite = values[np.isfinite(values)]
    if ((finite < -1 - 1e-12) | (finite > 1 + 1e-12)).any():
        raise ValueError("correlation outside [-1, 1]")
    # Independent recomputation for the first pair with full overlap.
    a, b = matrix.columns[:2]
    pair = stock_returns[[a, b]].dropna()
    x, y = pair[a].to_numpy(), pair[b].to_numpy()
    manual = ((x - x.mean()) * (y - y.mean())).sum() / np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum())
    difference = abs(manual - matrix.loc[a, b])
    if difference > 1e-10:
        raise ValueError(f"correlation recomputation mismatch {difference:.3e}")
    return float(difference)


def run(returns: pd.DataFrame, stock_summary: pd.DataFrame, paths: Paths = Paths()) -> dict[str, pd.DataFrame]:
    benchmark = config.short_name(config.BENCHMARK)
    stock_returns = returns.drop(columns=benchmark)
    ordered = stock_summary.sort_values(["Sector", "Ticker"])
    sectors = dict(zip(stock_summary["Ticker"], stock_summary["Sector"]))

    matrix = correlation_matrix(stock_returns, list(ordered["Ticker"]))
    difference = validate(matrix, stock_returns)
    pairs = ranked_pairs(matrix, sectors)
    rolling = rolling_with_benchmark(returns)
    market = average_pairwise_correlation(stock_returns).to_frame()

    matrix.to_csv(paths.processed / "correlation_matrix.csv")
    pairs.to_csv(paths.processed / "correlation_pairs.csv", index=False)
    rolling.to_parquet(paths.processed / "rolling_correlation_benchmark.parquet")
    market.to_parquet(paths.processed / "average_pairwise_correlation.parquet")
    same = pairs.groupby("Same Sector")["Correlation"].mean()
    print(
        f"Correlation: {len(pairs)} pairs; same-sector mean {same.get(True, np.nan):.2f} vs cross-sector "
        f"{same.get(False, np.nan):.2f}; recomputation PASS (diff {difference:.1e})"
    )
    return {"matrix": matrix, "pairs": pairs, "rolling": rolling, "market": market}
