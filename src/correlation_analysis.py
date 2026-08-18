"""Analyze correlations between stock daily returns for Phase 7."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
ROLLING_WINDOW = 60
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_return_series(ticker: str) -> pd.Series:
    """Load and validate one Phase 4 daily-return series."""
    path = PROCESSED_DATA_DIR / f"{ticker}_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Return file not found for {ticker}")
    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: invalid or empty Date index")
    if not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError(f"{ticker}: dates must be sorted and unique")
    if "Daily Return" not in data.columns:
        raise ValueError(f"{ticker}: Daily Return column is missing")
    returns = pd.to_numeric(data["Daily Return"], errors="coerce")
    if returns.iloc[0] is not np.nan and not pd.isna(returns.iloc[0]):
        raise ValueError(f"{ticker}: first daily return must be NaN")
    if returns.iloc[1:].isna().any() or not np.isfinite(returns.iloc[1:]).all():
        raise ValueError(f"{ticker}: unexpected missing or infinite return")
    return returns.rename(ticker)


def build_return_matrix() -> pd.DataFrame:
    """Align all daily-return series on their Date index."""
    matrix = pd.concat([load_return_series(ticker) for ticker in TICKERS], axis=1).sort_index()
    if list(matrix.columns) != TICKERS or matrix.index.duplicated().any():
        raise ValueError("Aligned return matrix has invalid columns or duplicate dates")
    return matrix


def build_pair_rankings(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return each unique stock pair once, ranked by correlation."""
    rows = [
        {"Stock A": first, "Stock B": second, "Correlation": correlation_matrix.loc[first, second]}
        for first, second in combinations(TICKERS, 2)
    ]
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False).reset_index(drop=True)


def calculate_rolling_correlations(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate trailing rolling Pearson correlations for all unique pairs."""
    rolling = {}
    for first, second in combinations(TICKERS, 2):
        rolling[f"{first}-{second}"] = return_matrix[first].rolling(ROLLING_WINDOW).corr(return_matrix[second])
    result = pd.DataFrame(rolling, index=return_matrix.index)
    if result.iloc[:ROLLING_WINDOW].notna().any().any():
        raise ValueError("Rolling correlations became available before the 60-day window")
    valid = result.dropna(how="all").stack()
    if not np.isfinite(valid).all() or ((valid < -1) | (valid > 1)).any():
        raise ValueError("Rolling correlations contain invalid values")
    return result


def build_correlation_summary(correlation_matrix: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-period and rolling correlation for every unique pair."""
    rows = []
    for first, second in combinations(TICKERS, 2):
        pair = f"{first}-{second}"
        values = rolling[pair].dropna()
        rows.append({
            "Pair": pair,
            "Full-period Correlation": correlation_matrix.loc[first, second],
            "Minimum Rolling Correlation": values.min(),
            "Maximum Rolling Correlation": values.max(),
            "Average Rolling Correlation": values.mean(),
        })
    return pd.DataFrame(rows)


def validate_correlations(return_matrix: pd.DataFrame, correlation_matrix: pd.DataFrame, rolling: pd.DataFrame) -> float:
    """Run matrix, pairwise, and rolling sanity checks; return max pairwise difference."""
    if not np.allclose(np.diag(correlation_matrix), 1.0, atol=1e-12):
        raise ValueError("Correlation diagonal is not 1")
    if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-12):
        raise ValueError("Correlation matrix is not symmetric")
    if ((correlation_matrix < -1) | (correlation_matrix > 1)).any().any():
        raise ValueError("Correlation value is outside [-1, 1]")
    maximum_difference = 0.0
    for first, second in combinations(TICKERS, 2):
        values = return_matrix[[first, second]].dropna()
        x, y = values[first].to_numpy(), values[second].to_numpy()
        independent = ((x - x.mean()) * (y - y.mean())).sum() / np.sqrt(
            ((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum()
        )
        maximum_difference = max(maximum_difference, abs(independent - correlation_matrix.loc[first, second]))
    return maximum_difference


def save_heatmap(correlation_matrix: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(TICKERS)), TICKERS)
    axis.set_yticks(range(len(TICKERS)), TICKERS)
    axis.set_title("Daily Return Correlation Matrix")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Stock")
    for row in range(len(TICKERS)):
        for column in range(len(TICKERS)):
            axis.text(column, row, f"{correlation_matrix.iloc[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(figure)


def save_pair_comparison(pair_rankings: pd.DataFrame) -> None:
    selected = pd.concat([pair_rankings.head(3), pair_rankings.tail(3)]).drop_duplicates()
    labels = selected["Stock A"] + "-" + selected["Stock B"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(labels, selected["Correlation"], color="steelblue")
    axis.set_title("Strongest and Weakest Daily-Return Correlations")
    axis.set_xlabel("Pearson correlation")
    axis.set_ylabel("Stock pair")
    axis.axvline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "correlation_pair_comparison.png", dpi=150)
    plt.close(figure)


def save_rolling_plot(rolling: pd.DataFrame, pair_rankings: pd.DataFrame) -> None:
    selected = [
        f"{pair_rankings.iloc[0]['Stock A']}-{pair_rankings.iloc[0]['Stock B']}",
        f"{pair_rankings.iloc[-1]['Stock A']}-{pair_rankings.iloc[-1]['Stock B']}",
    ]
    figure, axis = plt.subplots(figsize=(11, 6))
    for pair in dict.fromkeys(selected):
        axis.plot(rolling.index, rolling[pair], label=pair)
    axis.set_title("60-Day Rolling Daily-Return Correlation")
    axis.set_xlabel("Date")
    axis.set_ylabel("Pearson correlation")
    axis.set_ylim(-1, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "correlation_rolling_pairs.png", dpi=150)
    plt.close(figure)


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate and save all Phase 7 correlation outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return_matrix = build_return_matrix()
    correlation_matrix = return_matrix.corr(method="pearson")
    pair_rankings = build_pair_rankings(correlation_matrix)
    rolling = calculate_rolling_correlations(return_matrix)
    summary = build_correlation_summary(correlation_matrix, rolling)
    maximum_difference = validate_correlations(return_matrix, correlation_matrix, rolling)
    correlation_matrix.to_csv(PROCESSED_DATA_DIR / "correlation_matrix.csv")
    pair_rankings.to_csv(PROCESSED_DATA_DIR / "correlation_pairs.csv", index=False)
    summary.to_csv(PROCESSED_DATA_DIR / "correlation_summary.csv", index=False)
    save_heatmap(correlation_matrix)
    save_pair_comparison(pair_rankings)
    save_rolling_plot(rolling, pair_rankings)
    print(f"Correlation sanity check: PASS (maximum difference {maximum_difference:.3e})")
    print(f"Rolling correlation sanity check: PASS ({ROLLING_WINDOW}-day trailing window)")
    print(f"Highest pair: {pair_rankings.iloc[0]['Stock A']}-{pair_rankings.iloc[0]['Stock B']} ({pair_rankings.iloc[0]['Correlation']:.6f})")
    print(f"Lowest pair: {pair_rankings.iloc[-1]['Stock A']}-{pair_rankings.iloc[-1]['Stock B']} ({pair_rankings.iloc[-1]['Correlation']:.6f})")
    return return_matrix, correlation_matrix, pair_rankings, summary


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as error:
        print(f"Correlation analysis failed: {error}")
        raise SystemExit(1)
