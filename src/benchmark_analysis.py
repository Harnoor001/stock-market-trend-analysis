"""Compare the five project stocks with the SPY market benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
ROLLING_WINDOW = 60
TRADING_DAYS_PER_YEAR = 252
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_returns(ticker: str) -> pd.DataFrame:
    """Load and validate one return dataset."""
    path = PROCESSED_DATA_DIR / f"{ticker}_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Return file not found: {path}")
    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty or not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError(f"{ticker}: invalid dates")
    if "Close" not in data.columns or "Daily Return" not in data.columns:
        raise ValueError(f"{ticker}: required return columns are missing")
    returns = data["Daily Return"].iloc[1:]
    if pd.notna(data["Daily Return"].iloc[0]) or returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError(f"{ticker}: invalid daily returns")
    return data


def build_benchmark_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the summary and 60-day stock-SPY rolling-correlation data."""
    stock_data = {ticker: load_returns(ticker) for ticker in TICKERS}
    spy_data = load_returns("SPY")
    spy_total_return = float(spy_data["Cumulative Return"].iloc[-1])
    spy_volatility = float(spy_data["Daily Return"].dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    rows = []
    rolling_series = {}
    for ticker in TICKERS:
        stock = stock_data[ticker]
        stock_total_return = float(stock["Cumulative Return"].iloc[-1])
        stock_volatility = float(stock["Daily Return"].dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        rows.append({
            "Stock": ticker,
            "Stock Total Return": stock_total_return,
            "SPY Total Return": spy_total_return,
            "Excess Total Return": stock_total_return - spy_total_return,
            "Relative Performance": (1 + stock_total_return) / (1 + spy_total_return),
            "Stock Annualized Volatility": stock_volatility,
            "SPY Annualized Volatility": spy_volatility,
            "Volatility Difference": stock_volatility - spy_volatility,
            "Volatility Ratio": stock_volatility / spy_volatility,
        })
        aligned = pd.concat([stock["Daily Return"].rename(ticker), spy_data["Daily Return"].rename("SPY")], axis=1, join="inner").sort_index()
        rolling_series[ticker] = aligned[ticker].rolling(ROLLING_WINDOW).corr(aligned["SPY"])
    rolling = pd.DataFrame(rolling_series)
    if rolling.iloc[:ROLLING_WINDOW].notna().any().any():
        raise ValueError("Rolling correlation is available before the full 60-observation window")
    valid = rolling.stack().dropna()
    if not np.isfinite(valid).all() or ((valid < -1) | (valid > 1)).any():
        raise ValueError("Rolling correlation contains invalid values")
    summary = pd.DataFrame(rows)
    summary["Latest Rolling Correlation with SPY"] = [rolling[ticker].dropna().iloc[-1] for ticker in TICKERS]
    return summary, rolling, pd.DataFrame({"SPY Total Return": [spy_total_return], "SPY Annualized Volatility": [spy_volatility]})


def validate_summary(summary: pd.DataFrame, spy_metrics: pd.DataFrame) -> float:
    """Validate formulas and return the maximum formula difference."""
    if list(summary["Stock"]) != TICKERS or len(summary) != 5:
        raise ValueError("Benchmark summary must contain exactly five stocks")
    spy_return = spy_metrics["SPY Total Return"].iloc[0]
    spy_volatility = spy_metrics["SPY Annualized Volatility"].iloc[0]
    differences = []
    for row in summary.itertuples(index=False):
        differences.extend([
            abs(row[1] - (row[2] + row[3])),
            abs(row[3] - (row[1] - spy_return)),
            abs(row[4] - ((1 + row[1]) / (1 + spy_return))),
            abs(row[7] - (row[5] - spy_volatility)),
            abs(row[8] - (row[5] / spy_volatility)),
        ])
    maximum_difference = max(differences)
    if maximum_difference > 1e-12:
        raise ValueError(f"Benchmark formula validation failed: {maximum_difference}")
    return float(maximum_difference)


def save_figures(summary: pd.DataFrame, rolling: pd.DataFrame) -> None:
    """Save benchmark comparison figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = summary["Stock"]

    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels)); width = 0.36
    axis.bar(x - width / 2, summary["Stock Total Return"] * 100, width, label="Stock")
    axis.bar(x + width / 2, [summary["SPY Total Return"].iloc[0] * 100] * len(labels), width, label="SPY", color="darkorange")
    axis.set_xticks(x, labels); axis.set_ylabel("Total return (%)"); axis.set_xlabel("Stock")
    axis.set_title("Stock Total Return vs SPY"); axis.legend(); figure.tight_layout()
    figure.savefig(FIGURES_DIR / "benchmark_performance_comparison.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, summary["Excess Total Return"] * 100, color="steelblue")
    axis.axhline(0, color="black", linewidth=0.8); axis.set_ylabel("Excess total return (%)")
    axis.set_xlabel("Stock"); axis.set_title("Excess Total Return Relative to SPY")
    figure.tight_layout(); figure.savefig(FIGURES_DIR / "benchmark_excess_return.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, summary["Stock Annualized Volatility"] * 100, width, label="Stock")
    axis.bar(x + width / 2, [summary["SPY Annualized Volatility"].iloc[0] * 100] * len(labels), width, label="SPY", color="darkorange")
    axis.set_xticks(x, labels); axis.set_ylabel("Annualized volatility (%)"); axis.set_xlabel("Stock")
    axis.set_title("Stock Annualized Volatility vs SPY"); axis.legend(); figure.tight_layout()
    figure.savefig(FIGURES_DIR / "benchmark_volatility_comparison.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for ticker in TICKERS: axis.plot(rolling.index, rolling[ticker], label=ticker)
    axis.set_ylim(-1, 1); axis.set_xlabel("Date"); axis.set_ylabel("Pearson correlation")
    axis.set_title("60-Day Rolling Correlation with SPY"); axis.legend(); figure.tight_layout()
    figure.savefig(FIGURES_DIR / "benchmark_rolling_correlation.png", dpi=150); plt.close(figure)


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the benchmark analysis and save its additive outputs."""
    summary, rolling, spy_metrics = build_benchmark_comparison()
    maximum_difference = validate_summary(summary, spy_metrics)
    summary.to_csv(PROCESSED_DATA_DIR / "benchmark_summary.csv", index=False)
    rolling.to_csv(PROCESSED_DATA_DIR / "benchmark_rolling_correlation.csv", index=True)
    save_figures(summary, rolling)
    print(summary.round(6).to_string(index=False))
    print(f"\nBenchmark formula validation: PASS (maximum difference {maximum_difference:.3e})")
    print("Rolling correlation validation: PASS (60 observations)")
    print("Saved: data/processed/benchmark_summary.csv and data/processed/benchmark_rolling_correlation.csv")
    return summary, rolling


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as error:
        print(f"Benchmark analysis failed: {error}")
        raise SystemExit(1)
