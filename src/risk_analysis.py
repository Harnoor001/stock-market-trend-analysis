"""Calculate volatility and descriptive risk measures for Phase 5."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW = 30
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_return_data(ticker: str) -> pd.DataFrame:
    """Load and validate one Phase 4 return dataset."""
    path = PROCESSED_DATA_DIR / f"{ticker}_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Return file not found: {path}")

    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty:
        raise ValueError(f"{ticker}: return dataset is empty")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: Date is not a datetime index")
    if not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError(f"{ticker}: dates are not valid and unique")
    if "Daily Return" not in data.columns:
        raise ValueError(f"{ticker}: Daily Return column is missing")
    if not pd.api.types.is_numeric_dtype(data["Daily Return"]):
        raise ValueError(f"{ticker}: Daily Return is not numeric")
    if not pd.isna(data["Daily Return"].iloc[0]):
        raise ValueError(f"{ticker}: first Daily Return must be NaN")

    daily_returns = data["Daily Return"].iloc[1:]
    if daily_returns.isna().any():
        raise ValueError(f"{ticker}: unexpected missing daily returns found")
    if not daily_returns.notna().all():
        raise ValueError(f"{ticker}: invalid daily return values found")
    if not daily_returns.replace([float("inf"), float("-inf")], pd.NA).notna().all():
        raise ValueError(f"{ticker}: infinite daily return values found")
    if len(daily_returns) < ROLLING_WINDOW:
        raise ValueError(f"{ticker}: insufficient observations for {ROLLING_WINDOW}-day rolling volatility")
    return data


def calculate_rolling_volatility(daily_returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Calculate annualized rolling volatility using a trading-day window."""
    rolling = daily_returns.rolling(window=window, min_periods=window).std() * sqrt(TRADING_DAYS_PER_YEAR)
    if not rolling.iloc[:window].isna().all():
        raise ValueError("Initial rolling-volatility values must be NaN")
    valid_values = rolling.dropna()
    if valid_values.empty or not valid_values.notna().all():
        raise ValueError("Rolling volatility contains unexpected missing values")
    if (valid_values < 0).any() or not valid_values.replace([float("inf"), float("-inf")], pd.NA).notna().all():
        raise ValueError("Rolling volatility contains invalid values")
    return rolling


def analyze_stock(ticker: str) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Calculate volatility metrics and rolling volatility for one stock."""
    data = load_return_data(ticker)
    daily_returns = data["Daily Return"].iloc[1:]
    daily_volatility = float(daily_returns.std())
    annualized_volatility = daily_volatility * sqrt(TRADING_DAYS_PER_YEAR)
    if daily_volatility < 0 or not pd.notna(daily_volatility) or not pd.notna(annualized_volatility):
        raise ValueError(f"{ticker}: volatility calculation is invalid")

    rolling_volatility = calculate_rolling_volatility(data["Daily Return"], ROLLING_WINDOW)
    result = data.copy()
    result["Rolling Annualized Volatility"] = rolling_volatility
    summary = {
        "Stock": ticker,
        "Daily Volatility": daily_volatility,
        "Daily Volatility (%)": daily_volatility * 100,
        "Annualized Volatility": annualized_volatility,
        "Annualized Volatility (%)": annualized_volatility * 100,
        "Minimum Daily Return": float(daily_returns.min()),
        "Maximum Daily Return": float(daily_returns.max()),
        "Mean Daily Return": float(daily_returns.mean()),
        "Total Return": float(data["Cumulative Return"].iloc[-1]),
    }
    return result, summary


def build_risk_summary() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Analyze all configured stocks and return rolling data plus a summary."""
    risk_data = {}
    summary_rows = []
    for ticker in TICKERS:
        risk_data[ticker], metrics = analyze_stock(ticker)
        summary_rows.append(metrics)
    return risk_data, pd.DataFrame(summary_rows).set_index("Stock")


def save_risk_summary(summary: pd.DataFrame) -> Path:
    """Save the risk summary to the processed-data directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "risk_summary.csv"
    summary.to_csv(path, index=True)
    return path


def plot_annualized_volatility(summary: pd.DataFrame, output_path: Path | None = None):
    """Create the annualized-volatility comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    (summary["Annualized Volatility (%)"] ).plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("Annualized Volatility Comparison")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Annualized volatility (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_volatility(risk_data: dict[str, pd.DataFrame], output_path: Path | None = None):
    """Create the 30-day rolling annualized-volatility chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in TICKERS:
        ax.plot(
            risk_data[ticker].index,
            risk_data[ticker]["Rolling Annualized Volatility"] * 100,
            label=ticker,
        )
    ax.set_title("30-Day Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_vs_return(summary: pd.DataFrame, output_path: Path | None = None):
    """Create a descriptive annualized-risk versus total-return chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = summary["Annualized Volatility (%)"]
    y_values = summary["Total Return"] * 100
    ax.scatter(x_values, y_values, color="steelblue", s=80)
    for ticker in summary.index:
        ax.annotate(ticker, (x_values[ticker], y_values[ticker]), xytext=(6, 6), textcoords="offset points")
    ax.set_title("Annualized Volatility and Total Return")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Total return (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def save_risk_figures(risk_data: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    """Save all Phase 5 figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        (plot_annualized_volatility(summary, FIGURES_DIR / "risk_annualized_volatility.png"),),
        (plot_rolling_volatility(risk_data, FIGURES_DIR / "risk_rolling_volatility.png"),),
        (plot_risk_vs_return(summary, FIGURES_DIR / "risk_vs_return.png"),),
    ]
    for (figure,) in figures:
        plt.close(figure)


def independent_volatility_sanity_check(risk_data: dict[str, pd.DataFrame], summary: pd.DataFrame) -> float:
    """Compare independently recomputed annualized volatility values."""
    differences = []
    for ticker in TICKERS:
        daily_returns = risk_data[ticker]["Daily Return"].iloc[1:]
        independent_daily = daily_returns.std()
        independent_annualized = independent_daily * sqrt(TRADING_DAYS_PER_YEAR)
        differences.append(abs(independent_annualized - summary.loc[ticker, "Annualized Volatility"]))
    maximum_difference = max(differences)
    if maximum_difference > 1e-12:
        raise ValueError(f"Volatility sanity check failed: maximum difference was {maximum_difference}")
    return float(maximum_difference)


def main() -> int:
    """Run the Phase 5 risk analysis and save results."""
    try:
        risk_data, summary = build_risk_summary()
        maximum_difference = independent_volatility_sanity_check(risk_data, summary)
        summary_path = save_risk_summary(summary)
        save_risk_figures(risk_data, summary)
        print("Risk summary")
        print(summary.round(6).to_string())
        print(f"\n30-day rolling window; annualization assumption: {TRADING_DAYS_PER_YEAR} trading days")
        print(f"Independent volatility sanity check: PASS (maximum difference {maximum_difference:.3e})")
        print(f"Saved: {summary_path.relative_to(PROJECT_ROOT)}")
        return 0
    except Exception as error:
        print(f"Risk analysis failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
