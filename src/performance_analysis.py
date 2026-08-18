"""Calculate stock returns and performance metrics for Phase 4."""

from pathlib import Path

import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_cleaned_data(ticker: str) -> pd.DataFrame:
    """Load and validate one cleaned stock dataset."""
    path = PROCESSED_DATA_DIR / f"{ticker}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty:
        raise ValueError(f"{ticker}: dataset is empty")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: Date is not a datetime index")
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates are not sorted chronologically")
    if data.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate dates found")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{ticker}: missing columns {missing_columns}")
    if not pd.api.types.is_numeric_dtype(data["Close"]):
        raise ValueError(f"{ticker}: Close column is not numeric")
    if data["Close"].isna().any() or not data["Close"].notna().all():
        raise ValueError(f"{ticker}: Close column contains missing values")
    return data


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Add daily returns and cumulative returns without modifying the input."""
    result = data.copy()
    result["Daily Return"] = result["Close"].pct_change()

    # The first day has no prior observation. It remains NaN in Daily Return;
    # a zero is used only as the starting point for cumulative growth.
    result["Cumulative Growth"] = (1 + result["Daily Return"].fillna(0)).cumprod()
    result["Cumulative Return"] = result["Cumulative Growth"] - 1

    daily_returns = result["Daily Return"].dropna()
    if result["Daily Return"].iloc[0] is not pd.NA and not pd.isna(result["Daily Return"].iloc[0]):
        raise ValueError("The first daily return must be missing")
    if not daily_returns.notna().all() or not daily_returns.map(pd.api.types.is_number).all():
        raise ValueError("Daily returns contain unexpected missing or non-numeric values")
    finite_daily_returns = daily_returns.replace([float("inf"), float("-inf")], pd.NA).notna().all()
    finite_cumulative_values = result[["Cumulative Growth", "Cumulative Return"]].replace(
        [float("inf"), float("-inf")], pd.NA
    ).notna().all().all()
    if not finite_daily_returns or not finite_cumulative_values:
        raise ValueError("Return calculations contain infinite values")
    return result


def calculate_performance_metrics(ticker: str, data: pd.DataFrame) -> dict[str, float | str]:
    """Calculate summary performance metrics for one stock."""
    daily_returns = data["Daily Return"].dropna()
    initial_price = float(data["Close"].iloc[0])
    final_price = float(data["Close"].iloc[-1])
    total_price_change = final_price - initial_price
    total_price_return = (final_price / initial_price) - 1
    cumulative_return = float(data["Cumulative Return"].iloc[-1])

    return {
        "Stock": ticker,
        "Initial Price": initial_price,
        "Final Price": final_price,
        "Total Price Change": total_price_change,
        "Total Price Return": total_price_return,
        "Average Daily Return": float(daily_returns.mean()),
        "Best Single-Day Return": float(daily_returns.max()),
        "Worst Single-Day Return": float(daily_returns.min()),
        "Cumulative Return": cumulative_return,
    }


def build_performance_summary() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all stocks, calculate returns, and build the summary table."""
    return_data = {}
    metrics = []
    for ticker in TICKERS:
        return_data[ticker] = calculate_returns(load_cleaned_data(ticker))
        metrics.append(calculate_performance_metrics(ticker, return_data[ticker]))
    return return_data, pd.DataFrame(metrics).set_index("Stock")


def save_results(return_data: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    """Save return datasets and the performance summary to processed data."""
    summary.to_csv(PROCESSED_DATA_DIR / "performance_summary.csv", index=True)
    for ticker, data in return_data.items():
        data[["Close", "Daily Return", "Cumulative Growth", "Cumulative Return"]].to_csv(
            PROCESSED_DATA_DIR / f"{ticker}_returns.csv", index=True
        )


def validate_sanity(summary: pd.DataFrame, tolerance: float = 1e-10) -> None:
    """Confirm price return and compounded return agree."""
    differences = (summary["Total Price Return"] - summary["Cumulative Return"]).abs()
    if differences.max() > tolerance:
        raise ValueError(f"Return sanity check failed: maximum difference was {differences.max()}")


def main() -> int:
    """Run the Phase 4 performance analysis and save results."""
    try:
        return_data, summary = build_performance_summary()
        validate_sanity(summary)
        save_results(return_data, summary)

        print("Performance summary")
        display_columns = ["Initial Price", "Final Price", "Total Price Return", "Average Daily Return", "Best Single-Day Return", "Worst Single-Day Return", "Cumulative Return"]
        print(summary[display_columns].round(6).to_string())
        print("\nReturn sanity check: PASS")
        print("Saved: data/processed/performance_summary.csv")
        return 0
    except Exception as error:
        print(f"Performance analysis failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
