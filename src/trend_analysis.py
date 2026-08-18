"""Calculate moving averages, trend classifications, and descriptive crossovers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
SMA_WINDOWS = {"SMA20": 20, "SMA50": 50, "SMA200": 200}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_cleaned_data(ticker: str) -> pd.DataFrame:
    """Load and validate one cleaned Phase 3 dataset."""
    path = PROCESSED_DATA_DIR / f"{ticker}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {path}")

    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if data.empty:
        raise ValueError(f"{ticker}: dataset is empty")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: Date is not a datetime index")
    if not data.index.is_monotonic_increasing or data.index.duplicated().any():
        raise ValueError(f"{ticker}: dates are not sorted and unique")
    if "Close" not in data.columns:
        raise ValueError(f"{ticker}: Close column is missing")
    if not pd.api.types.is_numeric_dtype(data["Close"]):
        raise ValueError(f"{ticker}: Close is not numeric")
    if not data["Close"].notna().all() or not data["Close"].replace([float("inf"), float("-inf")], pd.NA).notna().all():
        raise ValueError(f"{ticker}: Close contains invalid values")
    if len(data) < max(SMA_WINDOWS.values()):
        raise ValueError(f"{ticker}: insufficient observations for SMA200")
    return data


def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate non-centered simple moving averages from Close prices."""
    result = data[["Close"]].copy()
    for column, window in SMA_WINDOWS.items():
        result[column] = result["Close"].rolling(window=window, center=False).mean()

    for column, window in SMA_WINDOWS.items():
        if result[column].iloc[: window - 1].notna().any():
            raise ValueError(f"{column}: values were created before the full window")
        if result[column].iloc[window - 1 :].isna().any():
            raise ValueError(f"{column}: unexpected missing values after the initial window")
        valid_values = result[column].iloc[window - 1 :]
        if not valid_values.replace([float("inf"), float("-inf")], pd.NA).notna().all():
            raise ValueError(f"{column}: invalid moving-average values")
    return result


def detect_crossovers(data: pd.DataFrame) -> pd.DataFrame:
    """Mark descriptive SMA20/SMA50 crossover events without look-ahead."""
    result = data.copy()
    available = result["SMA20"].notna() & result["SMA50"].notna()
    previous_available = available.shift(1, fill_value=False)
    previous_sma20 = result["SMA20"].shift(1)
    previous_sma50 = result["SMA50"].shift(1)

    result["Bullish Crossover"] = (
        available
        & previous_available
        & (previous_sma20 <= previous_sma50)
        & (result["SMA20"] > result["SMA50"])
    )
    result["Bearish Crossover"] = (
        available
        & previous_available
        & (previous_sma20 >= previous_sma50)
        & (result["SMA20"] < result["SMA50"])
    )
    result["Crossover Type"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result.loc[result["Bullish Crossover"], "Crossover Type"] = "Bullish"
    result.loc[result["Bearish Crossover"], "Crossover Type"] = "Bearish"
    return result


def classify_trend(latest: pd.Series) -> str:
    """Classify the latest price relationship as Bullish, Bearish, or Mixed."""
    if latest["Close"] > latest["SMA20"] > latest["SMA50"] > latest["SMA200"]:
        return "Bullish"
    if latest["Close"] < latest["SMA20"] < latest["SMA50"] < latest["SMA200"]:
        return "Bearish"
    return "Mixed"


def build_trend_summary(ticker: str, data: pd.DataFrame) -> dict[str, float | int | str]:
    """Build the latest trend and crossover summary for one stock."""
    latest = data.iloc[-1]
    crossovers = data[data["Crossover Type"].notna()]
    latest_crossover = crossovers.iloc[-1] if not crossovers.empty else None
    latest_crossover_date = crossovers.index[-1].strftime("%Y-%m-%d") if latest_crossover is not None else ""
    latest_crossover_type = str(latest_crossover["Crossover Type"]) if latest_crossover is not None else ""

    return {
        "Stock": ticker,
        "Latest Close": float(latest["Close"]),
        "SMA20": float(latest["SMA20"]),
        "SMA50": float(latest["SMA50"]),
        "SMA200": float(latest["SMA200"]),
        "Close vs SMA200 (%)": float((latest["Close"] / latest["SMA200"] - 1) * 100),
        "Trend Classification": classify_trend(latest),
        "Bullish Crossovers": int(data["Bullish Crossover"].sum()),
        "Bearish Crossovers": int(data["Bearish Crossover"].sum()),
        "Latest Crossover Date": latest_crossover_date,
        "Latest Crossover Type": latest_crossover_type,
    }


def analyze_stock(ticker: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Calculate moving averages, crossovers, and summary for one stock."""
    data = calculate_moving_averages(load_cleaned_data(ticker))
    data = detect_crossovers(data)
    validate_crossover_sanity(data)
    return data, build_trend_summary(ticker, data)


def build_trend_analysis() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Analyze all configured stocks and return trend data plus summary."""
    trend_data = {}
    summary_rows = []
    for ticker in TICKERS:
        trend_data[ticker], summary = analyze_stock(ticker)
        summary_rows.append(summary)
    return trend_data, pd.DataFrame(summary_rows).set_index("Stock")


def validate_crossover_sanity(data: pd.DataFrame) -> None:
    """Verify each crossover against its immediately preceding observation."""
    for position in range(1, len(data)):
        previous = data.iloc[position - 1]
        current = data.iloc[position]
        if current["Bullish Crossover"]:
            if not (previous["SMA20"] <= previous["SMA50"] and current["SMA20"] > current["SMA50"]):
                raise ValueError("Invalid bullish crossover detected")
        if current["Bearish Crossover"]:
            if not (previous["SMA20"] >= previous["SMA50"] and current["SMA20"] < current["SMA50"]):
                raise ValueError("Invalid bearish crossover detected")
    if (data["Bullish Crossover"] & data["Bearish Crossover"]).any():
        raise ValueError("A date cannot contain both crossover types")


def independent_sma_sanity_check(trend_data: dict[str, pd.DataFrame]) -> float:
    """Compare latest rolling means against independently calculated means."""
    differences = []
    for data in trend_data.values():
        close = data["Close"]
        latest_position = len(close)
        for column, window in SMA_WINDOWS.items():
            independent_value = close.iloc[latest_position - window : latest_position].mean()
            differences.append(abs(independent_value - data[column].iloc[-1]))
    return float(max(differences))


def independent_crossover_sanity_check(trend_data: dict[str, pd.DataFrame]) -> None:
    """Independently verify all detected crossover relationships."""
    for data in trend_data.values():
        validate_crossover_sanity(data)


def save_trend_outputs(trend_data: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    """Save trend datasets and the trend summary without overwriting prior outputs."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DATA_DIR / "trend_summary.csv", index=True)
    for ticker, data in trend_data.items():
        data.to_csv(PROCESSED_DATA_DIR / f"{ticker}_trends.csv", index=True)


def plot_moving_averages(ticker: str, data: pd.DataFrame, output_path: Path | None = None):
    """Plot Close and all three simple moving averages."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Close", linewidth=1.2)
    ax.plot(data.index, data["SMA20"], label="SMA20")
    ax.plot(data.index, data["SMA50"], label="SMA50")
    ax.plot(data.index, data["SMA200"], label="SMA200")
    ax.set_title(f"{ticker} Close Price and Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_crossovers(ticker: str, data: pd.DataFrame, output_path: Path | None = None):
    """Plot Close, SMA20, SMA50, and descriptive crossover markers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Close", linewidth=1.1)
    ax.plot(data.index, data["SMA20"], label="SMA20")
    ax.plot(data.index, data["SMA50"], label="SMA50")
    bullish = data[data["Bullish Crossover"]]
    bearish = data[data["Bearish Crossover"]]
    ax.scatter(bullish.index, bullish["Close"], marker="^", color="green", label="Bullish crossover", zorder=3)
    ax.scatter(bearish.index, bearish["Close"], marker="v", color="red", label="Bearish crossover", zorder=3)
    ax.set_title(f"{ticker} SMA20/SMA50 Crossovers")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def save_trend_figures(trend_data: dict[str, pd.DataFrame]) -> None:
    """Save one moving-average and one crossover figure per stock."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, data in trend_data.items():
        moving_average_figure = plot_moving_averages(
            ticker, data, FIGURES_DIR / f"trend_{ticker}_moving_averages.png"
        )
        crossover_figure = plot_crossovers(ticker, data, FIGURES_DIR / f"trend_{ticker}_crossovers.png")
        plt.close(moving_average_figure)
        plt.close(crossover_figure)


def main() -> int:
    """Run Phase 6 trend analysis and save all outputs."""
    try:
        trend_data, summary = build_trend_analysis()
        independent_crossover_sanity_check(trend_data)
        maximum_sma_difference = independent_sma_sanity_check(trend_data)
        if maximum_sma_difference > 1e-10:
            raise ValueError(f"SMA sanity check failed: maximum difference {maximum_sma_difference}")
        save_trend_outputs(trend_data, summary)
        save_trend_figures(trend_data)
        print("Trend summary")
        print(summary.round(6).to_string())
        print(f"\nSMA sanity check: PASS (maximum difference {maximum_sma_difference:.3e})")
        print("Crossover sanity check: PASS")
        print("Saved: data/processed/trend_summary.csv")
        return 0
    except Exception as error:
        print(f"Trend analysis failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
