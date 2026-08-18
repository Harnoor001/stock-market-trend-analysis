"""Integrate the completed Phase 4–7 results for the final project view."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the existing Phase 4–7 summary files."""
    performance = pd.read_csv(PROCESSED_DATA_DIR / "performance_summary.csv")
    risk = pd.read_csv(PROCESSED_DATA_DIR / "risk_summary.csv")
    trend = pd.read_csv(PROCESSED_DATA_DIR / "trend_summary.csv")
    pairs = pd.read_csv(PROCESSED_DATA_DIR / "correlation_pairs.csv")
    for name, data in [("performance", performance), ("risk", risk), ("trend", trend)]:
        if set(data["Stock"]) != set(TICKERS) or len(data) != len(TICKERS):
            raise ValueError(f"{name}: expected exactly one row for each stock")
    if len(pairs) != 10 or pairs[["Stock A", "Stock B"]].duplicated().any():
        raise ValueError("Correlation pairs must contain 10 unique pairs")
    return performance, risk, trend, pairs


def build_final_dataset(performance: pd.DataFrame, risk: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """Combine summary metrics and add independent descriptive ranks."""
    risk_columns = ["Stock", "Daily Volatility", "Annualized Volatility"]
    trend_columns = [
        "Stock", "Latest Close", "SMA20", "SMA50", "SMA200", "Close vs SMA200 (%)",
        "Trend Classification", "Bullish Crossovers", "Bearish Crossovers",
        "Latest Crossover Date", "Latest Crossover Type",
    ]
    result = performance.merge(risk[risk_columns], on="Stock", validate="one_to_one")
    result = result.merge(trend[trend_columns], on="Stock", validate="one_to_one")
    result["Performance Rank"] = result["Total Price Return"].rank(method="min", ascending=False).astype(int)
    result["Risk Rank"] = result["Annualized Volatility"].rank(method="min", ascending=False).astype(int)
    result = result.set_index("Stock").reindex(TICKERS).reset_index()
    required = ["Stock", "Total Price Return", "Annualized Volatility", "Trend Classification"]
    if result[required].isna().any().any() or result["Stock"].duplicated().any():
        raise ValueError("Final dataset contains missing required values or duplicate stocks")
    return result


def save_figures(final_data: pd.DataFrame) -> None:
    """Save final performance, risk, and risk-performance figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = final_data["Stock"]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, final_data["Total Price Return"] * 100, color="steelblue")
    axis.set_title("Total Price Return Comparison")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Total price return (%)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "final_performance_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, final_data["Annualized Volatility"] * 100, color="darkorange")
    axis.set_title("Annualized Volatility Comparison")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Annualized volatility (%)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "final_risk_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(final_data["Annualized Volatility"] * 100, final_data["Total Price Return"] * 100, s=80, color="teal")
    for _, row in final_data.iterrows():
        axis.annotate(row["Stock"], (row["Annualized Volatility"] * 100, row["Total Price Return"] * 100), xytext=(5, 5), textcoords="offset points")
    axis.set_title("Risk vs Historical Performance")
    axis.set_xlabel("Annualized volatility (%)")
    axis.set_ylabel("Total price return (%)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "final_risk_vs_performance.png", dpi=150)
    plt.close(figure)


def analysis_period() -> tuple[str, str]:
    """Read the historical period from an existing Phase 4 return file."""
    data = pd.read_csv(PROCESSED_DATA_DIR / "AAPL_returns.csv", parse_dates=["Date"])
    return data["Date"].min().strftime("%Y-%m-%d"), data["Date"].max().strftime("%Y-%m-%d")


def create_insights(final_data: pd.DataFrame, pairs: pd.DataFrame) -> str:
    """Generate concise observations directly from the integrated data."""
    highest_return = final_data.loc[final_data["Total Price Return"].idxmax()]
    lowest_return = final_data.loc[final_data["Total Price Return"].idxmin()]
    highest_risk = final_data.loc[final_data["Annualized Volatility"].idxmax()]
    lowest_risk = final_data.loc[final_data["Annualized Volatility"].idxmin()]
    best_day = final_data.loc[final_data["Best Single-Day Return"].idxmax()]
    worst_day = final_data.loc[final_data["Worst Single-Day Return"].idxmin()]
    highest_pair = pairs.iloc[0]
    lowest_pair = pairs.iloc[-1]
    high_region = final_data[(final_data["Total Price Return"] >= final_data["Total Price Return"].median()) & (final_data["Annualized Volatility"] >= final_data["Annualized Volatility"].median())]["Stock"].tolist()
    lower_volatility = final_data[final_data["Annualized Volatility"] < final_data["Annualized Volatility"].median()]["Stock"].tolist()
    trends = {classification: final_data.loc[final_data["Trend Classification"] == classification, "Stock"].tolist() for classification in ["Bullish", "Mixed", "Bearish"]}
    start, end = analysis_period()

    def names(values: list[str]) -> str:
        return ", ".join(values) if values else "None"

    lines = [
        "STOCK MARKET TREND ANALYSIS", "===========================", "",
        f"Analysis Period: {start} to {end}", "",
        "PERFORMANCE", "-----------",
        f"Highest total return: {highest_return['Stock']} ({highest_return['Total Price Return']:.2%})",
        f"Lowest total return: {lowest_return['Stock']} ({lowest_return['Total Price Return']:.2%})", "",
        "RISK", "----",
        f"Highest annualized volatility: {highest_risk['Stock']} ({highest_risk['Annualized Volatility']:.2%})",
        f"Lowest annualized volatility: {lowest_risk['Stock']} ({lowest_risk['Annualized Volatility']:.2%})", "",
        "SINGLE-DAY MOVEMENTS", "--------------------",
        f"Highest single-day return: {best_day['Stock']} ({best_day['Best Single-Day Return']:.2%})",
        f"Largest single-day loss: {worst_day['Stock']} ({worst_day['Worst Single-Day Return']:.2%})", "",
        "TRENDS", "------", f"Bullish: {names(trends['Bullish'])}", f"Mixed: {names(trends['Mixed'])}", f"Bearish: {names(trends['Bearish'])}", "",
        "CORRELATION", "-----------",
        f"Highest correlated pair: {highest_pair['Stock A']}-{highest_pair['Stock B']} ({highest_pair['Correlation']:.4f})",
        f"Lowest correlated pair: {lowest_pair['Stock A']}-{lowest_pair['Stock B']} ({lowest_pair['Correlation']:.4f})", "",
        "KEY OBSERVATIONS", "----------------",
        f"1. {highest_return['Stock']} had the highest historical total price return, while {lowest_return['Stock']} had the lowest.",
        f"2. {highest_risk['Stock']} had the highest annualized volatility, and {lowest_risk['Stock']} had the lowest.",
        f"3. Stocks in the at-or-above-median return and volatility region: {names(high_region)}.",
        f"4. Stocks below the median annualized volatility: {names(lower_volatility)}.",
        "5. These are descriptive historical comparisons; they do not establish future outcomes or investment suitability.",
    ]
    return "\n".join(lines) + "\n"


def validate(final_data: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Validate the final integrated dataset and its required inputs."""
    if list(final_data["Stock"]) != TICKERS or len(final_data) != 5:
        raise ValueError("Final dataset must contain exactly the five stocks")
    numeric = final_data.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Final numeric fields contain non-finite values")
    if set(pairs["Stock A"]).union(pairs["Stock B"]) != set(TICKERS):
        raise ValueError("Correlation pairs do not contain all stocks")
    if set(final_data["Performance Rank"]) != set(range(1, 6)) or set(final_data["Risk Rank"]) != set(range(1, 6)):
        raise ValueError("Ranks are not valid")


def run_analysis() -> tuple[pd.DataFrame, str]:
    """Run the final integration and save all Phase 8 outputs."""
    performance, risk, trend, pairs = load_inputs()
    final_data = build_final_dataset(performance, risk, trend)
    validate(final_data, pairs)
    insights = create_insights(final_data, pairs)
    final_data.to_csv(PROCESSED_DATA_DIR / "final_stock_analysis.csv", index=False)
    (PROCESSED_DATA_DIR / "final_insights.txt").write_text(insights, encoding="utf-8")
    save_figures(final_data)
    print(final_data[["Stock", "Total Price Return", "Annualized Volatility", "Trend Classification", "Performance Rank", "Risk Rank"]].to_string(index=False))
    print("\nFinal integration validation: PASS")
    print("Saved: data/processed/final_stock_analysis.csv and data/processed/final_insights.txt")
    return final_data, insights


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as error:
        print(f"Final analysis failed: {error}")
        raise SystemExit(1)
