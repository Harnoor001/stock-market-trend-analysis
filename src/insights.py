"""Generate the plain-text findings file and a few static summary figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.config import Paths


def _pct(value: float) -> str:
    return f"{value:.1%}"


def build_insights(stock: pd.DataFrame, benchmark: pd.DataFrame, sectors: pd.DataFrame,
                   pairs: pd.DataFrame, trend: pd.DataFrame, quality: pd.DataFrame,
                   market_corr: pd.Series) -> str:
    full = stock[stock["History"] == "Full history"].merge(trend[["Ticker", "Trend"]], on="Ticker", how="left")
    bench = benchmark.iloc[0]

    def top(column: str, ascending: bool = False, n: int = 3) -> str:
        rows = full.sort_values(column, ascending=ascending).head(n)
        return ", ".join(f"{r['Ticker']} ({_pct(r[column])})" for _, r in rows.iterrows())

    def ratio_top(column: str, n: int = 3) -> str:
        rows = full.sort_values(column, ascending=False).head(n)
        return ", ".join(f"{r['Ticker']} ({r[column]:.2f})" for _, r in rows.iterrows())

    beat = int((full["Excess CAGR"] > 0).sum())
    partial = quality.loc[quality["Status"] == "Partial history", "Ticker"].tolist()
    excluded = quality.loc[quality["Status"].isin(["Excluded", "Missing"]), "Ticker"].tolist()
    same_sector = pairs.groupby("Same Sector")["Correlation"].mean()
    trends = full["Trend"].value_counts().to_dict()
    market = market_corr.dropna()
    deepest = full.loc[full["Max Drawdown"].idxmin()]
    unrecovered = full[full["Max Drawdown Recovery"].isna()]

    lines = [
        "NIFTY 50 MARKET ANALYSIS",
        "========================",
        "",
        f"Period: {bench['Start Date']} to {bench['End Date']} ({bench['Observations']} trading days)",
        f"Universe: {len(stock)} stocks with data; {len(full)} with full history.",
        f"Benchmark: {config.BENCHMARK_NAME}",
        f"Risk-free rate assumption: {_pct(config.RISK_FREE_RATE)} a year",
        "",
        "BENCHMARK",
        "---------",
        f"CAGR {_pct(bench['CAGR'])}, volatility {_pct(bench['Annualized Volatility'])}, "
        f"Sharpe {bench['Sharpe Ratio']:.2f}, max drawdown {_pct(bench['Max Drawdown'])}",
        "",
        "PERFORMANCE (full-history stocks)",
        "---------------------------------",
        f"Highest CAGR: {top('CAGR')}",
        f"Lowest CAGR: {top('CAGR', ascending=True)}",
        f"Best risk-adjusted (Sharpe): {ratio_top('Sharpe Ratio')}",
        f"{beat} of {len(full)} stocks beat the benchmark's CAGR over the same dates.",
        "",
        "RISK",
        "----",
        f"Most volatile: {top('Annualized Volatility')}",
        f"Least volatile: {top('Annualized Volatility', ascending=True)}",
        f"Deepest drawdown: {deepest['Ticker']} ({_pct(deepest['Max Drawdown'])}, trough {deepest['Max Drawdown Trough']})",
        f"Stocks still below their pre-drawdown peak: {len(unrecovered)} of {len(full)}",
        f"Highest beta: {ratio_top('Beta')}",
        "",
        "SECTORS (median of full-history stocks)",
        "---------------------------------------",
        *[f"{r['Sector']}: CAGR {_pct(r['Median CAGR'])}, volatility {_pct(r['Median Volatility'])}, "
          f"Sharpe {r['Median Sharpe']:.2f} ({int(r['Stocks'])} stock{'s' if r['Stocks'] != 1 else ''})"
          for _, r in sectors.iterrows()],
        "",
        "CORRELATION",
        "-----------",
        f"Average same-sector pair correlation {same_sector.get(True, float('nan')):.2f} "
        f"vs cross-sector {same_sector.get(False, float('nan')):.2f}.",
        f"Most correlated pair: {pairs.iloc[0]['Stock A']}-{pairs.iloc[0]['Stock B']} ({pairs.iloc[0]['Correlation']:.2f})",
        f"Least correlated pair: {pairs.iloc[-1]['Stock A']}-{pairs.iloc[-1]['Stock B']} ({pairs.iloc[-1]['Correlation']:.2f})",
    ]
    if len(market):
        lines.append(
            f"Market-wide {config.ROLLING_CORRELATION_WINDOW}-day average pairwise correlation ranged "
            f"{market.min():.2f}-{market.max():.2f} (peak around {market.idxmax().date()})."
        )
    lines += [
        "",
        "TRENDS (latest observation)",
        "---------------------------",
        ", ".join(f"{label}: {count}" for label, count in sorted(trends.items())),
        "",
        "DATA NOTES",
        "----------",
        f"Partial history (excluded from rankings): {', '.join(partial) or 'None'}",
        f"Excluded or missing: {', '.join(excluded) or 'None'}",
        "Universe is today's Nifty 50, so it is survivorship-biased: stocks that left the index are not included.",
        "These are descriptive historical comparisons, not forecasts or investment advice.",
    ]
    return "\n".join(lines) + "\n"


def save_figures(stock: pd.DataFrame, sectors: pd.DataFrame, market_corr: pd.Series, paths: Paths) -> None:
    full = stock[stock["History"] == "Full history"]

    fig, ax = plt.subplots(figsize=(11, 7))
    for sector, group in full.groupby("Sector"):
        ax.scatter(group["Annualized Volatility"] * 100, group["CAGR"] * 100, label=sector, s=55)
        for _, row in group.iterrows():
            ax.annotate(row["Ticker"], (row["Annualized Volatility"] * 100, row["CAGR"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Risk vs return by stock", xlabel="Annualized volatility (%)", ylabel="CAGR (%)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(paths.figures / "risk_vs_return.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = sectors.sort_values("Median CAGR")
    ax.barh(ordered["Sector"], ordered["Median CAGR"] * 100, color="steelblue")
    ax.set(title="Median CAGR by sector", xlabel="CAGR (%)")
    fig.tight_layout(); fig.savefig(paths.figures / "sector_median_cagr.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(market_corr.index, market_corr.values, color="darkred")
    ax.set(title=f"{config.ROLLING_CORRELATION_WINDOW}-day average pairwise correlation across the universe", ylabel="Correlation")
    fig.tight_layout(); fig.savefig(paths.figures / "average_pairwise_correlation.png", dpi=150); plt.close(fig)


def run(results: dict, quality: pd.DataFrame, paths: Paths = Paths(), figures: bool = True) -> str:
    market = results["market"]["Average Pairwise Correlation"]
    text = build_insights(results["stock_summary"], results["benchmark_summary"], results["sector_summary"],
                          results["pairs"], results["trend_summary"], quality, market)
    (paths.processed / "final_insights.txt").write_text(text, encoding="utf-8")
    if figures:
        save_figures(results["stock_summary"], results["sector_summary"], market, paths)
    print("Insights written to data/processed/final_insights.txt")
    return text
