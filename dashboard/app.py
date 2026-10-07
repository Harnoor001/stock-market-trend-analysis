"""Read-only Streamlit dashboard for the Nifty 50 market analysis.

Reads the files written by `python main.py` from data/processed. Single-stock
metrics are recomputed for the selected date range with the same tested
functions the pipeline uses (src/metrics.py); universe and sector tables show
the full analysis period.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, metrics  # noqa: E402

PROCESSED_DIR = Path(os.environ.get("STOCK_ANALYSIS_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
BENCHMARK = config.short_name(config.BENCHMARK)
PAGES = ["Overview", "Universe", "Risk", "Trends", "Correlation", "Data & Methodology"]

st.set_page_config(page_title="Nifty 50 Market Analysis", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] { padding: 0.45rem 0.25rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
REQUIRED = [
    "returns.parquet", "trends.parquet", "stock_summary.csv", "benchmark_summary.csv",
    "sector_summary.csv", "correlation_matrix.csv", "correlation_pairs.csv",
    "rolling_correlation_benchmark.parquet", "average_pairwise_correlation.parquet",
    "data_quality.csv", "final_insights.txt",
]


@st.cache_data
def load_data(directory: str) -> dict:
    """Load every processed output once; cached per data directory."""
    folder = Path(directory)
    missing = [name for name in REQUIRED if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files: {', '.join(missing)}")
    read = lambda name, **kw: pd.read_csv(folder / name, **kw)  # noqa: E731
    return {
        "returns": pd.read_parquet(folder / "returns.parquet"),
        "trends": pd.read_parquet(folder / "trends.parquet"),
        "summary": read("stock_summary.csv"),
        "benchmark": read("benchmark_summary.csv"),
        "sectors": read("sector_summary.csv"),
        "matrix": read("correlation_matrix.csv", index_col=0),
        "pairs": read("correlation_pairs.csv"),
        "rolling": pd.read_parquet(folder / "rolling_correlation_benchmark.parquet"),
        "market": pd.read_parquet(folder / "average_pairwise_correlation.parquet"),
        "quality": read("data_quality.csv"),
        "insights": (folder / "final_insights.txt").read_text(encoding="utf-8"),
    }


@st.cache_data
def range_metrics(returns: pd.Series, benchmark: pd.Series) -> dict:
    return metrics.summarize(returns, benchmark)


def pct(value, digits: int = 1) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}%}"


def ratio(value) -> str:
    return "–" if pd.isna(value) else f"{value:.2f}"


def window(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.loc[start:end].dropna()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def indexed_chart(stock: pd.Series, bench: pd.Series, ticker: str) -> go.Figure:
    frame = pd.concat([stock.rename(ticker), bench.rename(BENCHMARK)], axis=1, join="inner").dropna()
    wealth = (1 + frame).cumprod() * 100
    figure = px.line(wealth, title="Growth of ₹100 (total return, dividends reinvested)")
    figure.update_layout(yaxis_title="Value (₹)", xaxis_title=None, legend_title_text=None, hovermode="x unified")
    return figure


def drawdown_chart(stock: pd.Series, bench: pd.Series, ticker: str) -> go.Figure:
    frame = pd.concat([metrics.drawdown_series(stock).rename(ticker), metrics.drawdown_series(bench).rename(BENCHMARK)], axis=1, sort=True)
    figure = px.area(frame, title="Drawdown from previous peak")
    figure.update_traces(stackgroup=None, fill="tozeroy")
    figure.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title_text=None, hovermode="x unified")
    return figure


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_overview(data: dict, ticker: str, start, end) -> None:
    info = data["summary"].set_index("Ticker").loc[ticker]
    st.header(f"{ticker} · {info['Name']}")
    st.caption(f"{info['Sector']} · {info['History']} · metrics below cover {start.date()} to {end.date()}")
    stock = window(data["returns"][ticker], start, end)
    bench = window(data["returns"][BENCHMARK], start, end)
    if len(stock) < 30:
        st.warning("Not enough data for this stock in the selected range.")
        return
    m = range_metrics(stock, bench)
    cols = st.columns(6)
    cols[0].metric("CAGR", pct(m["CAGR"]), f"{m['Excess CAGR']:+.1%} vs Nifty", help="Benchmark measured over the same dates")
    cols[1].metric("Volatility", pct(m["Annualized Volatility"]))
    cols[2].metric("Sharpe", ratio(m["Sharpe Ratio"]), help=f"Risk-free rate {config.RISK_FREE_RATE:.0%} a year")
    cols[3].metric("Max drawdown", pct(m["Max Drawdown"]))
    cols[4].metric("Beta", ratio(m["Beta"]))
    cols[5].metric("Alpha (annual)", pct(m["Alpha (annual)"]))
    st.plotly_chart(indexed_chart(stock, bench, ticker), width="stretch")
    st.plotly_chart(drawdown_chart(stock, bench, ticker), width="stretch")
    if m["Max Drawdown Trough"]:
        recovery = (f"recovered on {m['Max Drawdown Recovery']} after {m['Days to Recover']} trading days"
                    if m["Max Drawdown Recovery"] else "has not yet recovered to its previous peak")
        st.info(f"Worst fall in this range: {pct(m['Max Drawdown'])}, bottoming on {m['Max Drawdown Trough']}; the stock {recovery}.")


def render_universe(data: dict, sectors_selected: list[str]) -> None:
    st.header("Universe")
    bench = data["benchmark"].iloc[0]
    st.caption(
        f"Full analysis period. Benchmark CAGR {pct(bench['CAGR'])}, volatility {pct(bench['Annualized Volatility'])}, "
        f"Sharpe {ratio(bench['Sharpe Ratio'])}. Partial-history stocks are shown but not ranked."
    )
    summary = data["summary"]
    if sectors_selected:
        summary = summary[summary["Sector"].isin(sectors_selected)]

    scatter = px.scatter(
        summary, x="Annualized Volatility", y="CAGR", color="Sector", hover_name="Ticker",
        hover_data={"Name": True, "Sharpe Ratio": ":.2f", "Max Drawdown": ":.1%"},
        title="Risk vs return", text="Ticker",
    )
    scatter.add_hline(y=bench["CAGR"], line_dash="dash", annotation_text="Nifty CAGR")
    scatter.update_traces(textposition="top center", textfont_size=9)
    scatter.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(scatter, width="stretch")

    columns = ["Ticker", "Name", "Sector", "History", "CAGR", "Excess CAGR", "Annualized Volatility", "Sharpe Ratio",
               "Sortino Ratio", "Max Drawdown", "Beta", "Alpha (annual)", "VaR 95% (1-day)"]
    percent = ["CAGR", "Excess CAGR", "Annualized Volatility", "Max Drawdown", "Alpha (annual)", "VaR 95% (1-day)"]
    table = summary[columns].sort_values("Sharpe Ratio", ascending=False)
    st.dataframe(
        table.style.format({**{c: "{:.1%}" for c in percent}, "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}", "Beta": "{:.2f}"}),
        hide_index=True, width="stretch", height=420,
    )

    st.subheader("Sectors")
    sectors = data["sectors"]
    if sectors_selected:
        sectors = sectors[sectors["Sector"].isin(sectors_selected)]
    bars = px.bar(sectors.sort_values("Median CAGR"), x="Median CAGR", y="Sector", orientation="h",
                  hover_data={"Stocks": True, "Median Sharpe": ":.2f", "Share Beating Benchmark": ":.0%"},
                  title="Median CAGR by sector (full-history stocks)")
    bars.update_layout(xaxis_tickformat=".0%", yaxis_title=None)
    st.plotly_chart(bars, width="stretch")


def render_risk(data: dict, ticker: str, start, end) -> None:
    st.header(f"Risk · {ticker}")
    stock = window(data["returns"][ticker], start, end)
    bench = window(data["returns"][BENCHMARK], start, end)
    if len(stock) < 30:
        st.warning("Not enough data for this stock in the selected range.")
        return
    m = range_metrics(stock, bench)
    cols = st.columns(5)
    cols[0].metric("VaR 95% (1 day)", pct(m["VaR 95% (1-day)"], 2), help="Loss exceeded on 5% of days")
    cols[1].metric("CVaR 95% (1 day)", pct(m["CVaR 95% (1-day)"], 2), help="Average loss on those worst 5% of days")
    cols[2].metric("Sortino", ratio(m["Sortino Ratio"]))
    cols[3].metric("Calmar", ratio(m["Calmar Ratio"]))
    cols[4].metric("Tracking error", pct(m["Tracking Error"]))

    rolling = pd.concat([
        metrics.rolling_volatility(stock).rename(ticker),
        metrics.rolling_volatility(bench).rename(BENCHMARK),
    ], axis=1, sort=True)
    figure = px.line(rolling, title=f"{config.ROLLING_VOLATILITY_WINDOW}-day rolling annualized volatility")
    figure.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title_text=None, hovermode="x unified")
    st.plotly_chart(figure, width="stretch")

    histogram = px.histogram(stock.rename("Daily return"), nbins=80, title="Distribution of daily returns")
    histogram.add_vline(x=-m["VaR 95% (1-day)"], line_dash="dash", line_color="red", annotation_text="VaR 95%")
    histogram.update_layout(xaxis_tickformat=".0%", showlegend=False, yaxis_title="Days")
    st.plotly_chart(histogram, width="stretch")


def render_trends(data: dict, ticker: str, start, end) -> None:
    st.header(f"Trends · {ticker}")
    frame = data["trends"]
    frame = frame[(frame["Ticker"] == ticker) & (frame["Date"] >= start) & (frame["Date"] <= end)]
    figure = go.Figure()
    for column in ["Close", "SMA20", "SMA50", "SMA200"]:
        figure.add_trace(go.Scatter(x=frame["Date"], y=frame[column], mode="lines", name=column))
    for flag, symbol, colour, label in [("Bullish Crossover", "triangle-up", "green", "SMA20 crosses above SMA50"),
                                        ("Bearish Crossover", "triangle-down", "red", "SMA20 crosses below SMA50")]:
        events = frame[frame[flag]]
        figure.add_trace(go.Scatter(x=events["Date"], y=events["Close"], mode="markers", name=label,
                                    marker=dict(symbol=symbol, color=colour, size=10)))
    figure.update_layout(title="Price and simple moving averages", yaxis_title="Price (₹)", hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    latest = data["trends"][data["trends"]["Ticker"] == ticker].iloc[-1]
    from src.trends import classify  # local import keeps page load light
    st.caption(f"Latest trend label: **{classify(latest)}** (Close > SMA20 > SMA50 > SMA200 is Bullish, the reverse is Bearish, anything else Mixed). Descriptive only.")


def render_correlation(data: dict, ticker: str) -> None:
    st.header("Correlation")
    matrix = data["matrix"]
    heatmap = px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                        title="Daily-return correlation (ordered by sector)")
    heatmap.update_layout(height=750)
    st.plotly_chart(heatmap, width="stretch")

    pairs = data["pairs"]
    left, right = st.columns(2)
    fmt = {"Correlation": "{:.2f}"}
    shown = ["Stock A", "Stock B", "Same Sector", "Correlation"]
    left.subheader("Most correlated pairs")
    left.dataframe(pairs.head(10)[shown].style.format(fmt), hide_index=True, width="stretch")
    right.subheader("Least correlated pairs")
    right.dataframe(pairs.tail(10).iloc[::-1][shown].style.format(fmt), hide_index=True, width="stretch")
    same = pairs.groupby("Same Sector")["Correlation"].mean()
    st.caption(f"Average same-sector correlation {same.get(True, float('nan')):.2f} vs cross-sector {same.get(False, float('nan')):.2f}.")

    if ticker in data["rolling"].columns:
        figure = px.line(data["rolling"][ticker].rename("Correlation"),
                         title=f"{config.ROLLING_CORRELATION_WINDOW}-day rolling correlation: {ticker} vs Nifty 50")
        figure.update_yaxes(range=[-1, 1]); figure.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(figure, width="stretch")
    market = px.line(data["market"], title=f"Market-wide {config.ROLLING_CORRELATION_WINDOW}-day average pairwise correlation")
    market.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Correlation")
    st.plotly_chart(market, width="stretch")
    st.caption("When this line rises, stocks move together and diversification across the index helps less.")


def render_methodology(data: dict) -> None:
    st.header("Data & Methodology")
    st.markdown(f"""
- **Universe:** Nifty 50 constituents as of 30 Sep 2026; prices from Yahoo Finance via yfinance.
- **Returns:** simple daily returns from the *adjusted* close (splits and dividends), i.e. total return.
- **Benchmark:** {config.BENCHMARK_NAME}. The ETF's adjusted close includes dividends, unlike the ^NSEI price index.
- **Annualisation:** {config.TRADING_DAYS_PER_YEAR} trading days; risk-free rate {config.RISK_FREE_RATE:.0%} a year (approximate 91-day T-bill average).
- **Sharpe** = annualised mean excess return ÷ annualised volatility of excess returns. **Sortino** uses downside deviation below the risk-free rate instead.
- **Max drawdown** = largest fall from a running peak of the growth-of-₹1 series. **Calmar** = CAGR ÷ |max drawdown|.
- **Beta / alpha** = CAPM regression on excess returns against the benchmark over shared dates; alpha is annualised.
- **VaR / CVaR 95%** = historical one-day loss exceeded on 5% of days, and the average loss on those days.
- **Partial history:** stocks covering under {config.FULL_HISTORY_COVERAGE:.0%} of the benchmark's trading days (recent listings, demergers) are shown but not ranked, and are compared with the benchmark over their own dates.
- **Limitations:** survivorship bias (today's members only), a constant risk-free rate, and descriptive rather than predictive analysis. Not investment advice.
""")
    st.subheader("Data quality")
    quality = data["quality"]
    st.dataframe(quality.style.format({"Coverage": "{:.1%}"}), hide_index=True, width="stretch")
    st.subheader("Generated findings")
    st.text(data["insights"])


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Nifty 50 Market Analysis")
    st.caption("Historical performance, risk and correlation across India's 50 largest stocks")
    try:
        data = load_data(str(PROCESSED_DIR))
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `python main.py` from the project root to generate the processed data.")
        st.stop()

    summary = data["summary"]
    dates = data["returns"].index
    start, end = dates.min(), dates.max()
    with st.sidebar:
        st.header("Controls")
        page = st.radio("Section", PAGES)
        sectors_selected = st.multiselect("Sector filter", sorted(summary["Sector"].unique()))
        options = summary[summary["Sector"].isin(sectors_selected)] if sectors_selected else summary
        labels = {row.Ticker: f"{row.Ticker} · {row.Name}" for row in options.itertuples()}
        tickers = sorted(labels)
        default = tickers.index("RELIANCE") if "RELIANCE" in tickers else 0
        ticker = st.selectbox("Stock", tickers, index=default, format_func=labels.get)
        selected = st.date_input("Date range", value=(start.date(), end.date()), min_value=start.date(), max_value=end.date())
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        range_start, range_end = pd.Timestamp(selected[0]), pd.Timestamp(selected[1])
    else:
        range_start, range_end = start, end

    if page == "Overview":
        render_overview(data, ticker, range_start, range_end)
    elif page == "Universe":
        render_universe(data, sectors_selected)
    elif page == "Risk":
        render_risk(data, ticker, range_start, range_end)
    elif page == "Trends":
        render_trends(data, ticker, range_start, range_end)
    elif page == "Correlation":
        render_correlation(data, ticker)
    else:
        render_methodology(data)


if __name__ == "__main__":
    main()
