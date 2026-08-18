"""Read-only Streamlit dashboard for the completed stock analysis project."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

st.set_page_config(page_title="Stock Market Trend Analysis", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] { padding: 0.45rem 0.25rem; }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        white-space: nowrap;
        overflow: visible;
        text-overflow: clip;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(filename: str, index_col: str | None = None) -> pd.DataFrame:
    """Read one validated processed CSV without modifying it."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required processed file is missing: data/processed/{filename}")
    return pd.read_csv(path, index_col=index_col, parse_dates=[index_col] if index_col else None)


@st.cache_data
def load_data() -> dict[str, object]:
    """Load all dashboard inputs once per Streamlit session."""
    return {
        "final": read_csv("final_stock_analysis.csv"),
        "benchmark": read_csv("benchmark_summary.csv"),
        "trend": read_csv("trend_summary.csv"),
        "correlation_matrix": read_csv("correlation_matrix.csv", index_col=0),
        "correlation_pairs": read_csv("correlation_pairs.csv"),
        "benchmark_rolling": read_csv("benchmark_rolling_correlation.csv", index_col="Date"),
        "insights": (PROCESSED_DIR / "final_insights.txt").read_text(encoding="utf-8"),
        "stock_returns": {ticker: read_csv(f"{ticker}_returns.csv", index_col="Date") for ticker in TICKERS},
        "stock_trends": {ticker: read_csv(f"{ticker}_trends.csv", index_col="Date") for ticker in TICKERS},
        "spy_returns": read_csv("SPY_returns.csv", index_col="Date"),
    }


def period_bounds(data: dict[str, object]) -> tuple[pd.Timestamp, pd.Timestamp]:
    spy = data["spy_returns"]
    return spy.index.min(), spy.index.max()


def indexed_performance(data: dict[str, object], ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Create temporary indexed display data without writing to disk."""
    stock = data["stock_returns"][ticker]["Close"].rename(ticker)
    spy = data["spy_returns"]["Close"].rename("SPY")
    chart = pd.concat([stock, spy], axis=1, join="inner").loc[start:end].dropna()
    return chart.div(chart.iloc[0]).mul(100).reset_index().melt(
        id_vars="Date", var_name="Series", value_name="Indexed Value"
    )


def performance_chart(data: dict[str, object], ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    chart = indexed_performance(data, ticker, start, end)
    figure = px.line(chart, x="Date", y="Indexed Value", color="Series", title="Indexed Performance (Start = 100)")
    figure.update_yaxes(title="Indexed value")
    figure.update_layout(legend_title_text="Series", hovermode="x unified")
    return figure


def render_overview(data: dict[str, object], ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    st.header("Overview")
    benchmark_row = data["benchmark"].set_index("Stock").loc[ticker]
    row = data["final"].set_index("Stock").loc[ticker]
    columns = st.columns(6)
    columns[0].metric("Total return", f"{row['Total Price Return']:.2%}")
    columns[1].metric("Excess vs SPY", f"{benchmark_row['Excess Total Return']:.2%}")
    columns[2].metric("Annualized volatility", f"{row['Annualized Volatility']:.2%}")
    columns[3].metric("Volatility ratio", f"{benchmark_row['Volatility Ratio']:.2f}x")
    columns[4].metric("Trend", row["Trend Classification"])
    columns[5].metric("Close vs SMA200", f"{row['Close vs SMA200 (%)']:.2f}%")
    st.caption(f"Historical analysis period: {start.date()} to {end.date()}")
    if benchmark_row["Excess Total Return"] >= 0:
        st.info(f"{ticker} returned more than SPY over the selected historical period, with annualized volatility of {row['Annualized Volatility']:.2%}.")
    else:
        st.info(f"{ticker} returned less than SPY over the selected historical period, with annualized volatility of {row['Annualized Volatility']:.2%}.")
    st.plotly_chart(performance_chart(data, ticker, start, end), width="stretch")


def render_performance(data: dict[str, object], ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    st.header("Performance")
    summary = data["benchmark"]
    row = summary.set_index("Stock").loc[ticker]
    selected_table = pd.DataFrame({
        "Metric": ["Stock total return", "SPY total return", "Excess total return", "Relative performance"],
        "Value": [f"{row['Stock Total Return']:.2%}", f"{row['SPY Total Return']:.2%}", f"{row['Excess Total Return']:.2%}", f"{row['Relative Performance']:.4f}"],
    })
    st.dataframe(selected_table, hide_index=True, width="stretch")
    st.plotly_chart(performance_chart(data, ticker, start, end), width="stretch")
    st.subheader("All-stock comparison")
    table = summary[["Stock", "Stock Total Return", "SPY Total Return", "Excess Total Return", "Relative Performance"]].copy()
    st.dataframe(table.style.format({column: "{:.2%}" for column in table.columns[1:4]}).format({"Relative Performance": "{:.4f}"}), hide_index=True, width="stretch")


def render_risk(data: dict[str, object], ticker: str) -> None:
    st.header("Risk")
    summary = data["benchmark"]
    row = summary.set_index("Stock").loc[ticker]
    columns = st.columns(4)
    columns[0].metric("Stock annualized volatility", f"{row['Stock Annualized Volatility']:.2%}")
    columns[1].metric("SPY annualized volatility", f"{row['SPY Annualized Volatility']:.2%}")
    columns[2].metric("Volatility difference", f"{row['Volatility Difference']:.2%}")
    columns[3].metric("Volatility ratio", f"{row['Volatility Ratio']:.2f}x")
    risk_chart = pd.concat([
        summary[["Stock", "Stock Annualized Volatility"]].rename(columns={"Stock Annualized Volatility": "Annualized Volatility"}),
        pd.DataFrame({"Stock": ["SPY"], "Annualized Volatility": [summary["SPY Annualized Volatility"].iloc[0]]}),
    ])
    st.plotly_chart(px.bar(risk_chart, x="Stock", y="Annualized Volatility", title="Annualized Volatility Comparison", labels={"Annualized Volatility": "Annualized volatility"}), width="stretch")
    performance = summary[["Stock", "Stock Total Return", "Stock Annualized Volatility"]].rename(columns={"Stock Total Return": "Total Return", "Stock Annualized Volatility": "Annualized Volatility"})
    spy_point = pd.DataFrame({"Stock": ["SPY"], "Total Return": [summary["SPY Total Return"].iloc[0]], "Annualized Volatility": [summary["SPY Annualized Volatility"].iloc[0]]})
    scatter = pd.concat([performance, spy_point], ignore_index=True)
    figure = px.scatter(scatter, x="Annualized Volatility", y="Total Return", text="Stock", color="Stock", title="Risk vs Historical Return", labels={"Annualized Volatility": "Annualized volatility", "Total Return": "Total return"})
    figure.update_traces(textposition="top center")
    st.plotly_chart(figure, width="stretch")
    risk_path = PROCESSED_DIR / f"{ticker}_risk.csv"
    if risk_path.exists():
        rolling_risk = read_csv(f"{ticker}_risk.csv", index_col="Date")
        if "Rolling Annualized Volatility" in rolling_risk:
            st.plotly_chart(px.line(rolling_risk, y="Rolling Annualized Volatility", title=f"30-Day Rolling Annualized Volatility — {ticker}"), width="stretch")
    else:
        st.info("Validated Phase 5 rolling-volatility series were not saved as per-stock processed files, so the available volatility summary and risk comparisons are shown.")


def render_trends(data: dict[str, object], ticker: str) -> None:
    st.header("Trends")
    trend = data["stock_trends"][ticker].reset_index()
    figure = go.Figure()
    for column in ["Close", "SMA20", "SMA50", "SMA200"]:
        figure.add_trace(go.Scatter(x=trend["Date"], y=trend[column], mode="lines", name=column))
    if "Bullish Crossover" in trend.columns:
        bullish = trend[trend["Bullish Crossover"]]
        bearish = trend[trend["Bearish Crossover"]]
        figure.add_trace(go.Scatter(x=bullish["Date"], y=bullish["Close"], mode="markers", name="Bullish crossover", marker_symbol="triangle-up", marker_color="green"))
        figure.add_trace(go.Scatter(x=bearish["Date"], y=bearish["Close"], mode="markers", name="Bearish crossover", marker_symbol="triangle-down", marker_color="red"))
    figure.update_layout(title=f"{ticker} Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(figure, width="stretch")
    row = data["final"].set_index("Stock").loc[ticker]
    values = [row["Trend Classification"], f"{row['Latest Close']:.2f}", f"{row['SMA20']:.2f}", f"{row['SMA50']:.2f}", f"{row['SMA200']:.2f}", f"{row['Close vs SMA200 (%)']:.2f}%", str(row["Latest Crossover Date"]), str(row["Latest Crossover Type"])]
    st.dataframe(pd.DataFrame({"Metric": ["Trend classification", "Latest Close", "SMA20", "SMA50", "SMA200", "Close vs SMA200", "Latest crossover date", "Latest crossover type"], "Value": values}), hide_index=True, width="stretch")


def render_correlation(data: dict[str, object], ticker: str) -> None:
    st.header("Correlation")
    matrix = data["correlation_matrix"]
    st.plotly_chart(px.imshow(matrix, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Daily Return Correlation Matrix"), width="stretch")
    pairs = data["correlation_pairs"]
    st.write(f"Highest correlated pair: **{pairs.iloc[0]['Stock A']}–{pairs.iloc[0]['Stock B']}** ({pairs.iloc[0]['Correlation']:.4f})")
    st.write(f"Lowest correlated pair: **{pairs.iloc[-1]['Stock A']}–{pairs.iloc[-1]['Stock B']}** ({pairs.iloc[-1]['Correlation']:.4f})")
    rolling = data["benchmark_rolling"][[ticker]].rename(columns={ticker: "Rolling correlation with SPY"})
    figure = px.line(rolling, y="Rolling correlation with SPY", title=f"60-Day Rolling Correlation — {ticker} vs SPY")
    figure.update_yaxes(range=[-1, 1])
    st.plotly_chart(figure, width="stretch")


def render_about(data: dict[str, object], start: pd.Timestamp, end: pd.Timestamp) -> None:
    st.header("Methodology & About")
    st.write("This dashboard reads validated processed datasets from the completed analysis pipeline. It covers historical daily market data for AAPL, MSFT, GOOGL, AMZN, and NVDA, with SPY used as the benchmark.")
    st.write("The dashboard presents performance, volatility, moving averages, descriptive trends, correlations, and rolling correlation with SPY. It is read-only and does not download data or run analysis scripts.")
    st.caption(f"Dataset period: {start.date()} to {end.date()}")
    st.warning("Most metrics are calculated over the historical analysis period. Trend classification and latest moving-average values represent the latest available observation in the dataset. This dashboard presents historical analysis and is not investment advice.")
    st.subheader("Final project insights")
    st.text(data["insights"])


def main() -> None:
    st.title("Stock Market Trend Analysis")
    st.caption("Historical Financial Analytics Dashboard")
    try:
        data = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run the project analysis scripts first to generate the validated processed datasets.")
        st.stop()
    start, end = period_bounds(data)
    with st.sidebar:
        st.header("Dashboard controls")
        ticker = st.selectbox("Stock", TICKERS, index=TICKERS.index("NVDA"))
        selected_range = st.date_input("Date range", value=(start.date(), end.date()), min_value=start.date(), max_value=end.date())
        page = st.radio("Section", ["Overview", "Performance", "Risk", "Trends", "Correlation", "Methodology & About"])
    if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
        selected_start, selected_end = pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])
    else:
        selected_start, selected_end = start, end
    if selected_start > selected_end:
        st.error("Select a valid date range.")
        st.stop()
    if page == "Overview": render_overview(data, ticker, selected_start, selected_end)
    elif page == "Performance": render_performance(data, ticker, selected_start, selected_end)
    elif page == "Risk": render_risk(data, ticker)
    elif page == "Trends": render_trends(data, ticker)
    elif page == "Correlation": render_correlation(data, ticker)
    else: render_about(data, start, end)


if __name__ == "__main__":
    main()
