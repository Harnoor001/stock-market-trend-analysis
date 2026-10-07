"""Simple moving averages, descriptive trend labels and SMA20/SMA50 crossovers.

Moving averages use the (split-adjusted) Close, i.e. the price an investor saw
on screen; returns elsewhere use the dividend-adjusted close.
"""

import numpy as np
import pandas as pd

from src import config
from src.config import Paths


def moving_averages(close: pd.Series, windows: dict[str, int] = config.SMA_WINDOWS) -> pd.DataFrame:
    """Trailing (non-centred) SMAs; values are NaN until the window is full."""
    result = close.to_frame("Close")
    for column, window in windows.items():
        result[column] = close.rolling(window, min_periods=window).mean()
    return result


def crossovers(data: pd.DataFrame) -> pd.DataFrame:
    """Flag the day SMA20 crosses SMA50, using only the previous day's values (no look-ahead)."""
    result = data.copy()
    previous_20, previous_50 = result["SMA20"].shift(1), result["SMA50"].shift(1)
    available = result[["SMA20", "SMA50"]].notna().all(axis=1) & previous_20.notna() & previous_50.notna()
    result["Bullish Crossover"] = available & (previous_20 <= previous_50) & (result["SMA20"] > result["SMA50"])
    result["Bearish Crossover"] = available & (previous_20 >= previous_50) & (result["SMA20"] < result["SMA50"])
    return result


def classify(latest: pd.Series) -> str:
    """Bullish if Close > SMA20 > SMA50 > SMA200, Bearish if fully reversed, otherwise Mixed."""
    if latest[["Close", "SMA20", "SMA50", "SMA200"]].isna().any():
        return "Insufficient data"
    if latest["Close"] > latest["SMA20"] > latest["SMA50"] > latest["SMA200"]:
        return "Bullish"
    if latest["Close"] < latest["SMA20"] < latest["SMA50"] < latest["SMA200"]:
        return "Bearish"
    return "Mixed"


def analyze(close: pd.Series) -> tuple[pd.DataFrame, dict]:
    data = crossovers(moving_averages(close))
    latest = data.iloc[-1]
    events = data[data["Bullish Crossover"] | data["Bearish Crossover"]]
    last_event = events.iloc[-1] if len(events) else None
    summary = {
        "Latest Close": float(latest["Close"]),
        "SMA20": latest["SMA20"],
        "SMA50": latest["SMA50"],
        "SMA200": latest["SMA200"],
        "Close vs SMA200": float(latest["Close"] / latest["SMA200"] - 1) if pd.notna(latest["SMA200"]) else np.nan,
        "Trend": classify(latest),
        "Bullish Crossovers": int(data["Bullish Crossover"].sum()),
        "Bearish Crossovers": int(data["Bearish Crossover"].sum()),
        "Latest Crossover Date": events.index[-1].date().isoformat() if last_event is not None else None,
        "Latest Crossover Type": ("Bullish" if last_event["Bullish Crossover"] else "Bearish") if last_event is not None else None,
    }
    return data, summary


def validate(trends: pd.DataFrame) -> None:
    """Independently recompute the latest SMA of every ticker."""
    for ticker, frame in trends.groupby("Ticker"):
        close = frame["Close"].to_numpy()
        for column, window in config.SMA_WINDOWS.items():
            if len(close) >= window and abs(close[-window:].mean() - frame[column].iloc[-1]) > 1e-8 * max(1.0, close[-1]):
                raise ValueError(f"{ticker}: {column} sanity check failed")
        if (frame["Bullish Crossover"] & frame["Bearish Crossover"]).any():
            raise ValueError(f"{ticker}: a day is flagged as both crossover types")


def run(prices: pd.DataFrame, paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, rows = [], []
    for ticker, frame in prices.groupby("Ticker"):
        data, summary = analyze(frame.set_index("Date")["Close"].sort_index())
        data.insert(0, "Ticker", ticker)
        frames.append(data.reset_index())
        rows.append({"Ticker": ticker, **summary})
    trends = pd.concat(frames, ignore_index=True)
    validate(trends)
    summary = pd.DataFrame(rows)
    trends.to_parquet(paths.processed / "trends.parquet", index=False)
    summary.to_csv(paths.processed / "trend_summary.csv", index=False)
    print(f"Trends: {summary['Trend'].value_counts().to_dict()}; SMA sanity check PASS")
    return trends, summary
