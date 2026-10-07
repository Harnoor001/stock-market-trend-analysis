"""Download daily OHLCV + adjusted-close data from Yahoo Finance.

Raw files are written one per symbol to data/raw/<SYMBOL>.csv and are not
committed; everything downstream reads them from disk, so the download is the
only step that needs network access.
"""

import time

import pandas as pd
import yfinance as yf

from src import config
from src.config import Paths

RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _extract_symbol(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pull one symbol out of a (possibly multi-index) yfinance result."""
    data = downloaded
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            if symbol in data.columns.get_level_values(level):
                data = data.xs(symbol, axis=1, level=level)
                break
        else:
            return pd.DataFrame(columns=RAW_COLUMNS)
    data = data.reset_index()
    data = data.rename(columns={"Datetime": "Date"})
    if "Date" not in data.columns:
        return pd.DataFrame(columns=RAW_COLUMNS)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    data = data[[column for column in RAW_COLUMNS if column in data.columns]]
    return data.dropna(how="all", subset=[c for c in RAW_COLUMNS[1:] if c in data.columns])


def _download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,  # keep both Close and Adj Close
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )


def download_all(symbols: list[str], start: str, end: str, attempts: int = config.DOWNLOAD_ATTEMPTS) -> dict[str, pd.DataFrame]:
    """Batch-download symbols, retrying only the ones that came back empty."""
    results: dict[str, pd.DataFrame] = {}
    pending = list(symbols)
    for attempt in range(1, attempts + 1):
        if not pending:
            break
        try:
            downloaded = _download(pending, start, end)
        except Exception as error:  # network/API errors: retry the whole batch
            print(f"Attempt {attempt}: download error ({error})")
            downloaded = pd.DataFrame()
        still_pending = []
        for symbol in pending:
            data = _extract_symbol(downloaded, symbol) if not downloaded.empty else pd.DataFrame()
            if data.empty or "Adj Close" not in data.columns:
                still_pending.append(symbol)
            else:
                results[symbol] = data
        pending = still_pending
        if pending and attempt < attempts:
            time.sleep(2 * attempt)
    for symbol in pending:
        print(f"{symbol}: no data returned after {attempts} attempts")
    return results


def run(paths: Paths = Paths(), symbols: list[str] | None = None, start: str = config.START_DATE, end: str = config.END_DATE) -> list[str]:
    """Download and save raw files; return the list of symbols that failed."""
    paths.ensure()
    symbols = symbols or config.all_symbols()
    results = download_all(symbols, start, end)
    for symbol, data in results.items():
        data.to_csv(paths.raw / f"{symbol}.csv", index=False)
    print(f"Downloaded {len(results)}/{len(symbols)} symbols ({start} to before {end}) into {paths.raw}")
    return [symbol for symbol in symbols if symbol not in results]
