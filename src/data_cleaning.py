"""Clean raw price files, build one tidy price panel and a data-quality report.

Rules (no value is ever imputed):
- rows with an invalid date or a missing price/volume field are dropped
- duplicate dates keep the first row
- non-positive prices are dropped
- zero-volume rows are dropped: no shares traded, so the "price" is a stale
  carry-forward (Yahoo inserts these on some exchange holidays)
- spin-offs listed in config.CORPORATE_ACTIONS are back-adjusted
- a symbol with fewer than MIN_OBSERVATIONS clean rows is excluded
- a symbol covering less than FULL_HISTORY_COVERAGE of the benchmark's
  trading days is kept but flagged as partial history
- daily adjusted-close moves above SUSPICIOUS_DAILY_MOVE are flagged for review
"""

import pandas as pd

from src import config
from src.config import Paths

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_prices(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a cleaned, date-indexed frame for one symbol."""
    missing = [column for column in ["Date", *PRICE_COLUMNS] if column not in raw.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")

    data = data.dropna(subset=["Date", *PRICE_COLUMNS])
    data = data.sort_values("Date").drop_duplicates(subset="Date", keep="first")
    prices = ["Open", "High", "Low", "Close", "Adj Close"]
    data = data[(data[prices] > 0).all(axis=1) & (data["Volume"] > 0)]
    data = data.set_index("Date")[PRICE_COLUMNS]
    data = apply_corporate_actions(data, symbol)
    validate_clean(data, symbol)
    return data


def apply_corporate_actions(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Back-adjust prices before each configured spin-off so the ex-date isn't a fake loss."""
    actions = config.CORPORATE_ACTIONS.get(symbol, [])
    if not actions:
        return data
    data = data.copy()
    for action in actions:
        before = data.index < pd.Timestamp(action["ex_date"])
        data.loc[before, ["Open", "High", "Low", "Close", "Adj Close"]] *= action["factor"]
    return data


def validate_clean(data: pd.DataFrame, symbol: str) -> None:
    if data.empty:
        raise ValueError(f"{symbol}: no valid rows after cleaning")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: index is not a DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError(f"{symbol}: dates are not sorted and unique")
    if data.isna().any().any():
        raise ValueError(f"{symbol}: missing values remain")


def quality_row(symbol: str, raw_rows: int, clean: pd.DataFrame | None, calendar: pd.DatetimeIndex | None,
                note: str = "", zero_volume: int = 0) -> dict:
    """Describe one symbol's data quality."""
    row = {
        "Ticker": config.short_name(symbol),
        "Symbol": symbol,
        "Raw Rows": raw_rows,
        "Clean Rows": 0 if clean is None else len(clean),
        "Rows Dropped": raw_rows - (0 if clean is None else len(clean)),
        "First Date": None,
        "Last Date": None,
        "Coverage": 0.0,
        "Zero Volume Rows Dropped": 0,
        "Suspicious Moves": 0,
        "Verified Large Moves": 0,
        "Status": "Missing",
        "Notes": note,
    }
    if clean is None or clean.empty:
        return row

    returns = clean["Adj Close"].pct_change().dropna()
    large = returns[returns.abs() > config.SUSPICIOUS_DAILY_MOVE]
    verified = config.VERIFIED_MOVES.get(symbol, {})
    unverified = [d for d in large.index if d.date().isoformat() not in verified]
    row.update({
        "First Date": clean.index.min().date().isoformat(),
        "Last Date": clean.index.max().date().isoformat(),
        "Zero Volume Rows Dropped": zero_volume,
        "Suspicious Moves": len(unverified),
        "Verified Large Moves": len(large) - len(unverified),
    })
    if calendar is not None and len(calendar):
        row["Coverage"] = float(clean.index.isin(calendar).sum() / len(calendar))
    else:
        row["Coverage"] = 1.0

    if len(clean) < config.MIN_OBSERVATIONS:
        row["Status"] = "Excluded"
        row["Notes"] = row["Notes"] or f"fewer than {config.MIN_OBSERVATIONS} observations"
    elif row["Coverage"] < config.FULL_HISTORY_COVERAGE:
        row["Status"] = "Partial history"
        row["Notes"] = row["Notes"] or f"data starts {row['First Date']} (listing, demerger or symbol change)"
    else:
        row["Status"] = "Full history"
    for action in config.CORPORATE_ACTIONS.get(symbol, []):
        row["Notes"] = (row["Notes"] + "; " if row["Notes"] else "") + f"adjusted for {action['note']}"
    if row["Suspicious Moves"]:
        days = ", ".join(d.date().isoformat() for d in unverified)
        row["Notes"] = (row["Notes"] + "; " if row["Notes"] else "") + f"unexplained large moves on {days}: review"
    if row["Verified Large Moves"]:
        row["Notes"] = (row["Notes"] + "; " if row["Notes"] else "") + "large moves checked: " + "; ".join(
            f"{day} {reason}" for day, reason in verified.items())
    return row


def run(paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every raw file; save prices.parquet and data_quality.csv."""
    paths.ensure()
    cleaned: dict[str, pd.DataFrame] = {}
    raw_rows: dict[str, int] = {}
    zero_volume: dict[str, int] = {}
    errors: dict[str, str] = {}

    for symbol in config.all_symbols():
        path = paths.raw / f"{symbol}.csv"
        if not path.exists():
            raw_rows[symbol] = 0
            errors[symbol] = "raw file not found (download failed or not run)"
            continue
        raw = pd.read_csv(path)
        raw_rows[symbol] = len(raw)
        if "Volume" in raw.columns:
            zero_volume[symbol] = int((pd.to_numeric(raw["Volume"], errors="coerce") == 0).sum())
        try:
            cleaned[symbol] = clean_prices(raw, symbol)
        except ValueError as error:
            errors[symbol] = str(error)

    if config.BENCHMARK not in cleaned:
        raise RuntimeError(f"Benchmark {config.BENCHMARK} has no usable data: {errors.get(config.BENCHMARK)}")
    calendar = cleaned[config.BENCHMARK].index

    quality = pd.DataFrame([
        quality_row(symbol, raw_rows[symbol], cleaned.get(symbol), calendar, errors.get(symbol, ""), zero_volume.get(symbol, 0))
        for symbol in config.all_symbols()
    ])
    benchmark_mask = quality["Symbol"] == config.BENCHMARK
    quality.loc[benchmark_mask, "Status"] = "Benchmark"

    usable = quality.loc[quality["Status"] != "Excluded", "Symbol"]
    frames = []
    for symbol in usable:
        if symbol in cleaned:
            frame = cleaned[symbol].reset_index()
            frame.insert(1, "Ticker", config.short_name(symbol))
            frames.append(frame)
    prices = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices.to_parquet(paths.processed / "prices.parquet", index=False)
    quality.to_csv(paths.processed / "data_quality.csv", index=False)

    counts = quality["Status"].value_counts().to_dict()
    print(f"Cleaning complete: {counts}")
    for row in quality[quality["Status"].isin(["Missing", "Excluded", "Partial history"])].itertuples():
        print(f"  {row.Ticker}: {row.Status} - {row.Notes}")
    return prices, quality
