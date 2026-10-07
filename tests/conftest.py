"""Shared fixtures: synthetic raw price files shaped like real yfinance output."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.config import Paths  # noqa: E402

LATE_LISTING = "JIOFIN.NS"  # partial history
TOO_SHORT = "TMPV.NS"  # fewer than MIN_OBSERVATIONS rows -> excluded


def _synthetic_prices(dates: pd.DatetimeIndex, returns: np.ndarray, start_price: float, dividend_drag: float) -> pd.DataFrame:
    adjusted = start_price * np.cumprod(1 + returns)
    # Close sits slightly above Adj Close early on, as when past dividends are back-adjusted out.
    close = adjusted * (1 + dividend_drag * np.linspace(1, 0, len(dates)))
    noise = np.abs(returns) + 0.002
    return pd.DataFrame({
        "Date": dates,
        "Open": close * (1 - noise / 2),
        "High": close * (1 + noise),
        "Low": close * (1 - noise),
        "Close": close,
        "Adj Close": adjusted,
        "Volume": np.random.default_rng(0).integers(1e5, 1e7, len(dates)),
    })


def write_synthetic_raw(paths: Paths, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(config.START_DATE, pd.Timestamp(config.END_DATE) - pd.Timedelta(days=1))
    n = len(dates)
    market = rng.normal(0.0005, 0.01, n)
    sectors = sorted({sector for _, sector in config.UNIVERSE.values()})
    sector_factor = {sector: rng.normal(0, 0.006, n) for sector in sectors}
    paths.ensure()

    benchmark = _synthetic_prices(dates, market, 200.0, 0.0)
    benchmark.to_csv(paths.raw / f"{config.BENCHMARK}.csv", index=False)

    for i, (symbol, (_, sector)) in enumerate(config.UNIVERSE.items()):
        beta = rng.uniform(0.6, 1.5)
        returns = 0.0002 + beta * market + sector_factor[sector] + rng.normal(0, 0.012, n)
        frame = _synthetic_prices(dates, returns, rng.uniform(100, 3000), rng.uniform(0, 0.08))
        if symbol == LATE_LISTING:
            frame = frame[frame["Date"] >= "2023-08-21"]
        if symbol == TOO_SHORT:
            frame = frame[frame["Date"] >= "2026-01-01"]
        if i == 0:
            # Corrupt rows the cleaner must drop: a duplicate date and a missing price.
            frame = pd.concat([frame, frame.iloc[[10]]])
            frame.loc[frame.index[20], "Close"] = np.nan
        frame.to_csv(paths.raw / f"{symbol}.csv", index=False)


@pytest.fixture(scope="session")
def pipeline_results(tmp_path_factory):
    from src import pipeline

    paths = Paths(root=tmp_path_factory.mktemp("project"))
    write_synthetic_raw(paths)
    results = pipeline.run(paths, download=False, figures=True)
    return paths, results
