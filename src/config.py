"""Single source of truth for the analysis universe, period, assumptions and paths.

Every other module imports from here, so changing the universe or the analysis
window is a one-file edit.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Universe: Nifty 50 constituents effective 30 September 2026
# (August 2026 list, with BSE Ltd replacing Wipro in the September rebalance).
# Yahoo Finance symbol -> (company name, NSE macro sector)
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, tuple[str, str]] = {
    "ADANIENT.NS": ("Adani Enterprises", "Metals & Mining"),
    "ADANIPORTS.NS": ("Adani Ports & SEZ", "Services"),
    "APOLLOHOSP.NS": ("Apollo Hospitals", "Healthcare"),
    "ASIANPAINT.NS": ("Asian Paints", "Consumer Durables"),
    "AXISBANK.NS": ("Axis Bank", "Financial Services"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto", "Automobile"),
    "BAJAJFINSV.NS": ("Bajaj Finserv", "Financial Services"),
    "BAJFINANCE.NS": ("Bajaj Finance", "Financial Services"),
    "BEL.NS": ("Bharat Electronics", "Capital Goods"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecommunication"),
    "BSE.NS": ("BSE Ltd", "Financial Services"),
    "CIPLA.NS": ("Cipla", "Healthcare"),
    "COALINDIA.NS": ("Coal India", "Oil, Gas & Fuels"),
    "DRREDDY.NS": ("Dr. Reddy's Laboratories", "Healthcare"),
    "EICHERMOT.NS": ("Eicher Motors", "Automobile"),
    "ETERNAL.NS": ("Eternal (Zomato)", "Consumer Services"),
    "GRASIM.NS": ("Grasim Industries", "Construction Materials"),
    "HCLTECH.NS": ("HCL Technologies", "Information Technology"),
    "HDFCBANK.NS": ("HDFC Bank", "Financial Services"),
    "HDFCLIFE.NS": ("HDFC Life Insurance", "Financial Services"),
    "HINDALCO.NS": ("Hindalco Industries", "Metals & Mining"),
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "ICICIBANK.NS": ("ICICI Bank", "Financial Services"),
    "INDIGO.NS": ("InterGlobe Aviation", "Services"),
    "INFY.NS": ("Infosys", "Information Technology"),
    "ITC.NS": ("ITC", "FMCG"),
    "JIOFIN.NS": ("Jio Financial Services", "Financial Services"),
    "JSWSTEEL.NS": ("JSW Steel", "Metals & Mining"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Financial Services"),
    "LT.NS": ("Larsen & Toubro", "Construction"),
    "M&M.NS": ("Mahindra & Mahindra", "Automobile"),
    "MARUTI.NS": ("Maruti Suzuki", "Automobile"),
    "MAXHEALTH.NS": ("Max Healthcare", "Healthcare"),
    "NESTLEIND.NS": ("Nestle India", "FMCG"),
    "NTPC.NS": ("NTPC", "Power"),
    "ONGC.NS": ("ONGC", "Oil, Gas & Fuels"),
    "POWERGRID.NS": ("Power Grid Corporation", "Power"),
    "RELIANCE.NS": ("Reliance Industries", "Oil, Gas & Fuels"),
    "SBILIFE.NS": ("SBI Life Insurance", "Financial Services"),
    "SBIN.NS": ("State Bank of India", "Financial Services"),
    "SHRIRAMFIN.NS": ("Shriram Finance", "Financial Services"),
    "SUNPHARMA.NS": ("Sun Pharmaceutical", "Healthcare"),
    "TATACONSUM.NS": ("Tata Consumer Products", "FMCG"),
    "TATASTEEL.NS": ("Tata Steel", "Metals & Mining"),
    "TCS.NS": ("Tata Consultancy Services", "Information Technology"),
    "TECHM.NS": ("Tech Mahindra", "Information Technology"),
    "TITAN.NS": ("Titan Company", "Consumer Durables"),
    "TMPV.NS": ("Tata Motors Passenger Vehicles", "Automobile"),
    "TRENT.NS": ("Trent", "Consumer Services"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Construction Materials"),
}

# Benchmark: the Nifty BeES ETF. Its adjusted close tracks the Nifty 50 *total
# return* (dividends included), so it compares like-for-like with stock returns
# computed from adjusted closes. The ^NSEI price index excludes dividends and
# would overstate every stock's excess return by roughly the index yield.
BENCHMARK = "NIFTYBEES.NS"
BENCHMARK_NAME = "Nifty 50 (Nifty BeES ETF, total return)"

# Price corrections for events Yahoo's adjusted close gets wrong, found by
# auditing the data-quality report. Prices before `ex_date` are multiplied by
# `factor` (the standard split-style back-adjustment), so the return on
# `ex_date` becomes the genuine market move instead of a fake crash.
CORPORATE_ACTIONS: dict[str, list[dict]] = {
    "TMPV.NS": [{
        "ex_date": "2025-10-14",
        "factor": 400.00 / 660.75,  # PV price discovered in special session / prior Tata Motors close
        "note": "Tata Motors demerger: CV business spun off 1:1 (price discovery ₹400 vs ₹660.75 close)",
    }],
    "TRENT.NS": [{
        # Yahoo applied the June 2026 1:2 bonus adjustment (x 2/3) only to prices
        # from 2026-01-01, leaving a fake -33% day; finish the adjustment.
        "ex_date": "2026-01-01",
        "factor": 2 / 3,
        "note": "1:2 bonus (Jun 2026) that Yahoo only partly adjusted",
    }],
}

# ---------------------------------------------------------------------------
# Analysis period (fixed so results are reproducible; end date is exclusive).
# ---------------------------------------------------------------------------
START_DATE = "2021-10-01"
END_DATE = "2026-10-01"

# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
# Approximate average 91-day Indian T-bill yield over the window. A constant
# rate is a simplification; it shifts every Sharpe/Sortino/alpha equally.
RISK_FREE_RATE = 0.06
VAR_CONFIDENCE = 0.95

ROLLING_VOLATILITY_WINDOW = 30
ROLLING_CORRELATION_WINDOW = 60
SMA_WINDOWS = {"SMA20": 20, "SMA50": 50, "SMA200": 200}

# Data-quality rules
MIN_OBSERVATIONS = 252  # stocks with less than ~1 year of data are excluded
FULL_HISTORY_COVERAGE = 0.95  # share of benchmark days needed to count as full history
SUSPICIOUS_DAILY_MOVE = 0.25  # |daily return| above this is flagged for review

# Large moves checked against news and confirmed genuine, so the quality
# report shows them as reviewed rather than open flags.
VERIFIED_MOVES: dict[str, dict[str, str]] = {
    "ADANIENT.NS": {
        "2023-02-01": "Hindenburg report sell-off",
        "2023-02-02": "Hindenburg report sell-off",
    },
}

DOWNLOAD_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Paths. Passed around as an object so tests can point the pipeline at a
# temporary directory instead of the real project folders.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def figures(self) -> Path:
        return self.root / "outputs" / "figures"

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.figures):
            directory.mkdir(parents=True, exist_ok=True)


def tickers() -> list[str]:
    return list(UNIVERSE)


def all_symbols() -> list[str]:
    return [*UNIVERSE, BENCHMARK]


def short_name(symbol: str) -> str:
    """Display symbol without the exchange suffix, e.g. 'RELIANCE.NS' -> 'RELIANCE'."""
    return symbol.removesuffix(".NS")
