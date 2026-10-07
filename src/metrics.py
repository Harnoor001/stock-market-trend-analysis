"""Pure performance and risk metric functions.

All functions take *simple daily returns* (not prices) as a pandas Series and
ignore NaN. Annualisation uses TRADING_DAYS_PER_YEAR. Nothing here reads or
writes files, so every function can be unit-tested in isolation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

PERIODS = config.TRADING_DAYS_PER_YEAR


def _clean(returns: pd.Series) -> pd.Series:
    returns = pd.Series(returns, dtype="float64").dropna()
    if not np.isfinite(returns).all():
        raise ValueError("returns contain infinite values")
    return returns


def daily_risk_free(annual_rate: float = config.RISK_FREE_RATE, periods: int = PERIODS) -> float:
    """Convert an annual rate to the equivalent compounded per-period rate."""
    return (1 + annual_rate) ** (1 / periods) - 1


def total_return(returns: pd.Series) -> float:
    returns = _clean(returns)
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, periods: int = PERIODS) -> float:
    """Compound annual growth rate, using the number of return observations as time."""
    returns = _clean(returns)
    if returns.empty:
        return float("nan")
    years = len(returns) / periods
    return float((1 + total_return(returns)) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods: int = PERIODS) -> float:
    returns = _clean(returns)
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = config.RISK_FREE_RATE, periods: int = PERIODS) -> float:
    """Annualised mean excess return divided by annualised excess-return volatility."""
    excess = _clean(returns) - daily_risk_free(risk_free, periods)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def downside_deviation(returns: pd.Series, target: float = 0.0, periods: int = PERIODS) -> float:
    """Annualised root-mean-square of returns below `target` (all observations in the denominator)."""
    shortfall = np.minimum(_clean(returns) - target, 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float = config.RISK_FREE_RATE, periods: int = PERIODS) -> float:
    """Annualised mean excess return divided by annualised downside deviation below the risk-free rate."""
    rf = daily_risk_free(risk_free, periods)
    returns = _clean(returns)
    downside = downside_deviation(returns, target=rf, periods=periods)
    if downside == 0:
        return float("nan")
    return float((returns - rf).mean() * periods / downside)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak of the wealth index (0 at new highs, negative otherwise)."""
    returns = _clean(returns)
    wealth = (1 + returns).cumprod()
    # Include the starting value of 1 so a loss on day one counts as a drawdown.
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


@dataclass(frozen=True)
class Drawdown:
    max_drawdown: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None
    days_to_recover: int | None  # trading days from trough back to the previous peak
    current_drawdown: float


def max_drawdown(returns: pd.Series) -> Drawdown:
    """Largest peak-to-trough fall, with the dates of the peak, trough and recovery."""
    drawdown = drawdown_series(returns)
    if drawdown.empty:
        return Drawdown(float("nan"), None, None, None, None, float("nan"))
    trough = drawdown.idxmin()
    worst = float(drawdown.loc[trough])
    if worst == 0:
        return Drawdown(0.0, None, None, None, None, float(drawdown.iloc[-1]))

    before = drawdown.loc[:trough]
    at_peak = before[before == 0]
    peak = at_peak.index[-1] if len(at_peak) else None  # None: fell from the very first price
    after = drawdown.loc[trough:]
    recovered = after[after >= 0]
    recovery = recovered.index[0] if len(recovered) else None
    days = None
    if recovery is not None:
        days = int(drawdown.index.get_loc(recovery) - drawdown.index.get_loc(trough))
    return Drawdown(worst, peak, trough, recovery, days, float(drawdown.iloc[-1]))


def calmar_ratio(returns: pd.Series, periods: int = PERIODS) -> float:
    mdd = max_drawdown(returns).max_drawdown
    if not mdd:
        return float("nan")
    return float(cagr(returns, periods) / abs(mdd))


def value_at_risk(returns: pd.Series, confidence: float = config.VAR_CONFIDENCE) -> float:
    """Historical one-day VaR, reported as a positive loss fraction."""
    returns = _clean(returns)
    return float(-np.quantile(returns, 1 - confidence))


def conditional_value_at_risk(returns: pd.Series, confidence: float = config.VAR_CONFIDENCE) -> float:
    """Historical expected shortfall: average loss on days at or beyond the VaR threshold."""
    returns = _clean(returns)
    threshold = np.quantile(returns, 1 - confidence)
    return float(-returns[returns <= threshold].mean())


def _align(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([_clean(returns).rename("asset"), _clean(benchmark).rename("benchmark")], axis=1, join="inner")
    if len(aligned) < 2:
        raise ValueError("not enough overlapping observations with the benchmark")
    return aligned


def beta(returns: pd.Series, benchmark: pd.Series, risk_free: float = config.RISK_FREE_RATE, periods: int = PERIODS) -> float:
    """CAPM beta estimated on excess returns over overlapping dates."""
    aligned = _align(returns, benchmark) - daily_risk_free(risk_free, periods)
    variance = aligned["benchmark"].var(ddof=1)
    return float(aligned["asset"].cov(aligned["benchmark"]) / variance)


def jensens_alpha(returns: pd.Series, benchmark: pd.Series, risk_free: float = config.RISK_FREE_RATE, periods: int = PERIODS) -> float:
    """Annualised CAPM alpha: mean excess return not explained by beta x benchmark excess return."""
    aligned = _align(returns, benchmark) - daily_risk_free(risk_free, periods)
    b = aligned["asset"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1)
    return float((aligned["asset"].mean() - b * aligned["benchmark"].mean()) * periods)


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods: int = PERIODS) -> float:
    aligned = _align(returns, benchmark)
    return float((aligned["asset"] - aligned["benchmark"]).std(ddof=1) * np.sqrt(periods))


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = PERIODS) -> float:
    aligned = _align(returns, benchmark)
    active = aligned["asset"] - aligned["benchmark"]
    std = active.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(active.mean() / std * np.sqrt(periods))


def correlation(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned = _align(returns, benchmark)
    return float(aligned["asset"].corr(aligned["benchmark"]))


def rolling_volatility(returns: pd.Series, window: int = config.ROLLING_VOLATILITY_WINDOW, periods: int = PERIODS) -> pd.Series:
    return _clean(returns).rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods)


def summarize(returns: pd.Series, benchmark: pd.Series | None = None, risk_free: float = config.RISK_FREE_RATE) -> dict[str, float | str | int | None]:
    """Every headline metric for one return series, in one flat dict."""
    returns = _clean(returns)
    dd = max_drawdown(returns)
    row: dict[str, float | str | int | None] = {
        "Start Date": returns.index.min().date().isoformat() if len(returns) else None,
        "End Date": returns.index.max().date().isoformat() if len(returns) else None,
        "Observations": int(len(returns)),
        "Total Return": total_return(returns),
        "CAGR": cagr(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free),
        "Sortino Ratio": sortino_ratio(returns, risk_free),
        "Max Drawdown": dd.max_drawdown,
        "Max Drawdown Peak": dd.peak_date.date().isoformat() if dd.peak_date is not None else None,
        "Max Drawdown Trough": dd.trough_date.date().isoformat() if dd.trough_date is not None else None,
        "Max Drawdown Recovery": dd.recovery_date.date().isoformat() if dd.recovery_date is not None else None,
        "Days to Recover": dd.days_to_recover,
        "Current Drawdown": dd.current_drawdown,
        "Calmar Ratio": calmar_ratio(returns),
        "VaR 95% (1-day)": value_at_risk(returns),
        "CVaR 95% (1-day)": conditional_value_at_risk(returns),
        "Best Day": float(returns.max()),
        "Worst Day": float(returns.min()),
        "Positive Days": float((returns > 0).mean()),
    }
    if benchmark is not None:
        row.update({
            "Beta": beta(returns, benchmark, risk_free),
            "Alpha (annual)": jensens_alpha(returns, benchmark, risk_free),
            "Correlation with Benchmark": correlation(returns, benchmark),
            "Tracking Error": tracking_error(returns, benchmark),
            "Information Ratio": information_ratio(returns, benchmark),
            # Benchmark CAGR over this asset's own dates, so partial-history stocks compare fairly.
            "Benchmark CAGR (same dates)": cagr(_clean(benchmark).loc[returns.index.min():returns.index.max()]),
        })
        row["Excess CAGR"] = row["CAGR"] - row["Benchmark CAGR (same dates)"]
    return row
