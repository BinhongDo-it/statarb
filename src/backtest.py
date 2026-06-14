"""
Signal generation, portfolio construction, backtesting and performance metrics.

The trading logic is a standard z-score reversal on the spread of each pair:
  - estimate a rolling hedge ratio (beta) and intercept (alpha) on log prices,
  - form the spread, standardise it into a rolling z-score,
  - enter when |z| > entry, scaling the two legs to stay roughly dollar-neutral,
  - exit when |z| falls back inside an exit band.

Performance is evaluated out-of-sample, net of transaction costs, and compared
against a buy-and-hold Bitcoin benchmark via the information ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def generate_signals(prices: pd.DataFrame, pairs, window: int = 90) -> dict:
    """
    For each pair, compute a rolling hedge ratio (beta), the spread, and its
    z-score. Returns a dict keyed by pair -> DataFrame [beta, spread, z].

    No look-ahead: every quantity used to form the signal on day t is computed
    from data up to and including day t, then shifted one day so that trading
    on day t only ever sees information available at the close of day t-1.
    """
    mp = max(window // 2, 20)
    out = {}
    for i, j in pairs:
        li = np.log(prices[i].replace(0, np.nan).ffill())
        lj = np.log(prices[j].replace(0, np.nan).ffill())

        # Rolling hedge ratio from covariance / variance of log prices.
        beta = li.rolling(window, min_periods=mp).cov(lj) / li.rolling(window, min_periods=mp).var()
        alpha = lj.rolling(window, min_periods=mp).mean() - beta * li.rolling(window, min_periods=mp).mean()

        # Spread = residual of lj on li. Standardise ONCE into a z-score that
        # measures how many std devs the spread sits from its recent norm.
        spread = lj - (beta * li + alpha)
        mu = spread.rolling(window, min_periods=mp).mean()
        sd = spread.rolling(window, min_periods=mp).std()
        z = (spread - mu) / sd

        # Shift everything by one day -> signal for day t uses only <= t-1 info.
        out[(i, j)] = pd.DataFrame(
            {"beta": beta.shift(1), "spread": spread.shift(1), "z": z.shift(1)}
        )
    return out


# ----------------------------------------------------------------------------
# Portfolio
# ----------------------------------------------------------------------------
def _pair_states(z: pd.Series, entry: float, exit_band: float) -> pd.Series:
    """
    Map a pair's z-score series to a per-day state in {-1, 0, +1} using an
    explicit entry/exit state machine (no overlap-flattening tricks):

      state = +1  -> spread is low (z < -entry): long j, short i, bet on rise
      state = -1  -> spread is high (z > entry): short j, long i, bet on fall
      state =  0  -> flat

    Once in a position we hold until |z| falls back inside exit_band, then flat.
    This makes the exit actually fire, instead of being overwritten.
    """
    state = np.zeros(len(z))
    cur = 0
    zv = z.to_numpy()
    for t in range(len(zv)):
        zt = zv[t]
        if np.isnan(zt):
            state[t] = cur
            continue
        if cur == 0:
            if zt > entry:
                cur = -1
            elif zt < -entry:
                cur = +1
        else:  # in a position; exit when spread reverts inside the band
            if abs(zt) <= exit_band:
                cur = 0
        state[t] = cur
    return pd.Series(state, index=z.index)


def build_positions(
    signals: dict,
    columns: pd.Index,
    index: pd.Index,
    entry: float = 1.0,
    exit_band: float = 0.2,
) -> pd.DataFrame:
    """
    Translate z-score signals into dollar-neutral daily weights via a proper
    per-pair state machine, then normalise total gross exposure to 1.
    """
    pos = pd.DataFrame(0.0, index=index, columns=columns)

    for (i, j), sig in signals.items():
        z = sig["z"].reindex(index)
        beta = sig["beta"].reindex(index).fillna(0.0)
        state = _pair_states(z, entry, exit_band)

        # state +1: long j (+1), short i (-beta).  state -1: short j (-1), long i (+beta).
        pos[j] += state
        pos[i] += -state * beta

    gross = pos.abs().sum(axis=1)
    pos = pos.divide(gross.where(gross > 0, 1.0), axis=0)
    return pos


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    return (positions.fillna(0) - positions.shift().fillna(0)).abs().sum(axis=1)


def backtest(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    tcost_bps: float = 20.0,
) -> pd.Series:
    """Return the net (after-cost) daily strategy return series."""
    rets = np.log(prices.replace(0, np.nan).ffill()).diff()
    aligned = positions.shift().reindex(rets.index).fillna(0)
    gross = (aligned * rets.reindex(columns=positions.columns)).sum(axis=1)
    to = compute_turnover(positions).reindex(gross.index).fillna(0)
    net = gross - to * tcost_bps * 1e-4
    return net.dropna()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def sharpe_ratio(rets: pd.Series) -> float:
    if rets.std() == 0:
        return 0.0
    return (rets.mean() * TRADING_DAYS) / (rets.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(rets: pd.Series) -> float:
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return dd.min()


def information_ratio(strat: pd.Series, benchmark: pd.Series) -> float:
    df = pd.concat([strat, benchmark], axis=1, keys=["s", "b"]).dropna()
    if len(df) < TRADING_DAYS:
        beta = 0.0
    else:
        roll_corr = df["s"].rolling(TRADING_DAYS).corr(df["b"])
        beta = (roll_corr * df["s"].rolling(TRADING_DAYS).std() /
                df["b"].rolling(TRADING_DAYS).std()).mean()
    resid = df["s"] - beta * df["b"]
    if resid.std() == 0:
        return 0.0
    return resid.mean() / resid.std() * np.sqrt(TRADING_DAYS)


def summarise(rets: pd.Series, benchmark: pd.Series | None = None) -> dict:
    out = {
        "Sharpe": round(sharpe_ratio(rets), 3),
        "AnnReturn": round(rets.mean() * TRADING_DAYS, 4),
        "AnnVol": round(rets.std() * np.sqrt(TRADING_DAYS), 4),
        "MaxDD": round(max_drawdown(rets), 4),
        "HitRate": round((rets > 0).mean(), 3),
    }
    if benchmark is not None:
        out["InfoRatio"] = round(information_ratio(rets, benchmark), 3)
    return out


def break_even_cost(prices, positions, oos_start, hi: float = 200.0, tol: float = 0.1) -> float:
    """
    Round-trip cost (bps) at which out-of-sample Sharpe crosses zero, via
    bisection. Returns NaN if Sharpe is already <= 0 at zero cost, i.e. there is
    no positive gross edge for cost to erode (the loss is in the signal, not the
    friction). Useful for stating exactly how cost-fragile any edge is.
    """
    s0 = sharpe_ratio(backtest(prices, positions, tcost_bps=0.0).loc[oos_start:])
    if s0 <= 0:
        return float("nan")
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        s = sharpe_ratio(backtest(prices, positions, tcost_bps=mid).loc[oos_start:])
        if abs(s) < 1e-4 or (hi - lo) < tol:
            return mid
        if s > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
