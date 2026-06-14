"""
Data acquisition for the crypto stat-arb study.

Single source: ccxt + Binance US.
  - select_universe(): rank all USDT pairs by 24h quote volume, take the top-N.
    Volume is the right liquidity screen for stat-arb (we need names we can
    actually trade in and out of), and it keeps everything on one venue.
  - build_price_panel(): pull daily OHLCV closes for those symbols, align,
    drop sparse names, cache to CSV.

US users: exchange_id='binanceus'. To use the larger main venue (more coins,
needs unrestricted access), pass exchange_id='binance' - nothing else changes.

Workflow: fetch once -> cache CSV -> all analysis reads local via load_panel().
"""

from __future__ import annotations

import os
import time

import pandas as pd


def _make_exchange(exchange_id: str = "binanceus"):
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})  # ccxt self-throttles
    ex.load_markets()
    return ex


def _retry(fn, *, retries: int = 5, base_wait: float = 5.0):
    """Retry a ccxt call with exponential backoff on transient errors."""
    import ccxt
    for attempt in range(retries):
        try:
            return fn()
        except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
            if attempt == retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"  {type(e).__name__}; waiting {wait:.0f}s, retry {attempt + 1}/{retries - 1}")
            time.sleep(wait)
    raise RuntimeError("unreachable")

"""
for pairs: any param lower than 80, it will not genearte enough pairs to be meaningful,
first trial used 30 and got only 16 usable coins for analysis 
"""
def select_universe(
    exchange,
    n: int = 80, 
    quote: str = "USDT",
    exclude: tuple[str, ...] = ("USDC", "BUSD", "DAI", "TUSD", "FDUSD"),
) -> list[str]:
    """
    Top-n most-liquid {base}/{quote} spot pairs by 24h quote volume.
    Stablecoin bases are excluded (a stable/stable pair has no spread to trade).
    """
    tickers = _retry(exchange.fetch_tickers)

    rows = []
    for sym, t in tickers.items():
        if not sym.endswith(f"/{quote}"):
            continue
        base = sym.split("/")[0]
        if base in exclude:
            continue
        qv = t.get("quoteVolume")
        if qv:
            rows.append((sym, qv))

    rows.sort(key=lambda r: r[1], reverse=True)
    return [sym for sym, _ in rows[:n]]


def fetch_daily_ohlcv(exchange, symbol: str, since_days: int = 365, timeframe: str = "1d") -> "pd.Series | None":
    """Daily close prices for one symbol, paginated. Date-indexed Series or None."""
    if symbol not in exchange.markets:
        return None

    cursor = exchange.milliseconds() - since_days * 24 * 60 * 60 * 1000
    rows: list[list] = []

    while True:
        batch = _retry(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000))
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "vol"]).drop_duplicates("ts_ms")
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.normalize()
    s = df.set_index("date")["close"]
    s.name = symbol.split("/")[0]
    return s


def build_price_panel(
    symbols: list[str],
    since_days: int = 365,
    min_coverage: float = 0.90,
    exchange_id: str = "binanceus",
    cache_path: "str | None" = None,
    exchange=None,
) -> pd.DataFrame:
    """
    Wide daily-close panel (index=date, columns=base asset) for the given
    symbols. Drops names with <min_coverage data. Optionally caches to CSV.
    Pass an existing `exchange` to reuse one already created by select_universe.
    """
    exchange = exchange or _make_exchange(exchange_id)

    series = []
    for sym in symbols:
        s = fetch_daily_ohlcv(exchange, sym, since_days=since_days)
        if s is not None and not s.empty:
            series.append(s)
            print(f"  ok  {sym:14s} ({len(s)} days)")
        else:
            print(f"  --  {sym:14s} (no data)")

    if not series:
        raise RuntimeError(f"No price series fetched from {exchange_id}.")

    panel = pd.concat(series, axis=1).sort_index()
    panel = panel.loc[:, panel.notna().mean() >= min_coverage]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        panel.to_csv(cache_path)
        print(f"cached -> {cache_path}  shape={panel.shape}")
    return panel


def load_panel(cache_path: str) -> pd.DataFrame:
    """Read a previously-cached price panel from disk."""
    return pd.read_csv(cache_path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    ex = _make_exchange("binanceus")
    symbols = select_universe(ex, n=100)
    print(f"universe ({len(symbols)}): {symbols}")
    panel = build_price_panel(symbols, since_days=365, cache_path="data/panel.csv", exchange=ex)
    print(panel.shape)
    print(panel.head())
