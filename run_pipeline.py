"""
End-to-end pipeline: walk-forward backtest comparing the three pair-selection
methods on the same data, out-of-sample, net of costs, vs a BTC-style benchmark.

Run data_loader first to cache data/panel.csv, then:
    python run_pipeline.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

from data_loader import load_panel
from pair_selection import SELECTORS
from backtest import generate_signals, build_positions, backtest, summarise

PANEL = "data/panel.csv"   # cached panel from data_loader (run that first)
WINDOW = 90
LOOKBACK_DAYS = 365   # in-sample length for re-selecting pairs
REBALANCE = "6MS"     # re-select pairs every 6 months
ENTRY, EXIT = 1.0, 0.2
TCOST_BPS = 20.0


def walk_forward(prices: pd.DataFrame, selector, oos_start: pd.Timestamp) -> pd.Series:
    """Re-select pairs each rebalance date on a trailing window; stitch OOS returns."""
    rebal_dates = pd.date_range(oos_start, prices.index[-1], freq=REBALANCE)
    full_pos = pd.DataFrame(0.0, index=prices.loc[oos_start:].index, columns=prices.columns)

    for start in rebal_dates:
        in_start = start - pd.DateOffset(days=LOOKBACK_DAYS)
        in_end = start - pd.DateOffset(days=1)
        end = min(start + pd.DateOffset(months=6) - pd.DateOffset(days=1), prices.index[-1])

        insample = prices.loc[in_start:in_end].dropna(axis=1, how="any")
        if insample.shape[1] < 2 or insample.empty:
            continue

        pairs = selector(insample)
        if not pairs:
            continue

        sig = generate_signals(prices, pairs, window=WINDOW)
        pos = build_positions(sig, prices.columns, prices.loc[start:end].index,
                              entry=ENTRY, exit_band=EXIT)
        full_pos.loc[start:end, :] = pos.reindex(full_pos.loc[start:end].index).fillna(0)

    return backtest(prices, full_pos, tcost_bps=TCOST_BPS)


def main():
    prices = load_panel(PANEL)
    print(f"Panel: {prices.shape[0]} dates x {prices.shape[1]} coins")

    # Benchmark: buy-and-hold the first column (stands in for BTC).
    bench = np.log(prices.iloc[:, 0].replace(0, np.nan).ffill()).diff().dropna()

    oos_start = prices.index[0] + pd.DateOffset(days=LOOKBACK_DAYS + 30)
    rows = {}
    for name, selector in SELECTORS.items():
        net = walk_forward(prices, selector, oos_start)
        net_oos = net.loc[oos_start:]
        rows[name] = summarise(net_oos, benchmark=bench.reindex(net_oos.index))
        print(f"  {name:14s} pairs/last selection done, OOS days={len(net_oos)}")

    table = pd.DataFrame(rows).T
    print("\n=== Out-of-sample comparison (net of costs) ===")
    print(table.to_string())
    os.makedirs("results", exist_ok=True)
    table.to_csv("results/method_comparison.csv")
    print("\nSaved -> results/method_comparison.csv")


if __name__ == "__main__":
    main()