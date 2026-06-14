# Statistical Arbitrage in Cryptocurrencies

A research study comparing three pair-selection methods for a market-neutral
pairs-trading strategy on daily cryptocurrency prices. The trading logic is held
fixed across methods, so the only variable is **how the pairs are chosen** — which
isolates the contribution of pair selection to out-of-sample performance.

The strategy trades the mean-reverting spread between related coins: when the
spread deviates far from its recent norm, short the rich leg and long the cheap
leg, betting on convergence. The research question is which selection method
produces the best out-of-sample, after-cost, risk-adjusted return.

## Research question

Pairs trading is only as good as its pairs. This project benchmarks three ways of
choosing them on identical data, identical trading rules, and an identical
walk-forward backtest, so any difference in results attributes cleanly to the
selection step rather than to the trading rule or parameters.

1. **Cointegration** — OLS on log prices, Augmented Dickey-Fuller (ADF) test on
   the residuals; keep each coin's most strongly mean-reverting partner.
2. **Correlation** — rolling log-price correlation; keep the top-*k* partners
   above a threshold. Simpler and more responsive to regime shifts.
3. **K-means clustering** *(research contribution)* — cluster coins on their
   standardised return profiles first, then test for cointegration only *within*
   a cluster. Clustering acts as a prior that shrinks the O(N²) candidate space
   and suppresses spurious pairs.

## Why these methods

**Cointegration over plain correlation.** Correlation measures co-movement, but
two assets that both trend up are highly correlated while their spread may keep
diverging — correlation does not imply mean reversion. Cointegration is stricter:
it requires a stationary linear combination of two non-stationary price series,
and that stationary residual *is* the tradeable, mean-reverting spread.

**Correlation kept as a baseline.** Cointegration testing is static and
expensive, and crypto regimes shift frequently, so static relationships can
break. Rolling correlation is lighter and more responsive. Including it tests a
real question — does the stricter, costlier method actually buy better
out-of-sample performance? — rather than assuming it does.

**K-means as a prior.** Testing every O(N²) pair both inflates false positives
(multiple-comparisons problem) and is computationally heavy. Clustering coins on
return behaviour first groups economically similar names, so cointegration is
tested only on a smaller, more plausible candidate set — cutting both compute and
spurious pairs.

## Method

- **Data**: daily close prices via the ccxt / Binance.US API. The universe is the
  top liquid USDT pairs by 24-hour quote volume (volume is the right liquidity
  screen for a strategy that needs to trade in and out). Coins with <90% data
  coverage are dropped; from a top-100 screen, 46 mature coins survive over a
  ~4-year window.
- **Signal**: a rolling hedge ratio (β) on log prices forms the spread, which is
  standardised into a rolling z-score (deviations from its recent norm).
- **No look-ahead**: every quantity used to trade on day *t* is computed from data
  up to *t*, then shifted one day, so trading on day *t* sees only information
  available at the close of *t−1*.
- **Walk-forward**: pairs are re-selected every six months on a trailing one-year
  window, then traded over the next six months. Selection and evaluation never
  overlap in time, mirroring real deployment.
- **Portfolio**: dollar-neutral, β-scaled legs; gross exposure normalised to one.
- **Evaluation**: out-of-sample only (~1,065 trading days), net of transaction
  costs, with Sharpe, annualised return/vol, max drawdown, hit rate, and the
  information ratio versus a buy-and-hold Bitcoin benchmark.

## Results

Out-of-sample, 46-coin universe, net of a 20 bps round-trip cost:

| Method | Sharpe | Ann. Return | Ann. Vol | Max DD |
|---|---|---|---|---|
| Cointegration | −0.42 | −8.9% | 21.5% | −45% |
| Correlation | −0.66 | −11.3% | 17.2% | −49% |
| K-means | −0.45 | −8.0% | 17.7% | −42% |

**The headline result is negative — and the more interesting finding is *why*.**

### Transaction-cost sensitivity locates the loss in the signal, not in friction

Because the net result sits near break-even, the conclusion's sign could hinge on
the cost assumption. Sweeping the round-trip cost isolates where the loss lives:

| Method | 0 bps | 5 bps | 10 bps | 20 bps | 30 bps | 50 bps |
|---|---|---|---|---|---|---|
| Cointegration | 0.02 | −0.09 | −0.20 | −0.42 | −0.63 | −1.06 |
| Correlation | −0.07 | −0.22 | −0.36 | −0.66 | −0.95 | −1.53 |
| K-means | 0.07 | −0.06 | −0.19 | −0.45 | −0.72 | −1.24 |

At zero cost, all three Sharpes sit within ±0.07 of zero. The negative net
results are therefore **not** a profitable strategy eroded by fees — they are a
strategy with **essentially no gross edge**, shifted below water by transaction
costs. The loss lives in the signal layer (no exploitable mean reversion in this
market), not in execution friction.

(Cost levels map to real scenarios: 5 bps = maker-only in the deepest names
(BTC/ETH/SOL); 10 bps = standard taker on liquid majors; 20 bps = blended fee +
slippage across the mixed 46-coin basket; 30 bps = small-caps with wider spreads
or a pricier venue; 50 bps = a slippage-dominated stress floor.)

### What the comparison shows

- **Cointegration and K-means beat correlation.** At zero cost, correlation is the
  only method already negative (−0.07), while cointegration (0.02) and K-means
  (0.07) reach roughly break-even. This confirms the methodological premise:
  correlation alone selects pairs whose spreads do not revert.
- **K-means is marginally best on gross Sharpe and cheaper to compute.** Its
  cluster-then-test design matches full pairwise cointegration on quality while
  testing far fewer pairs — the research increment's value is candidate-space
  reduction at no loss of quality, not a return boost.
- **All three share nearly the same cost slope** (~0.2 Sharpe lost per 10 bps),
  reflecting similar turnover.

### Attribution

The absence of gross edge traces to crypto's structure: assets carry high common
BTC-beta and move together, so independent mean-reverting spreads are scarce; and
the 2022–2025 window is dense with regime shifts (LUNA/FTX collapse through the
2024 rally) that break cointegration relationships at exactly the points a
reversion strategy is exposed.

## Relation to prior work

The cointegration and correlation baselines follow the standard Engle-Granger
pairs-trading methodology, using a public write-up (Johnny Tung's *Statistical
Arbitrage in Cryptocurrencies* series) as a reference for the two baselines — they
are the field-standard approach. This project's own contributions are: (1)
implementing the K-means cluster-then-cointegrate selector that the reference
proposes but does not build; (2) adding the transaction-cost sensitivity analysis
that attributes the loss to the signal layer; (3) independent validation on a
different data source (Binance via ccxt), time window, and universe; and (4)
refactoring into modular, reusable components.

## Repository layout

```
src/
  data_loader.py     ccxt/Binance fetch, volume-based universe, caching
  pair_selection.py  the three selectors (interchangeable interface)
  backtest.py        signals, dollar-neutral positions, backtest, metrics
run_pipeline.py      walk-forward comparison of all three methods
notebooks/
  cost_sensitivity.ipynb   cost sweep and Sharpe-vs-cost analysis
data/                cached price panel (gitignored)
results/             saved comparison tables
```

## Reproducing

```bash
conda create -n statarb python=3.11 && conda activate statarb
conda install pandas numpy statsmodels scikit-learn matplotlib scipy
pip install ccxt          # ccxt is pip-native; install last

# 1. fetch and cache the price panel
python src/data_loader.py            # selects universe, pulls daily closes -> data/panel.csv

# 2. run the method comparison
python run_pipeline.py               # -> results/method_comparison.csv

# 3. cost sensitivity: open notebooks/cost_sensitivity.ipynb and run all
```

US users hit Binance.US (`exchange_id="binanceus"`); pass `"binance"` for the
larger venue if access permits — nothing else changes.

## Limitations

- Daily data favours slower signals; intraday would change the cost/turnover
  trade-off.
- Costs are modelled as a flat round-trip bps on turnover; slippage and funding
  are not separately modelled.
- The universe is screened on current volume, introducing some survivorship bias.
- Parameters (windows, thresholds, rebalance frequency) are not swept for
  robustness — deliberately, to avoid in-sample tuning that would inflate
  out-of-sample results.

The negative direction of the result is robust; only the precise figures depend
on these choices.

## Reference
Johnny Tung (WallStreetQuant), "Statistical Arbitrage in Cryptocurrencies,"
Medium. https://medium.com/@johnnya12399/statistical-arbitrage-in-cryptocurrencies-part-1-7ed626ed9629
