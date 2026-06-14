"""
Pair-selection methods for statistical-arbitrage pairs trading.

Three approaches, increasing in sophistication, so the backtest can compare
their out-of-sample quality on the same data:

  1. cointegration  - OLS on log prices + Augmented Dickey-Fuller on residuals.
                       Picks the most strongly mean-reverting partner per coin.
  2. correlation    - rolling correlation of log prices; keep the top-k partners
                       above a threshold. Simpler, more responsive to regime shifts.
  3. kmeans         - cluster coins on standardised return profiles, then only
                       look for cointegrated partners *within* a cluster. This is
                       the original research increment in this project: it uses the
                       cluster structure as a prior to cut the candidate space and
                       reduce spurious pairs.

All functions take an in-sample price frame and return a list of (coin_i, coin_j)
tuples, so the backtester can treat them interchangeably.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# 1. Cointegration
# ----------------------------------------------------------------------------
def _adf_pvalue_for_pair(log_px: pd.DataFrame, i: str, j: str) -> tuple[float, float]:
    """OLS log(j) ~ log(i); ADF on residuals. Returns (p_value, adf_stat)."""
    x = log_px[i].to_numpy()
    y = log_px[j].to_numpy()
    model = sm.OLS(y, sm.add_constant(x)).fit()
    resid = y - model.params[1] * x - model.params[0]
    stat, pval = adfuller(resid)[0], adfuller(resid)[1]
    return pval, stat


def select_cointegrated_pairs(
    prices: pd.DataFrame, pvalue_max: float = 0.05
) -> list[tuple[str, str]]:
    """
    For each coin, keep its single most strongly mean-reverting partner
    (most negative ADF statistic) provided the relationship is significant.
    """
    log_px = np.log(prices.replace(0, np.nan).ffill().dropna(how="all", axis=1))
    coins = list(log_px.columns)
    best: dict[str, tuple[str, float, float]] = {}

    for i, j in itertools.combinations(coins, 2):
        try:
            pval, stat = _adf_pvalue_for_pair(log_px, i, j)
        except Exception:
            continue
        if pval >= pvalue_max:
            continue
        for anchor, partner in ((i, j), (j, i)):
            if anchor not in best or stat < best[anchor][2]:
                best[anchor] = (partner, pval, stat)

    pairs = {tuple(sorted((anchor, info[0]))) for anchor, info in best.items()}
    return sorted(pairs)


# ----------------------------------------------------------------------------
# 2. Correlation
# ----------------------------------------------------------------------------
def select_correlated_pairs(
    prices: pd.DataFrame, top_n: int = 3, corr_threshold: float = 0.9
) -> list[tuple[str, str]]:
    """For each coin, keep its top-n partners by absolute log-price correlation."""
    log_px = np.log(prices.replace(0, np.nan).ffill())
    corr = log_px.corr()
    pairs: set[tuple[str, str]] = set()

    for coin in corr.columns:
        ranked = corr[coin].drop(labels=[coin]).reindex(
            corr[coin].drop(labels=[coin]).abs().sort_values(ascending=False).index
        )
        ranked = ranked[ranked.abs() >= corr_threshold].head(top_n)
        for partner in ranked.index:
            pairs.add(tuple(sorted((coin, partner))))

    return sorted(pairs)


# ----------------------------------------------------------------------------
# 3. K-means clustering  (research increment)
# ----------------------------------------------------------------------------
def select_kmeans_pairs(
    prices: pd.DataFrame,
    n_clusters: int = 4,
    pvalue_max: float = 0.05,
    random_state: int = 42,
) -> list[tuple[str, str]]:
    """
    Cluster coins on their standardised daily-return series, then run the
    cointegration test only on within-cluster combinations.

    Rationale: testing every O(N^2) pair inflates false positives and is
    expensive. Clustering on return behaviour groups economically similar
    coins first, so we test a smaller, more plausible candidate set.
    """
    rets = np.log(prices.replace(0, np.nan).ffill()).diff().dropna(how="all")
    rets = rets.dropna(axis=1)
    if rets.shape[1] < 2:
        return []

    # Feature matrix: coins as rows, standardised return profile as features.
    feats = StandardScaler().fit_transform(rets.T.to_numpy())
    k = min(n_clusters, feats.shape[0])
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(feats)

    coins = list(rets.columns)
    cluster_map: dict[int, list[str]] = {}
    for coin, lab in zip(coins, labels):
        cluster_map.setdefault(lab, []).append(coin)

    log_px = np.log(prices.replace(0, np.nan).ffill())
    pairs: set[tuple[str, str]] = set()

    for members in cluster_map.values():
        if len(members) < 2:
            continue
        for i, j in itertools.combinations(members, 2):
            try:
                pval, _ = _adf_pvalue_for_pair(log_px[[i, j]].dropna(), i, j)
            except Exception:
                continue
            if pval < pvalue_max:
                pairs.add(tuple(sorted((i, j))))

    return sorted(pairs)


SELECTORS = {
    "cointegration": select_cointegrated_pairs,
    "correlation": select_correlated_pairs,
    "kmeans": select_kmeans_pairs,
}
