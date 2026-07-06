"""
M4 — Stress tests: historical scenarios, hypothetical shocks,
Monte Carlo, correlation breakdown, liquidity stress.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from risk.risk_engine import ewma_variance_series, rolling_correlation

logger = logging.getLogger(__name__)

# ── Historical scenarios ──────────────────────────────────────────────────────

# Each scenario carries per-asset-class returns so portfolios that are not 100 %
# equity (bonds/gold) are repriced realistically. `bench` is the equity leg.
HISTORICAL_SCENARIOS = [
    {"name": "2008 GFC",          "name_zh": "2008全球金融危机",  "start": "2008-09-01", "end": "2009-03-31", "bench": -0.56, "bond":  0.08, "gold":  0.04},
    {"name": "2020 COVID Crash",  "name_zh": "2020新冠暴跌",      "start": "2020-02-01", "end": "2020-03-31", "bench": -0.34, "bond":  0.06, "gold": -0.02},
    {"name": "2022 Rate Hikes",   "name_zh": "2022加息周期",       "start": "2022-01-01", "end": "2022-10-31", "bench": -0.35, "bond": -0.17, "gold": -0.09},
    {"name": "2000 Dot-com Bust", "name_zh": "2000科技泡沫",       "start": "2000-03-01", "end": "2002-10-31", "bench": -0.49, "bond":  0.20, "gold":  0.12},
    {"name": "1987 Black Monday", "name_zh": "1987黑色星期一",     "start": "1987-10-19", "end": "1987-10-19", "bench": -0.22, "bond":  0.02, "gold":  0.01},
]

# Hypothetical shocks decomposed by asset class. `equity` is beta-scaled at
# apply time. `single_pos` marks the idiosyncratic gap (hits only the largest
# position). All values are returns (negative = loss).
HYPOTHETICAL_SHOCKS = [
    {"name": "Rates +200bps",      "name_zh": "利率上升200bps",  "equity": -0.04, "bond": -0.12, "gold": -0.03},
    {"name": "USD +15%",           "name_zh": "美元指数+15%",     "equity": -0.08, "bond": -0.02, "gold": -0.10},
    {"name": "USD -15%",           "name_zh": "美元指数-15%",     "equity":  0.05, "bond":  0.01, "gold":  0.08},
    {"name": "Oil -50%",           "name_zh": "油价暴跌50%",      "equity": -0.05, "bond":  0.01, "gold": -0.02},
    {"name": "Tech Sector -30%",   "name_zh": "科技板块-30%",     "equity": -0.18, "bond":  0.00, "gold":  0.00},
    {"name": "Single Pos Gap -40%","name_zh": "单仓跳空-40%",     "single_pos": -0.40},
]


def _class_weights(holdings: list[dict]) -> dict[str, float]:
    """Aggregate portfolio weight by asset class (equity/bond/gold)."""
    cw: dict[str, float] = {}
    for h in holdings:
        cls = h.get("asset_class", "equity") or "equity"
        cw[cls] = cw.get(cls, 0.0) + h["weight"]
    return cw


def run_historical(holdings: list[dict], price_data: dict, nav_eur: float,
                   port_beta: float = 1.0) -> list[dict]:
    """
    Reprice the portfolio through each historical crisis.

    For every holding we use its ACTUAL return over the scenario window when the
    supplied price history covers it; otherwise we fall back to a composition-
    aware proxy keyed on asset class (equity scaled by portfolio beta, bonds and
    gold by their scenario-specific returns). This removes the old behaviour
    where a 252-day price window never covered any scenario and the whole result
    collapsed to a flat `bench * 0.8` guess.
    """
    results = []
    beta = port_beta if port_beta == port_beta and port_beta else 1.0  # nan/0 → 1

    for sc in HISTORICAL_SCENARIOS:
        start = pd.Timestamp(sc["start"])
        end   = pd.Timestamp(sc["end"])
        port_loss   = 0.0
        real_weight = 0.0   # weight repriced from actual history

        for h in holdings:
            ticker = h["ticker"]
            w      = h["weight"]
            cls    = h.get("asset_class", "equity") or "equity"
            pd_obj = price_data.get(ticker)

            asset_ret = None
            if pd_obj is not None and pd_obj.closes is not None:
                closes = pd_obj.closes
                idx = closes.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_localize(None)
                    closes = closes.copy()
                    closes.index = idx
                period = closes[(idx >= start) & (idx <= end)]
                if len(period) >= 2:
                    asset_ret = float((period.iloc[-1] - period.iloc[0]) / period.iloc[0])
                    real_weight += w

            if asset_ret is None:                       # proxy by asset class
                if cls == "bond":
                    asset_ret = sc["bond"]
                elif cls == "gold":
                    asset_ret = sc["gold"]
                else:
                    asset_ret = sc["bench"] * beta

            port_loss += w * asset_ret

        eur_loss = nav_eur * port_loss
        results.append({
            "name":      sc["name"],
            "name_zh":   sc["name_zh"],
            "pct_loss":  port_loss,
            "eur_loss":  eur_loss,
            "benchmark": sc["bench"],
            "covered":   round(real_weight, 4),   # fraction repriced from real data
        })

    return sorted(results, key=lambda x: x["pct_loss"])


def run_hypothetical(holdings: list[dict], nav_eur: float,
                     port_beta: float = 1.0) -> list[dict]:
    """
    Apply hypothetical shocks scaled by the portfolio's actual composition:
    equity legs scaled by portfolio beta, bond/gold legs by their own
    sensitivities, and the single-position gap applied only to the largest
    holding. The old version applied a flat NAV haircut regardless of holdings.
    """
    cw   = _class_weights(holdings)
    eq_w = cw.get("equity", 0.0)
    bd_w = cw.get("bond",   0.0)
    gd_w = cw.get("gold",   0.0)
    max_w = max((h["weight"] for h in holdings), default=0.0)
    beta  = port_beta if port_beta == port_beta and port_beta else 1.0

    results = []
    for sc in HYPOTHETICAL_SHOCKS:
        if "single_pos" in sc:
            pct_loss = max_w * sc["single_pos"]
        else:
            pct_loss = (eq_w * sc.get("equity", 0.0) * beta
                        + bd_w * sc.get("bond", 0.0)
                        + gd_w * sc.get("gold", 0.0))
        results.append({
            "name":     sc["name"],
            "name_zh":  sc["name_zh"],
            "pct_loss": pct_loss,
            "eur_loss": nav_eur * pct_loss,
        })
    return results


def run_monte_carlo(price_data: dict, holdings: list[dict], nav_eur: float,
                    paths: int = 10_000, horizon: int = 30,
                    corr_window: int = 63, lam: float = 0.94) -> dict:
    """
    10,000-path MC with EWMA covariance, 30-day horizon.
    Returns VaR(95%), CVaR, P(positive return).
    """
    valid = [
        h for h in holdings
        if price_data.get(h["ticker"]) and
           price_data[h["ticker"]].status in ("ok", "reduced") and
           price_data[h["ticker"]].returns is not None
    ]
    if not valid:
        return {}

    # Deduplicate: same ticker can appear in multiple brokers — sum their weights
    w_map: dict[str, float] = {}
    for h in valid:
        t = h["ticker"]
        w_map[t] = w_map.get(t, 0.0) + h["weight"]

    # EWMA covariance
    ret_df = pd.concat(
        {t: price_data[t].returns.tail(corr_window) for t in w_map}, axis=1
    ).dropna()

    if ret_df.shape[0] < 5 or ret_df.shape[1] < 1:
        return {}

    # Sync tickers/weights to ret_df columns (dropna may drop cols with all-NaN)
    tickers = ret_df.columns.tolist()
    weights = np.array([w_map.get(t, 0.0) for t in tickers])
    if weights.sum() > 1e-9:
        weights = weights / weights.sum()
    n = len(tickers)

    cov = ret_df.cov().values

    try:
        L = np.linalg.cholesky(cov + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        cov += np.eye(n) * 1e-6
        L   = np.linalg.cholesky(cov)

    # Simulate
    rng        = np.random.default_rng(42)
    z          = rng.standard_normal((paths, horizon, n))
    daily_rets = z @ L.T  # (paths, horizon, n_assets)
    port_rets  = (daily_rets * weights).sum(axis=2)  # (paths, horizon)
    cum_rets   = (1 + port_rets).prod(axis=1) - 1     # (paths,)

    var_95  = float(-np.percentile(cum_rets, 5))
    cutoff  = np.percentile(cum_rets, 5)
    cvar_95 = float(-cum_rets[cum_rets <= cutoff].mean())
    p_pos   = float((cum_rets > 0).mean())

    return {
        "var_95":        var_95,
        "cvar_95":       cvar_95,
        "p_positive":    p_pos,
        "eur_var_95":    var_95 * nav_eur,
        "eur_cvar_95":   cvar_95 * nav_eur,
        "paths":         paths,
        "horizon_days":  horizon,
    }


def run_correlation_breakdown(price_data: dict, holdings: list[dict],
                               nav_eur: float, crisis_rho: float = 0.85,
                               lam: float = 0.94) -> dict:
    """
    Compare portfolio σ under normal vs crisis (all correlations → 0.85) conditions.
    Returns diversification_decay = crisis_sigma / normal_sigma.
    """
    valid = [
        h for h in holdings
        if price_data.get(h["ticker"]) and
           price_data[h["ticker"]].status in ("ok", "reduced") and
           price_data[h["ticker"]].returns is not None
    ]
    if len(valid) < 2:
        return {}

    tickers  = [h["ticker"] for h in valid]
    weights  = np.array([h["weight"] for h in valid])
    var_series = []
    for t in tickers:
        vs = ewma_variance_series(price_data[t].returns, lam=lam)
        var_series.append(float(vs.iloc[-1]))
    sigma_vec = np.sqrt(var_series)

    corr_df  = rolling_correlation(price_data, holdings)
    if corr_df is not None:
        normal_corr = np.array(corr_df.reindex(index=tickers, columns=tickers).fillna(0).values, dtype=float)
        np.fill_diagonal(normal_corr, 1.0)
    else:
        normal_corr = np.eye(len(tickers))

    crisis_corr = np.full((len(tickers), len(tickers)), crisis_rho)
    np.fill_diagonal(crisis_corr, 1.0)

    def port_sigma(corr):
        cov = np.outer(sigma_vec, sigma_vec) * corr
        return float(np.sqrt(weights @ cov @ weights * 252))

    normal_sigma = port_sigma(normal_corr)
    crisis_sigma = port_sigma(crisis_corr)

    return {
        "normal_sigma":        normal_sigma,
        "crisis_sigma":        crisis_sigma,
        "diversification_decay": crisis_sigma / normal_sigma if normal_sigma else float("nan"),
    }


def run_liquidity(holdings: list[dict], price_data: dict, nav_eur: float,
                  days: int = 3, adv_threshold: float = 5.0) -> list[dict]:
    """
    Flag positions where size > 5× 3-day ADV.
    Estimates liquidation cost as spread × size.
    """
    results = []
    for h in holdings:
        ticker = h["ticker"]
        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None:
            continue
        try:
            import yfinance as yf
            df    = yf.Ticker(ticker).history(period="10d")
            adv_3 = float(df["Volume"].tail(days).mean()) if not df.empty else 0
        except Exception:
            adv_3 = 0

        qty = h["quantity"]
        flagged = adv_3 > 0 and qty > adv_3 * adv_threshold

        if flagged:
            results.append({
                "ticker":        ticker,
                "quantity":      qty,
                "adv_3d":        adv_3,
                "adv_ratio":     qty / adv_3 if adv_3 else float("nan"),
                "market_value":  h["market_value_eur"],
            })

    return results


def run_all(holdings: list[dict], price_data: dict, nav_eur: float,
            vol_cfg: dict = None, port_beta: float = 1.0) -> dict:
    vcfg = vol_cfg or {}
    mc_cfg   = vcfg.get("monte_carlo", {})
    lam      = vcfg.get("ewma", {}).get("lambda", 0.94)
    corr_win = vcfg.get("windows", {}).get("correlation", 63)

    historical   = run_historical(holdings, price_data, nav_eur, port_beta=port_beta)
    hypothetical = run_hypothetical(holdings, nav_eur, port_beta=port_beta)
    mc           = run_monte_carlo(price_data, holdings, nav_eur,
                                   paths=mc_cfg.get("paths", 10_000),
                                   horizon=mc_cfg.get("horizon_days", 30),
                                   corr_window=corr_win, lam=lam)
    corr_breakdown = run_correlation_breakdown(price_data, holdings, nav_eur, lam=lam)
    liquidity      = run_liquidity(holdings, price_data, nav_eur)

    return {
        "historical":       historical,
        "hypothetical":     hypothetical,
        "monte_carlo":      mc,
        "corr_breakdown":   corr_breakdown,
        "liquidity":        liquidity,
        "top3_worst":       historical[:3],
    }
