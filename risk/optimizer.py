"""
Markowitz mean-variance optimizer.
  compute_frontier()         — Monte Carlo cloud + max-Sharpe portfolio
  rebalancing_suggestions()  — 3-tier rebalancing vs optimal weights
  marginal_impact()          — portfolio impact of adding a candidate asset
  ff_implied_mu()            — Fama-French 5-factor expected returns (replaces historical mean)
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
_FF5_CACHE   = Path("cache/ff5_daily.pkl")


def fetch_ff5() -> Optional[pd.DataFrame]:
    """Download Fama-French 5-factor daily data from Ken French's library; cache for 20 h."""
    import time
    if _FF5_CACHE.exists() and (time.time() - _FF5_CACHE.stat().st_mtime) < 72000:
        try:
            with open(_FF5_CACHE, "rb") as f:
                return _normalize_ff5_index(pickle.load(f))
        except Exception:
            pass
    try:
        import pandas_datareader.data as web
        raw = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")[0]
        ff5 = _normalize_ff5_index(raw / 100.0)   # percent → decimal
        _FF5_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_FF5_CACHE, "wb") as fh:
            pickle.dump(ff5, fh)
        logger.info(f"FF5 daily data downloaded: {len(ff5)} rows "
                    f"({ff5.index[0].date()} – {ff5.index[-1].date()})")
        return ff5
    except Exception as exc:
        logger.warning(f"FF5 fetch failed: {exc} — will fall back to historical mean")
        return None


def _normalize_ff5_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce FF5 to a tz-naive DatetimeIndex.

    pandas_datareader returns a PeriodIndex for daily Fama-French data, which
    has no .tz attribute and so blows up every downstream `_tz_naive` helper
    that expects a DatetimeIndex. Normalize once at the source.
    """
    if df is None:
        return df
    idx = df.index
    if isinstance(idx, pd.PeriodIndex):
        df = df.set_axis(idx.to_timestamp())
    elif isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df = df.set_axis(idx.tz_localize(None))
    return df


def ff_implied_mu(ret_df: pd.DataFrame, rf: float = 0.035) -> np.ndarray:
    """
    Fama-French 5-factor implied annualised expected returns.

    For each asset i:
      excess_i = alpha_i + β₁·Mkt-RF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε
      E[R_i]   = rf  +  (alpha_i + Σ βₖ · E[fₖ])  × 252

    Factor premiums are taken from the full FF history (daily mean × 252) for
    stability.  Alpha is included so stock-specific outperformance is captured.

    Falls back to ret_df.mean() × 252 when FF data is unavailable or fewer
    than 60 days align with the return series.
    """
    ff5 = fetch_ff5()
    if ff5 is None:
        return ret_df.mean().values * 252

    # Normalize indexes for the join:
    #   • yfinance returns a tz-aware DatetimeIndex
    #   • pandas_datareader's FF5 returns a PeriodIndex (no .tz attribute)
    # Coerce both to a tz-naive DatetimeIndex before joining.
    def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if isinstance(idx, pd.PeriodIndex):
            return df.set_axis(idx.to_timestamp())
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            return df.set_axis(idx.tz_localize(None))
        return df

    aligned = _tz_naive(ret_df).join(
        _tz_naive(ff5)[_FF5_FACTORS + ["RF"]], how="inner"
    )
    if len(aligned) < 60:
        logger.warning(f"Only {len(aligned)} days overlap with FF5 — using historical mean μ")
        return ret_df.mean().values * 252

    # Use only factor data up to the last training date — avoids lookahead in
    # walk-forward backtests where ret_df is a historical training slice.
    # In production ret_df covers the most recent window so this is equivalent
    # to the prior full-history approach.
    factor_prem_ann = aligned[_FF5_FACTORS].mean().values * 252

    X   = aligned[_FF5_FACTORS].values
    X_c = np.column_stack([np.ones(len(X)), X])   # intercept + 5 factors

    mu_out = np.empty(len(ret_df.columns))
    for i, col in enumerate(ret_df.columns):
        excess        = aligned[col].values - aligned["RF"].values
        coeffs, *_    = np.linalg.lstsq(X_c, excess, rcond=None)
        alpha_daily   = coeffs[0]
        betas         = coeffs[1:]
        mu_out[i]     = rf + alpha_daily * 252 + betas @ factor_prem_ann

    logger.debug("FF5 μ: " + ", ".join(f"{c}={v:.3f}"
                 for c, v in zip(ret_df.columns, mu_out)))
    return mu_out


def _stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
           rf: float = 0.035) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 1e-9 else -99.0
    return ret, vol, sharpe


def _max_sharpe(mu: np.ndarray, cov: np.ndarray,
                rf: float = 0.035,
                w_bounds: tuple = (0.005, 0.30)) -> Optional[np.ndarray]:
    n    = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [w_bounds] * n
    rng  = np.random.default_rng(42)

    # Try uniform start then 4 random starts; relax ftol progressively
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(4)]
    for w0 in starts:
        w0 = np.clip(w0, w_bounds[0], w_bounds[1])
        w0 = w0 / w0.sum()
        for ftol in (1e-9, 1e-7, 1e-5):
            res = minimize(
                lambda w: -_stats(w, mu, cov, rf)[2],
                w0, method="SLSQP",
                bounds=bnds, constraints=cons,
                options={"maxiter": 800, "ftol": ftol},
            )
            if res.success:
                return res.x
    logger.warning("Max-Sharpe: all starting points failed")
    return None


def _trace_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.035,
    n_points: int = 80,
) -> list[tuple[float, float]]:
    """
    Parametric minimum-variance frontier: for each target return in
    [min_asset_ret, max_asset_ret], solve the QP

        min  w' Σ w
        s.t. w' μ = target_ret,  w' 1 = 1,  w ≥ 0

    Returns a list of (vol, ret) pairs tracing the hyperbolic boundary.
    Only the *efficient* half (ret ≥ global-minimum-variance portfolio) is
    returned, as the lower half is dominated and never optimal.
    """
    n = len(mu)
    if n < 2:
        return []

    ret_lo = float(mu.min())
    ret_hi = float(mu.max())
    if ret_hi - ret_lo < 1e-6:
        return []

    curve: list[tuple[float, float]] = []
    # Extra headroom above max individual return so the curve extends fully
    for target in np.linspace(ret_lo, ret_hi * 1.05, n_points):
        cons = [
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
            {"type": "eq", "fun": lambda w: float(w.sum()) - 1.0},
        ]
        w0  = np.ones(n) / n
        res = minimize(
            lambda w: float(w @ cov @ w),
            w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success:
            continue

        # SLSQP can violate bounds slightly — clip and renormalise strictly
        w = np.clip(res.x, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum < 1e-9:
            continue
        w = w / w_sum

        # Accept only if the return constraint is actually satisfied
        actual_ret = float(w @ mu)
        tol = max(0.005, 0.03 * abs(target))
        if abs(actual_ret - target) > tol:
            continue

        vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
        curve.append((vol, actual_ret))

    if not curve:
        return []

    # Keep only the efficient frontier: ret ≥ global-min-variance portfolio
    min_vol_ret = min(curve, key=lambda p: p[0])[1]
    efficient   = [(v, r) for v, r in curve if r >= min_vol_ret - 1e-6]
    return sorted(efficient, key=lambda p: p[0])


def compute_frontier(price_data: dict, holdings: list[dict],
                     rf: float = 0.035, n_mc: int = 1200) -> dict:
    """
    Build risk-return frontier data for visualisation + optimisation.
    Only equity holdings with ok/reduced price history are used.

    Returns dict:
      mc      : list of (vol, ret, sharpe) for n_mc random portfolios
      current : {vol, ret, sharpe, weights}
      assets  : [{ticker, vol, ret}, ...]
      optimal : {vol, ret, sharpe, weights} | None
    """
    valid = [
        h for h in holdings
        if h.get("asset_class", "equity") == "equity"
        and price_data.get(h["ticker"])
        and price_data[h["ticker"]].status in ("ok", "reduced")
        and price_data[h["ticker"]].returns is not None
    ]
    if len(valid) < 2:
        logger.info("Frontier: fewer than 2 equity positions with price history — skipping")
        return {}

    # Deduplicate: same ticker can appear across brokers — combine weights
    w_map: dict[str, float] = {}
    for h in valid:
        t = h["ticker"]
        w_map[t] = w_map.get(t, 0.0) + h["weight"]

    ret_df = pd.concat(
        {t: price_data[t].returns for t in w_map}, axis=1
    ).dropna()

    # Sync tickers/weights to actual ret_df columns after dropna
    tickers = ret_df.columns.tolist()
    w_cur   = np.array([w_map.get(t, 0.0) for t in tickers])
    w_sum   = w_cur.sum()
    if w_sum > 1e-9:
        w_cur = w_cur / w_sum  # re-normalise to equity-only weights

    mu  = ff_implied_mu(ret_df, rf)      # FF 5-factor implied expected return (annualised)
    cov = ret_df.cov().values  * 252    # annualised covariance

    # Current portfolio
    cur_ret, cur_vol, cur_sharpe = _stats(w_cur, mu, cov, rf)

    # Per-asset stats
    assets = [
        {"ticker": t, "ret": float(mu[i]), "vol": float(np.sqrt(max(cov[i, i], 0.0)))}
        for i, t in enumerate(tickers)
    ]

    # Monte Carlo portfolios (background cloud — shows full feasible set)
    rng = np.random.default_rng(42)
    mc_rows: list[tuple[float, float, float]] = []
    for _ in range(n_mc):
        w = rng.dirichlet(np.ones(len(tickers)))
        mc_rows.append(_stats(w, mu, cov, rf))

    # Max-Sharpe portfolio
    opt_w = _max_sharpe(mu, cov, rf)
    optimal: Optional[dict] = None
    if opt_w is not None:
        o_ret, o_vol, o_sharpe = _stats(opt_w, mu, cov, rf)
        optimal = {
            "vol":     o_vol,
            "ret":     o_ret,
            "sharpe":  o_sharpe,
            "weights": {t: float(opt_w[i]) for i, t in enumerate(tickers)},
        }
    else:
        logger.warning("Max-Sharpe portfolio could not be computed")

    # Efficient frontier curve: min-variance portfolio at each target return
    frontier_curve = _trace_frontier(mu, cov, rf, n_points=80)

    return {
        "mc":             mc_rows,
        "frontier_curve": frontier_curve,   # list of (vol, ret) — the hyperbola
        "current": {
            "vol": cur_vol, "ret": cur_ret, "sharpe": cur_sharpe,
            "weights": {t: float(w_cur[i]) for i, t in enumerate(tickers)},
        },
        "assets":  assets,
        "optimal": optimal,
    }


def rebalancing_suggestions(holdings: list[dict], frontier: dict,
                             nav_eur: float) -> list[dict]:
    """
    Compare current weights to max-Sharpe optimal weights and generate
    3-tier rebalancing suggestions.

    Tier 1 — Act:   |delta| >= 5%, no Irish CGT obstacle
    Tier 2 — Watch: 2% <= |delta| < 5%, or reducing a winner (Irish CGT 33% applies)
    Tier 3 — Hold:  |delta| < 2% (trade cost likely exceeds benefit)
    """
    if not frontier or not frontier.get("optimal"):
        return []

    opt_w = frontier["optimal"]["weights"]
    h_map = {h["ticker"]: h for h in holdings}
    rows  = []

    for ticker, ow in opt_w.items():
        h   = h_map.get(ticker)
        if h is None:
            continue
        cw    = h.get("weight", 0.0)
        delta = ow - cw
        absd  = abs(delta)
        # Irish CGT flag (33%): reducing a position with unrealised gains
        cgt_flag = (delta < 0) and (h.get("unrealised_pnl_eur", 0) > 0)

        if absd < 0.02:
            tier, action, action_zh = 3, "Hold",  "维持"
        elif cgt_flag or absd < 0.05:
            tier, action, action_zh = 2, "Watch", "观察"
        else:
            tier, action, action_zh = 1, "Act",   "立即行动"

        rows.append({
            "ticker":      ticker,
            "cur_weight":  cw,
            "opt_weight":  ow,
            "delta":       delta,
            "eur_change":  absd * nav_eur,
            "tier":        tier,
            "action":      action,
            "action_zh":   action_zh,
            "direction":   "▲ Add" if delta > 0 else "▼ Cut",
            "direction_zh":"加仓"  if delta > 0 else "减仓",
            "cgt_flag":    cgt_flag,
        })

    return sorted(rows, key=lambda x: (x["tier"], -abs(x["delta"])))


def marginal_impact(
    candidate_ticker: str,
    candidate_pd,
    holdings: list[dict],
    price_data: dict,
    rf: float = 0.035,
) -> Optional[dict]:
    """
    Compute the marginal portfolio impact of adding candidate_ticker.

    Compares max-Sharpe optimal portfolio *without* vs *with* the candidate.
    Returns None when there is insufficient aligned price history.

    Keys returned:
      ticker, suggested_weight, delta_sharpe, delta_vol,
      corr_to_portfolio, ann_ret, ann_vol
    """
    if (candidate_pd is None
            or candidate_pd.status not in ("ok", "reduced")
            or candidate_pd.returns is None):
        return None

    valid = [
        h for h in holdings
        if h.get("asset_class", "equity") == "equity"
        and price_data.get(h["ticker"])
        and price_data[h["ticker"]].status in ("ok", "reduced")
        and price_data[h["ticker"]].returns is not None
    ]
    if not valid:
        return None

    # Deduplicate: same ticker across brokers — combine weights
    w_map: dict[str, float] = {}
    for h in valid:
        t = h["ticker"]
        w_map[t] = w_map.get(t, 0.0) + h["weight"]

    # Align returns: existing equity holdings (deduped) + candidate
    ret_dict = {t: price_data[t].returns for t in w_map}
    ret_dict[candidate_ticker] = candidate_pd.returns
    ret_df = pd.concat(ret_dict, axis=1).dropna()
    if len(ret_df) < 21:
        logger.info(f"marginal_impact {candidate_ticker}: only {len(ret_df)} aligned days — skipping")
        return None

    # Sync to actual columns; candidate is always last
    all_cols   = ret_df.columns.tolist()
    base_cols  = [c for c in all_cols if c != candidate_ticker]
    n          = len(base_cols)
    w_cur      = np.array([w_map.get(t, 0.0) for t in base_cols])
    if w_cur.sum() > 1e-9:
        w_cur = w_cur / w_cur.sum()

    # Reorder ret_df so candidate is last column
    ret_df = ret_df[base_cols + [candidate_ticker]]

    mu_all  = ff_implied_mu(ret_df, rf)
    # Tikhonov regularisation: guards against near-singular covariance matrices
    cov_all = ret_df.cov().values * 252 + np.eye(n + 1) * 1e-8

    # Baseline: optimal portfolio WITHOUT candidate
    mu_base  = mu_all[:n]
    cov_base = cov_all[:n, :n]
    opt_base = _max_sharpe(mu_base, cov_base, rf)
    if opt_base is not None:
        _, base_vol, base_sharpe = _stats(opt_base, mu_base, cov_base, rf)
    else:
        _, base_vol, base_sharpe = _stats(w_cur, mu_base, cov_base, rf)

    # Optimal WITH candidate — full optimizer, fallback to grid scan
    opt_with = _max_sharpe(mu_all, cov_all, rf)
    if opt_with is not None:
        _, opt_vol, opt_sharpe = _stats(opt_with, mu_all, cov_all, rf)
        suggested_weight = float(opt_with[-1])
    else:
        # Grid scan: hold existing weights fixed (normalised), vary candidate weight
        best_sharpe = -np.inf
        best_w_c    = 0.005
        for w_c in np.arange(0.005, 0.31, 0.01):
            w_full = np.append(w_cur * (1 - w_c), w_c)
            _, _, sh = _stats(w_full, mu_all, cov_all, rf)
            if sh > best_sharpe:
                best_sharpe = sh
                best_w_c    = w_c
        w_full = np.append(w_cur * (1 - best_w_c), best_w_c)
        _, opt_vol, opt_sharpe = _stats(w_full, mu_all, cov_all, rf)
        suggested_weight = float(best_w_c)

    # Correlation of candidate to current equity portfolio
    port_ret = (ret_df[base_cols] * w_cur).sum(axis=1)
    corr     = float(port_ret.corr(ret_df[candidate_ticker]))

    return {
        "ticker":            candidate_ticker,
        "suggested_weight":  suggested_weight,
        "delta_sharpe":      opt_sharpe - base_sharpe,
        "delta_vol":         opt_vol    - base_vol,
        "corr_to_portfolio": corr,
        "ann_ret":           float(mu_all[-1]),
        "ann_vol":           float(np.sqrt(max(cov_all[-1, -1], 0.0))),
    }
