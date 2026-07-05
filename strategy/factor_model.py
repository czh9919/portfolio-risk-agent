"""
Fama-French 5-factor regression per stock.
  run_factor_regression()      — OLS alpha/beta/IR + SE/t for all coefficients
  portfolio_factor_exposure()  — market-value-weighted aggregate betas + pooled SE
  compute_attribution()        — decompose recent portfolio return into factor contributions
  compute_stock_attribution()  — per-stock factor contribution to recent returns
  explain_stock_drivers()      — human-readable explanation of factor drivers
  compute_robust_signal()      — multi-window + PSR + deflated IR + OOS-validated signal
  enrich_suggestions()         — attach α-based "why" reason to each rebalancing suggestion
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF — Abramowitz & Stegun rational approximation."""
    if p <= 0.0: return float("-inf")
    if p >= 1.0: return float("inf")
    q = p if p < 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + t * (c1 + t * c2)) / (1.0 + t * (d1 + t * (d2 + t * d3)))
    return -x if p < 0.5 else x


def _psr(ir_daily: float, n: int, skew: float, kurt: float) -> float:
    """
    Probabilistic Sharpe Ratio: P(true daily IR > 0).

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Φ{ IR × √(n−1) / √(1 − γ₃·IR + ((γ₄−1)/4)·IR²) }

    γ₃ = skewness of residuals
    γ₄ = kurtosis (NOT excess; normal = 3)
    """
    if ir_daily != ir_daily:
        return float("nan")
    denom_sq = 1.0 - skew * ir_daily + ((kurt - 1.0) / 4.0) * ir_daily ** 2
    if denom_sq <= 0.0:
        return float("nan")
    z = ir_daily * math.sqrt(max(n - 1, 1)) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def _expected_max_ir_daily(n_strategies: int, t_obs: int) -> float:
    """
    Expected maximum daily IR across n_strategies i.i.d. strategies, each
    estimated from t_obs observations with true IR = 0.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio":
    E[max IR] ≈ (1/√T) × ((1−γ_E)·Φ⁻¹(1−1/N) + γ_E·Φ⁻¹(1−1/(N·e)))
    """
    if n_strategies <= 1 or t_obs < 2:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = _norm_ppf(1.0 - 1.0 / n_strategies)
    z2 = _norm_ppf(1.0 - 1.0 / (n_strategies * math.e))
    return ((1.0 - g) * z1 + g * z2) / math.sqrt(t_obs)

logger = logging.getLogger(__name__)

FACTORS    = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MIN_WINDOW = 63   # minimum aligned days for reliable OLS

# (ff5_col, beta_key, se_key, t_key, label_en, label_zh)
_FACTOR_META = [
    ("Mkt-RF", "beta_mkt", "se_mkt", "t_mkt", "Market",       "市场"),
    ("SMB",    "beta_smb", "se_smb", "t_smb", "Size (SMB)",   "规模(SMB)"),
    ("HML",    "beta_hml", "se_hml", "t_hml", "Value (HML)",  "价值(HML)"),
    ("RMW",    "beta_rmw", "se_rmw", "t_rmw", "Profit (RMW)", "盈利(RMW)"),
    ("CMA",    "beta_cma", "se_cma", "t_cma", "Invest (CMA)", "投资(CMA)"),
]

# Backward-compatible alias used by report_gen and other callers
_FACTOR_LABELS = [(m[0], m[1], m[4], m[5]) for m in _FACTOR_META]


def _t_stat(val: float, se: float) -> float:
    return val / se if (se == se and se > 1e-12) else float("nan")


def _stars(t_val) -> str:
    if t_val is None or t_val != t_val:
        return ""
    a = abs(t_val)
    if a >= 3.0: return "★★★"
    if a >= 2.0: return "★★"
    if a >= 1.5: return "★"
    return ""


def run_factor_regression(
    ticker: str,
    returns: pd.Series,
    ff5: pd.DataFrame,
    window: int = 252,
) -> Optional[dict]:
    """
    OLS regression of daily excess returns on FF5 factors.

    Uses the last min(window, available) aligned days; returns None
    when fewer than MIN_WINDOW days overlap with FF5 history.

    Returns dict:
      ticker, alpha_ann, t_alpha, se_alpha, ir, r_squared,
      beta_mkt/smb/hml/rmw/cma,
      se_betas: {factor: SE},   t_betas: {factor: t-stat},
      signal ("BUY"/"SELL"/"HOLD"), n_days
    """
    def _tz_naive(idx: pd.Index) -> pd.Index:
        return idx.tz_localize(None) if idx.tz is not None else idx

    ret = returns.copy()
    ret.index = _tz_naive(ret.index)

    ff5_n = ff5.copy()
    ff5_n.index = _tz_naive(ff5_n.index)

    aligned = ret.to_frame("ret").join(ff5_n[FACTORS + ["RF"]], how="inner")

    if len(aligned) < MIN_WINDOW:
        logger.debug(f"{ticker}: only {len(aligned)} aligned days (<{MIN_WINDOW}) — skipping")
        return None

    n_use   = min(window, len(aligned))
    aligned = aligned.iloc[-n_use:]
    n       = len(aligned)

    excess = aligned["ret"].values - aligned["RF"].values
    X      = aligned[FACTORS].values
    X_c    = np.column_stack([np.ones(n), X])   # intercept + 5 factors

    coeffs, _, rank, _ = np.linalg.lstsq(X_c, excess, rcond=None)
    residuals   = excess - X_c @ coeffs
    alpha_daily = float(coeffs[0])
    betas       = coeffs[1:]

    # Full OLS covariance matrix → SE for every coefficient
    df_resid = n - X_c.shape[1]
    if df_resid > 0 and rank == X_c.shape[1]:
        s2 = float(np.dot(residuals, residuals) / df_resid)
        try:
            xxt_inv      = np.linalg.inv(X_c.T @ X_c)
            se_all       = np.sqrt(np.maximum(s2 * np.diag(xxt_inv), 0.0))
            se_alpha     = float(se_all[0])
            se_betas_arr = se_all[1:]
        except np.linalg.LinAlgError:
            se_alpha     = float("nan")
            se_betas_arr = np.full(len(FACTORS), float("nan"))
    else:
        se_alpha     = float("nan")
        se_betas_arr = np.full(len(FACTORS), float("nan"))

    t_alpha     = _t_stat(alpha_daily, se_alpha)
    t_betas_arr = np.array([
        _t_stat(float(betas[i]), float(se_betas_arr[i])) for i in range(len(FACTORS))
    ])

    # Information Ratio = (alpha_daily / σ_residuals) × √252
    sigma_res = float(np.std(residuals, ddof=1)) if n > 1 else float("nan")
    ir = ((alpha_daily / sigma_res) * np.sqrt(252)
          if sigma_res and sigma_res > 1e-12
          else float("nan"))

    # R²
    ss_tot = float(np.dot(excess - excess.mean(), excess - excess.mean()))
    r2     = 1.0 - float(np.dot(residuals, residuals)) / ss_tot if ss_tot > 1e-12 else 0.0

    # Signal driven by t_alpha (statistical significance of excess return)
    if t_alpha != t_alpha:
        signal = "HOLD"
    elif t_alpha > 1.5:
        signal = "BUY"
    elif t_alpha < -1.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    se_betas_dict = {FACTORS[i]: float(se_betas_arr[i]) for i in range(len(FACTORS))}
    t_betas_dict  = {FACTORS[i]: float(t_betas_arr[i])  for i in range(len(FACTORS))}

    return {
        "ticker":    ticker,
        "alpha_ann": alpha_daily * 252,
        "t_alpha":   t_alpha,
        "se_alpha":  se_alpha,
        "ir":        ir,
        "r_squared": r2,
        "beta_mkt":  float(betas[0]),
        "beta_smb":  float(betas[1]),
        "beta_hml":  float(betas[2]),
        "beta_rmw":  float(betas[3]),
        "beta_cma":  float(betas[4]),
        "se_betas":  se_betas_dict,
        "t_betas":   t_betas_dict,
        "signal":    signal,
        "n_days":    n,
    }


def portfolio_factor_exposure(
    holdings: list[dict],
    price_data: dict,
    ff5: pd.DataFrame,
    window: int = 252,
) -> Optional[dict]:
    """
    Market-value-weighted aggregate factor betas for equity holdings.
    Returns betas + pooled SE (L2-weighted, assumes independence across stocks)
    + approximate portfolio-level t-stats.
    """
    equity = [
        h for h in holdings
        if h.get("asset_class", "equity") == "equity"
        and price_data.get(h["ticker"])
        and price_data[h["ticker"]].returns is not None
    ]
    if not equity:
        return None

    total_mv = sum(h.get("market_value_eur", 0.0) for h in equity)
    if total_mv <= 0:
        return None

    agg     = {k: 0.0 for k in ("alpha_ann", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma")}
    var_agg = {f: 0.0 for f in FACTORS}  # accumulates w²·SE² per factor
    used    = 0.0

    for h in equity:
        ticker = h["ticker"]
        mv     = h.get("market_value_eur", 0.0)
        if mv <= 0:
            continue
        reg = run_factor_regression(ticker, price_data[ticker].returns, ff5, window)
        if reg is None:
            continue
        w = mv / total_mv
        used += w
        for key in agg:
            agg[key] += w * reg[key]
        se_b = reg.get("se_betas", {})
        for fname in FACTORS:
            se_i = se_b.get(fname, float("nan"))
            if se_i == se_i:  # not NaN
                var_agg[fname] += (w * se_i) ** 2

    if used < 1e-9:
        return None

    # Re-normalise if some holdings had no valid regression
    if used < 0.999:
        for key in agg:
            agg[key] /= used
        for fname in FACTORS:
            var_agg[fname] /= used ** 2

    # Attach portfolio-level SE and t for each factor beta
    for fname, bkey, sekey, tkey, *_ in _FACTOR_META:
        se_val     = float(np.sqrt(var_agg.get(fname, 0.0)))
        agg[sekey] = se_val
        agg[tkey]  = _t_stat(agg[bkey], se_val)

    return agg


def compute_attribution(
    holdings: list[dict],
    price_data: dict,
    ff5: pd.DataFrame,
    port_exposure: dict,
    window: int = 21,
    trading_cost_pct: float = 0.0,
) -> Optional[dict]:
    """
    Decomposes the realized equity portfolio return over the last `window`
    trading days into FF5 factor contributions + residual alpha.

    trading_cost_pct: estimated round-trip cost as a fraction of portfolio
                      value (e.g. 0.001 = 0.1%). Subtracted from gross alpha
                      to yield alpha_net.

    Beta significance uses a direct OLS on the portfolio return series
    (up to 252 days) rather than propagated individual-stock SE — this gives
    correct degrees of freedom and avoids the independence assumption.
    Propagated SE is still returned as contrib_se for contribution bounds.

    Each factor in contributions dict:
      beta, beta_se (propagated), beta_t (direct portfolio OLS),
      factor_ret, contrib, contrib_se, pct_of_total, pct_of_excess

    Top-level keys added vs prior version:
      excess_return, alpha_gross, trading_cost_pct, alpha_net,
      alpha_net_pct, attribution_error, factor_total_pct, portfolio_ols,
      alpha (backward-compat alias for alpha_gross)
    """
    def _tz_naive(idx: pd.Index) -> pd.Index:
        return idx.tz_localize(None) if idx.tz is not None else idx

    equity = [
        h for h in holdings
        if h.get("asset_class", "equity") == "equity"
        and price_data.get(h["ticker"])
        and price_data[h["ticker"]].returns is not None
    ]
    if not equity:
        return None

    total_mv = sum(h.get("market_value_eur", 0.0) for h in equity)
    if total_mv <= 0:
        return None

    # Build value-weighted portfolio return series
    port_ret: Optional[pd.Series] = None
    for h in equity:
        mv = h.get("market_value_eur", 0.0)
        if mv <= 0:
            continue
        w   = mv / total_mv
        ret = price_data[h["ticker"]].returns.copy()
        ret.index = _tz_naive(ret.index)
        port_ret = ret * w if port_ret is None else port_ret.add(ret * w, fill_value=0.0)

    if port_ret is None or len(port_ret) < 5:
        return None

    ff5_n = ff5.copy()
    ff5_n.index = _tz_naive(ff5_n.index)

    aligned = port_ret.to_frame("port").join(ff5_n, how="inner").iloc[-window:]
    n = len(aligned)
    if n < 5:
        return None

    total_return  = float(aligned["port"].sum())
    rf_total      = float(aligned["RF"].sum())
    excess_return = total_return - rf_total

    # ── Direct portfolio OLS for reliable beta t-stats ────────────────────────
    # Run over a longer window so the t-stats have proper degrees of freedom.
    # Betas here may differ from weighted-avg port_exposure; that's expected —
    # use these t-stats for significance and port_exposure betas for attribution.
    portfolio_ols: Optional[dict] = None
    full_aligned = port_ret.to_frame("port").join(ff5_n, how="inner")
    n_ols = min(252, len(full_aligned))
    ols_slice = full_aligned.iloc[-n_ols:]
    if len(ols_slice) >= MIN_WINDOW:
        excess_ols = ols_slice["port"].values - ols_slice["RF"].values
        X_ols      = ols_slice[FACTORS].values
        X_c_ols    = np.column_stack([np.ones(len(ols_slice)), X_ols])
        c_ols, _, rank_ols, _ = np.linalg.lstsq(X_c_ols, excess_ols, rcond=None)
        res_ols    = excess_ols - X_c_ols @ c_ols
        df_ols     = len(ols_slice) - X_c_ols.shape[1]
        if df_ols > 0 and rank_ols == X_c_ols.shape[1]:
            s2_ols = float(np.dot(res_ols, res_ols) / df_ols)
            try:
                xxt_ols = np.linalg.inv(X_c_ols.T @ X_c_ols)
                se_ols  = np.sqrt(np.maximum(s2_ols * np.diag(xxt_ols), 0.0))
                portfolio_ols = {
                    "n_days":    len(ols_slice),
                    "alpha_ann": float(c_ols[0]) * 252,
                    "t_alpha":   _t_stat(float(c_ols[0]), float(se_ols[0])),
                    "betas":     {FACTORS[i]: float(c_ols[i + 1])    for i in range(len(FACTORS))},
                    "t_betas":   {FACTORS[i]: _t_stat(float(c_ols[i + 1]), float(se_ols[i + 1]))
                                  for i in range(len(FACTORS))},
                    "se_betas":  {FACTORS[i]: float(se_ols[i + 1])   for i in range(len(FACTORS))},
                }
            except np.linalg.LinAlgError:
                pass

    # ── Factor contribution decomposition ─────────────────────────────────────
    contributions: dict = {}
    factor_total      = rf_total
    total_contrib_var = 0.0

    for fname, bkey, sekey, tkey, *_ in _FACTOR_META:
        beta       = port_exposure.get(bkey, 0.0)
        factor_ret = float(aligned[fname].sum())
        contrib    = beta * factor_ret

        # Propagated SE from weighted portfolio beta uncertainty
        se_b = port_exposure.get(sekey, float("nan"))
        if se_b == se_b and se_b > 0:
            c_se = se_b * abs(factor_ret)
            total_contrib_var += c_se ** 2
        else:
            c_se = float("nan")

        # Prefer direct OLS t-stat for significance; fall back to propagated
        if portfolio_ols:
            beta_t = portfolio_ols["t_betas"].get(fname, float("nan"))
        else:
            beta_t = port_exposure.get(tkey, float("nan"))

        contributions[fname] = {
            "beta":       beta,
            "beta_se":    se_b,
            "beta_t":     beta_t,
            "factor_ret": factor_ret,
            "contrib":    contrib,
            "contrib_se": c_se,
        }
        factor_total += contrib

    # Percentage decompositions
    for entry in contributions.values():
        entry["pct_of_total"]  = (entry["contrib"] / total_return
                                  if abs(total_return) > 1e-10 else float("nan"))
        entry["pct_of_excess"] = (entry["contrib"] / excess_return
                                  if abs(excess_return) > 1e-10 else float("nan"))

    alpha_gross       = total_return - factor_total
    alpha_net         = alpha_gross - trading_cost_pct
    attribution_error = (float(np.sqrt(total_contrib_var))
                         if total_contrib_var > 0 else float("nan"))

    start = str(aligned.index[0].date()  if hasattr(aligned.index[0],  "date") else aligned.index[0])
    end   = str(aligned.index[-1].date() if hasattr(aligned.index[-1], "date") else aligned.index[-1])

    return {
        "window":            n,
        "start":             start,
        "end":               end,
        "total_return":      total_return,
        "rf":                rf_total,
        "excess_return":     excess_return,
        "contributions":     contributions,
        "factor_total":      factor_total,
        "factor_total_pct":  (factor_total / total_return
                              if abs(total_return) > 1e-10 else float("nan")),
        "alpha_gross":       alpha_gross,
        "trading_cost_pct":  trading_cost_pct,
        "alpha_net":         alpha_net,
        "alpha_net_pct":     (alpha_net / total_return
                              if abs(total_return) > 1e-10 else float("nan")),
        "attribution_error": attribution_error,
        "portfolio_ols":     portfolio_ols,
        # backward compat
        "alpha":             alpha_gross,
    }


def compute_stock_attribution(
    ticker: str,
    returns: pd.Series,
    ff5: pd.DataFrame,
    reg: dict,
    window: int = 63,
) -> Optional[dict]:
    """
    Applies regression betas from run_factor_regression() to the last `window`
    trading days to decompose the stock's historical return into factor contributions.

    Returns the same structure as compute_attribution() but for a single stock
    (no portfolio weighting).  Requires at least 5 aligned days.
    """
    def _tz_naive(idx: pd.Index) -> pd.Index:
        return idx.tz_localize(None) if idx.tz is not None else idx

    ret = returns.copy()
    ret.index = _tz_naive(ret.index)
    ff5_n = ff5.copy()
    ff5_n.index = _tz_naive(ff5_n.index)

    aligned = ret.to_frame("ret").join(ff5_n, how="inner").iloc[-window:]
    n = len(aligned)
    if n < 5:
        return None

    total_return  = float(aligned["ret"].sum())
    rf_total      = float(aligned["RF"].sum())
    excess_return = total_return - rf_total
    t_betas       = reg.get("t_betas", {})

    contributions: dict = {}
    factor_total = rf_total

    for fname, bkey, sekey, tkey, *_ in _FACTOR_META:
        beta       = reg.get(bkey, 0.0)
        factor_ret = float(aligned[fname].sum())
        contrib    = beta * factor_ret
        tv         = t_betas.get(fname)
        contributions[fname] = {
            "beta":       beta,
            "beta_t":     tv,
            "factor_ret": factor_ret,
            "contrib":    contrib,
        }
        factor_total += contrib

    alpha = total_return - factor_total

    for entry in contributions.values():
        entry["pct_of_total"]  = (entry["contrib"] / total_return
                                  if abs(total_return) > 1e-10 else float("nan"))
        entry["pct_of_excess"] = (entry["contrib"] / excess_return
                                  if abs(excess_return) > 1e-10 else float("nan"))

    # Dominant = factor with highest |pct_of_total|
    dominant = max(
        ((fname, abs(e.get("pct_of_total") or 0.0)) for fname, e in contributions.items()),
        key=lambda x: x[1], default=(None, 0.0)
    )[0]

    start = str(aligned.index[0].date()  if hasattr(aligned.index[0],  "date") else aligned.index[0])
    end   = str(aligned.index[-1].date() if hasattr(aligned.index[-1], "date") else aligned.index[-1])

    return {
        "window":          n,
        "start":           start,
        "end":             end,
        "total_return":    total_return,
        "rf":              rf_total,
        "excess_return":   excess_return,
        "contributions":   contributions,
        "factor_total":    factor_total,
        "alpha":           alpha,
        "dominant_driver": dominant,
    }


# Style thresholds and labels for explain_stock_drivers
_STYLE_LABELS = {
    #             (low_bound, high_bound, label_high_en, label_low_en, label_high_zh, label_low_zh)
    "Mkt-RF": (-99,   1.3, "high-β",      "defensive",     "高贝塔",   "防御型"),
    "SMB":    (-0.2,  0.3, "small-cap",   "large-cap",     "小盘",     "大盘"),
    "HML":    (-0.2,  0.3, "value",       "growth",        "价值",     "成长"),
    "RMW":    (-0.1,  0.2, "quality",     "speculative",   "高质量",   "投机性"),
    "CMA":    (-0.1,  0.2, "conservative","aggressive inv.","保守",    "激进投资"),
}

_BKEYS = {
    "Mkt-RF": "beta_mkt", "SMB": "beta_smb",
    "HML": "beta_hml",    "RMW": "beta_rmw", "CMA": "beta_cma",
}

_FNAME_LABELS = {
    "Mkt-RF": ("Market", "市场"), "SMB": ("Size",   "规模"),
    "HML":    ("Value",  "价值"), "RMW": ("Profit", "盈利"),
    "CMA":    ("Invest", "投资"),
}


def explain_stock_drivers(
    reg: dict,
    attribution: Optional[dict] = None,
    lang: str = "en",
) -> str:
    """
    One-line explanation of what drives a stock's returns.

    Format:
      "{Factor} {±pct}{stars} ({style}) · ... · Ann.α {±X%} t={Y}{stars}"

    - Factor contributions from 63-day attribution window (top 3 by |pct|)
    - Style label appended when beta is outside the neutral range
    - Annual alpha from 252-day regression appended if t(α) ≥ 1.5
    """
    zh    = (lang == "zh")
    parts: list[str] = []

    # ── Recent factor contributions (63d attribution) ──────────────────────────
    if attribution and attribution.get("contributions"):
        sorted_f = sorted(
            attribution["contributions"].items(),
            key=lambda x: abs(x[1].get("pct_of_total") or 0.0),
            reverse=True,
        )
        n_shown = 0
        for fname, info in sorted_f:
            pct = info.get("pct_of_total")
            if pct is None or pct != pct or abs(pct) < 0.05:
                continue

            beta = info.get("beta", 0.0) or 0.0
            tv   = info.get("beta_t")
            fn   = _FNAME_LABELS.get(fname, (fname, fname))[1 if zh else 0]
            sgn  = "+" if pct >= 0 else ""
            st   = _stars(tv)

            # Style label when beta is outside neutral zone
            lo, hi, lbl_hi_en, lbl_lo_en, lbl_hi_zh, lbl_lo_zh = _STYLE_LABELS.get(
                fname, (-99, 99, "", "", "", "")
            )
            if beta > hi:
                style = lbl_hi_zh if zh else lbl_hi_en
            elif beta < lo:
                style = lbl_lo_zh if zh else lbl_lo_en
            else:
                style = ""

            entry = f"{fn} {sgn}{pct*100:.0f}%{st}"
            if style:
                entry += f"({style})"
            parts.append(entry)
            n_shown += 1
            if n_shown >= 3:
                break

    # ── Long-term alpha significance (252d regression) ─────────────────────────
    t_alpha   = reg.get("t_alpha")
    alpha_ann = reg.get("alpha_ann", 0.0) or 0.0
    if t_alpha is not None and t_alpha == t_alpha and abs(t_alpha) >= 1.5:
        st  = _stars(t_alpha)
        sgn = "+" if alpha_ann >= 0 else ""
        lbl = "年化α" if zh else "Ann.α"
        parts.append(f"{lbl} {sgn}{alpha_ann*100:.1f}%{st}")

    return " · ".join(parts) if parts else "—"


def compute_robust_signal(
    ticker: str,
    returns: pd.Series,
    ff5: pd.DataFrame,
    n_strategies: int = 50,
    windows: tuple = (63, 126, 252),
    oos_train: int = 126,
    oos_test:  int = 21,
) -> Optional[dict]:
    """
    Statistically robust trading signal that addresses three weaknesses of
    single-window α + t-value screening:

    1. Window dependency  → multi-window consistency (63/126/252 days)
    2. IR inflation       → Probabilistic IR (PSR) + Deflated IR corrected for
                            n_strategies multiple-testing bias
    3. No OOS validation  → rolling forward-test hit rate

    Threshold logic for robust_signal:
      BUY  ← PSR > 0.85  AND  consistent positive α in ≥ 2 windows
             AND  ir_deflated > 0  AND  oos_hit_rate > 0.52 (when available)
      SELL ← same conditions, negative α direction
      HOLD ← otherwise

    Returns None when fewer than MIN_WINDOW aligned days are available.
    """
    def _tz_naive(idx: pd.Index) -> pd.Index:
        return idx.tz_localize(None) if idx.tz is not None else idx

    ret = returns.copy()
    ret.index = _tz_naive(ret.index)
    ff5_n = ff5.copy()
    ff5_n.index = _tz_naive(ff5_n.index)

    aligned = ret.to_frame("ret").join(ff5_n[FACTORS + ["RF"]], how="inner")
    if len(aligned) < MIN_WINDOW:
        return None

    # ── 1. Multi-window α consistency ────────────────────────────────────────
    win_results: dict[int, dict] = {}
    for w in windows:
        if len(aligned) < w:
            continue
        sl    = aligned.iloc[-w:]
        exc   = (sl["ret"] - sl["RF"]).values
        X_c   = np.column_stack([np.ones(w), sl[FACTORS].values])
        coeff, _, rank, _ = np.linalg.lstsq(X_c, exc, rcond=None)
        resid = exc - X_c @ coeff
        a_d   = float(coeff[0])
        df    = w - X_c.shape[1]
        if df > 0 and rank == X_c.shape[1]:
            s2 = float(np.dot(resid, resid) / df)
            try:
                xxt  = np.linalg.inv(X_c.T @ X_c)
                se_a = float(np.sqrt(max(s2 * xxt[0, 0], 0.0)))
                ta   = a_d / se_a if se_a > 1e-12 else float("nan")
            except np.linalg.LinAlgError:
                ta = float("nan")
        else:
            ta = float("nan")
        win_results[w] = {"alpha_daily": a_d, "t_alpha": ta}

    # Consistency: # windows significant at a t-threshold scaled by √(w/w_max)
    # and same sign as the majority. A constant true α with t=1.5 at the
    # longest window has expected t ≈ 1.5·√(w/w_max) at shorter windows, so a
    # flat 1.5 cutoff effectively demanded t≈3 at 252d and rejected nearly all
    # genuine signals.
    w_max = max(win_results) if win_results else max(windows)
    sig_signs = [
        np.sign(r["alpha_daily"])
        for w, r in win_results.items()
        if r["t_alpha"] == r["t_alpha"]
        and abs(r["t_alpha"]) > 1.5 * math.sqrt(w / w_max)
    ]
    if sig_signs:
        majority_sign = np.sign(sum(sig_signs))
        alpha_consistency = int(sum(1 for s in sig_signs if s == majority_sign))
    else:
        majority_sign     = 0
        alpha_consistency = 0

    # ── 2. PSR + Deflated IR (longest available window) ──────────────────────
    n_use   = min(max(windows), len(aligned))
    sl_long = aligned.iloc[-n_use:]
    exc_l   = (sl_long["ret"] - sl_long["RF"]).values
    X_c_l   = np.column_stack([np.ones(n_use), sl_long[FACTORS].values])
    c_l, _, rank_l, _ = np.linalg.lstsq(X_c_l, exc_l, rcond=None)
    resid_l  = exc_l - X_c_l @ c_l
    alpha_d  = float(c_l[0])
    sig_r    = float(np.std(resid_l, ddof=1)) if n_use > 1 else float("nan")
    ir_daily = alpha_d / sig_r if sig_r and sig_r > 1e-12 else float("nan")

    # Skewness and kurtosis of residuals (for PSR non-normality correction)
    if sig_r and sig_r > 0 and n_use > 3:
        z_resid = (resid_l - resid_l.mean()) / sig_r
        skew    = float(np.mean(z_resid ** 3))
        kurt    = float(np.mean(z_resid ** 4))   # NOT excess kurtosis; normal = 3
    else:
        skew, kurt = 0.0, 3.0

    psr = _psr(ir_daily, n_use, skew, kurt)

    # Deflated IR: subtract expected maximum IR from multiple testing
    exp_max       = _expected_max_ir_daily(n_strategies, n_use)
    ir_def_daily  = (ir_daily - exp_max) if ir_daily == ir_daily else float("nan")
    ir_def_ann    = (ir_def_daily * math.sqrt(252)
                     if ir_def_daily == ir_def_daily else float("nan"))
    ir_raw_ann    = (ir_daily * math.sqrt(252)
                     if ir_daily == ir_daily else float("nan"))

    # ── 3. Rolling OOS hit rate ───────────────────────────────────────────────
    oos_hit_rate: Optional[float] = None
    n_total = len(aligned)
    if n_total >= oos_train + 2 * oos_test:
        hits = total = 0
        for i in range(oos_train, n_total - oos_test + 1, oos_test):
            sl_tr  = aligned.iloc[i - oos_train:i]
            sl_te  = aligned.iloc[i:i + oos_test]
            exc_tr = (sl_tr["ret"] - sl_tr["RF"]).values
            X_tr   = np.column_stack([np.ones(oos_train), sl_tr[FACTORS].values])
            try:
                c_tr, _, _, _ = np.linalg.lstsq(X_tr, exc_tr, rcond=None)
                alpha_hat = float(c_tr[0])
                test_ret  = float((sl_te["ret"] - sl_te["RF"]).mean())
                if (alpha_hat > 0) == (test_ret > 0):
                    hits += 1
                total += 1
            except Exception:
                pass
        oos_hit_rate = hits / total if total >= 2 else None

    # ── Robust signal decision ────────────────────────────────────────────────
    psr_is_nan   = psr != psr
    psr_buy      = (not psr_is_nan) and psr > 0.85
    psr_sell     = (not psr_is_nan) and psr < 0.15
    cons_ok      = alpha_consistency >= max(1, len(windows) - 1)
    oos_buy_ok   = oos_hit_rate is None or oos_hit_rate > 0.52
    oos_sell_ok  = oos_hit_rate is None or oos_hit_rate < 0.48
    ir_pos       = ir_def_ann == ir_def_ann and ir_def_ann > 0
    ir_neg       = ir_def_ann == ir_def_ann and ir_def_ann < 0

    if   psr_buy  and cons_ok and majority_sign > 0 and oos_buy_ok  and ir_pos:
        robust_signal = "BUY"
    elif psr_sell and cons_ok and majority_sign < 0 and oos_sell_ok and ir_neg:
        robust_signal = "SELL"
    else:
        robust_signal = "HOLD"

    return {
        "ticker":            ticker,
        "psr":               psr,
        "ir_raw":            ir_raw_ann,
        "ir_deflated":       ir_def_ann,
        "alpha_consistency": alpha_consistency,
        "n_windows":         len(win_results),
        "oos_hit_rate":      oos_hit_rate,
        "robust_signal":     robust_signal,
        "window_alphas":     {w: r["alpha_daily"] * 252 for w, r in win_results.items()},
        "window_t_alphas":   {w: r["t_alpha"]         for w, r in win_results.items()},
    }


def enrich_suggestions(
    suggestions: list[dict],
    stock_factors: list[dict],
) -> list[dict]:
    """
    Adds 'reason' and 'reason_zh' fields to each rebalancing suggestion
    by cross-referencing the FF5 regression results.  Modifies in-place
    and also returns the list.
    """
    factor_map = {s["ticker"]: s for s in stock_factors}

    for sug in suggestions:
        reg   = factor_map.get(sug["ticker"])
        delta = sug.get("delta", 0.0)

        if reg is None:
            sug["reason"] = sug["reason_zh"] = ""
            continue

        alpha  = reg.get("alpha_ann", 0.0)
        t      = reg.get("t_alpha", float("nan"))
        signal = reg.get("signal", "HOLD")
        t_str  = f" t={t:+.1f}" if t == t else ""   # omit if NaN

        a_str    = f"α={alpha*100:+.1f}%{t_str}"
        cgt_flag = sug.get("cgt_flag", False)

        if delta > 0:
            if signal == "BUY":
                en = f"{a_str} → BUY signal; underweight"
                zh = f"{a_str} → 买入信号，当前低配"
            else:
                en = f"{a_str}; optimizer targets higher allocation"
                zh = f"{a_str}；优化器建议提高配置"
        else:
            if signal == "SELL":
                en = f"{a_str} → SELL signal; overweight"
                zh = f"{a_str} → 卖出信号，当前超配"
            elif cgt_flag:
                en = f"{a_str}; trim deferred — CGT 33% applies"
                zh = f"{a_str}；减仓受CGT 33%税务约束"
            else:
                en = f"{a_str}; trim to reduce concentration"
                zh = f"{a_str}；减仓以降低集中度"

        sug["reason"]    = en
        sug["reason_zh"] = zh

    return suggestions
