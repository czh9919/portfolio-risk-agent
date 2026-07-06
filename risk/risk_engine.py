"""
M3 — Risk engine: EWMA + GARCH(1,1) + fixed window.
All amounts normalised to EUR. Returns risk metrics as a dict.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

Z95 = 1.6449
Z99 = 2.3263


# ── Cornish-Fisher modified VaR ───────────────────────────────────────────────

def cornish_fisher_z(z: float, skew: float, exkurt: float) -> float:
    """
    Cornish-Fisher expansion: adjust the normal quantile for skewness (S) and
    excess kurtosis (K).  z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36
    """
    return (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * exkurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)


def cf_var(portfolio_returns: pd.Series, sigma_t: float,
           confidence: float = 0.95) -> float:
    """
    Cornish-Fisher modified VaR: EWMA volatility adjusted for the empirical
    skewness and excess kurtosis of the return distribution.
    Equity returns are typically negatively skewed and fat-tailed; the plain
    normal VaR underestimates by ~15–30% in stress regimes.  CF corrects this.

    We use the LOWER-tail quantile z = norm.ppf(1-confidence) < 0 so that
    negative skew pushes z_cf more negative → larger VaR (correct direction).
    pandas .kurt() returns *excess* kurtosis (Fisher definition).

    Drift is intentionally set to zero: at a 1-day horizon the sample mean is
    estimated with large error and is tiny vs σ; including it biases VaR
    (a recent up-move would understate downside risk).
    """
    if len(portfolio_returns) < 21 or np.isnan(sigma_t):
        return float("nan")
    z    = norm.ppf(1 - confidence)          # negative, e.g. -1.645 for 95%
    skew = float(portfolio_returns.skew())
    exk  = float(portfolio_returns.kurt())   # excess kurtosis
    z_cf = cornish_fisher_z(z, skew, exk)   # more negative for left-skewed returns
    return float(-z_cf * sigma_t)           # -z_cf > 0 gives the loss magnitude


# ── EVT / Peaks-over-Threshold VaR ───────────────────────────────────────────

def evt_var_gpd(portfolio_returns: pd.Series, confidence: float = 0.99,
                threshold_pct: float = 0.10) -> float:
    """
    Extreme Value Theory VaR via Peaks-over-Threshold and Generalized Pareto
    Distribution.  Reliable for extreme quantiles (99 %+) where the normal and
    CF approximations deteriorate.

    threshold_pct = fraction of worst returns used as threshold (default 10 %).
    Falls back to historical VaR when fewer than 20 threshold exceedances exist.
    """
    from scipy.stats import genpareto
    if len(portfolio_returns) < 60:
        return float("nan")
    losses = -portfolio_returns.values          # positive = loss
    u      = np.percentile(losses, (1 - threshold_pct) * 100)
    excess = losses[losses > u] - u
    if len(excess) < 20:
        logger.debug(f"EVT: only {len(excess)} exceedances — fallback to historical VaR")
        return fixed_var_cvar(portfolio_returns, confidence)[0]
    try:
        shape, _, scale = genpareto.fit(excess, floc=0)
        p = (1 - confidence) / threshold_pct   # conditional exceedance prob
        if abs(shape) < 1e-8:                   # exponential special case
            x = u - scale * np.log(p)
        else:
            x = u + scale / shape * (p ** (-shape) - 1)
        return float(x)
    except Exception as exc:
        logger.debug(f"EVT GPD fit failed ({exc}) — fallback to historical VaR")
        return fixed_var_cvar(portfolio_returns, confidence)[0]


# ── EWMA VaR ─────────────────────────────────────────────────────────────────

def ewma_variance_series(returns: pd.Series, lam: float = 0.94,
                         init_window: int = 21) -> pd.Series:
    """Compute EWMA variance for each day. Returns series aligned with returns."""
    r = returns.values
    var = np.empty(len(r))
    if len(r) < init_window:
        var[:] = np.var(r)
    else:
        var[0] = np.var(r[:init_window])
        for t in range(1, len(r)):
            var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(var, index=returns.index)


def ewma_covariance(ret_df: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """
    EWMA covariance matrix of the columns of ret_df (most recent row weighted
    highest). Returns an (n_assets, n_assets) numpy array aligned to ret_df.columns.
    """
    R = ret_df.values
    T = R.shape[0]
    if T == 0:
        n = R.shape[1]
        return np.zeros((n, n))
    decay = lam ** np.arange(T - 1, -1, -1)   # oldest → lowest weight
    decay = decay / decay.sum()
    mu    = (R * decay[:, None]).sum(axis=0)
    R_dm  = R - mu
    return (R_dm * decay[:, None]).T @ R_dm


def ewma_var(portfolio_returns: pd.Series, lam: float = 0.94,
             init_window: int = 21, confidence: float = 0.95) -> float:
    """
    One-day VaR as a fraction of portfolio value (positive = loss).

    Drift set to zero (μ=0): see cf_var — the 1-day mean is noise-dominated and
    biases short-horizon VaR.
    """
    if len(portfolio_returns) < init_window:
        return float("nan")
    var_series = ewma_variance_series(portfolio_returns, lam, init_window)
    sigma_t    = float(np.sqrt(var_series.iloc[-1]))
    z          = Z95 if confidence == 0.95 else Z99
    return z * sigma_t


# ── Fixed-window CVaR ─────────────────────────────────────────────────────────

def fixed_var_cvar(portfolio_returns: pd.Series,
                   confidence: float = 0.95) -> tuple[float, float]:
    """Historical simulation VaR and CVaR (positive = loss)."""
    if portfolio_returns.empty:
        return float("nan"), float("nan")
    cutoff = np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar   = float(portfolio_returns[portfolio_returns <= cutoff].mean())
    return float(-cutoff), float(-cvar)


# ── GARCH(1,1) per asset ──────────────────────────────────────────────────────

def garch_variance(returns: pd.Series, omega: float = 1e-6,
                   alpha: float = 0.05, beta: float = 0.94) -> float:
    """
    Fit GARCH(1,1) via arch library.
    Falls back to EWMA on convergence failure.
    """
    try:
        from arch import arch_model
        am  = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Zero", dist="Normal")
        res = am.fit(disp="off", show_warning=False)
        fc  = res.forecast(horizon=1)
        return float(fc.variance.values[-1, 0]) / (100 ** 2)
    except Exception as e:
        logger.debug(f"GARCH failed ({e}), falling back to EWMA")
        var_s = ewma_variance_series(returns)
        return float(var_s.iloc[-1])


# ── Portfolio returns ─────────────────────────────────────────────────────────

def portfolio_returns(price_data: dict, holdings: list[dict],
                      window: int = 252) -> pd.Series:
    """
    Compute weighted portfolio daily returns.
    Only includes holdings with status in {ok, reduced}.

    Weights are summed per ticker so the same symbol held across multiple
    brokers is aggregated rather than silently overwritten.
    """
    weights: dict[str, float] = {}
    for h in holdings:
        weights[h["ticker"]] = weights.get(h["ticker"], 0.0) + h["weight"]
    frames  = []
    for ticker, pd_obj in price_data.items():
        if pd_obj.status not in ("ok", "reduced"):
            continue
        w = weights.get(ticker, 0.0)
        if w == 0 or pd_obj.returns is None:
            continue
        frames.append(pd_obj.returns.tail(window) * w)

    if not frames:
        return pd.Series(dtype=float)

    df = pd.concat(frames, axis=1).fillna(0)
    return df.sum(axis=1)


# ── Correlation matrix ────────────────────────────────────────────────────────

def rolling_correlation(price_data: dict, holdings: list[dict],
                        window: int = 63) -> Optional[pd.DataFrame]:
    valid_tickers = [
        h["ticker"] for h in holdings
        if price_data.get(h["ticker"]) and
           price_data[h["ticker"]].status in ("ok", "reduced") and
           price_data[h["ticker"]].returns is not None
    ]
    if len(valid_tickers) < 2:
        return None
    ret_df = pd.concat(
        {t: price_data[t].returns.tail(window) for t in valid_tickers}, axis=1
    ).dropna()
    return ret_df.corr()


# ── Portfolio σ (GARCH) ───────────────────────────────────────────────────────

def portfolio_sigma(price_data: dict, holdings: list[dict],
                    vcfg: dict = None) -> float:
    """Portfolio annualised volatility using per-asset GARCH variances + rolling correlation."""
    vcfg    = vcfg or {}
    omega   = vcfg.get("omega", 1e-6)
    alpha   = vcfg.get("alpha", 0.05)
    beta    = vcfg.get("beta",  0.94)
    corr_df = rolling_correlation(price_data, holdings)

    valid = [
        h for h in holdings
        if price_data.get(h["ticker"]) and
           price_data[h["ticker"]].status in ("ok", "reduced") and
           price_data[h["ticker"]].returns is not None
    ]
    if not valid:
        return float("nan")

    weights = np.array([h["weight"] for h in valid])
    tickers = [h["ticker"] for h in valid]
    var_vec = np.array([garch_variance(price_data[t].returns, omega, alpha, beta) for t in tickers])
    sigma_vec = np.sqrt(var_vec)

    if corr_df is not None:
        corr = np.array(corr_df.reindex(index=tickers, columns=tickers).fillna(0).values, dtype=float)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.eye(len(tickers))

    cov = np.outer(sigma_vec, sigma_vec) * corr
    port_var = float(weights @ cov @ weights)
    return float(np.sqrt(port_var * 252))  # annualised


# ── Sharpe ratio ──────────────────────────────────────────────────────────────

def sharpe_ratio(portfolio_ret: pd.Series, rf: float = 0.035) -> float:
    if len(portfolio_ret) < 21:
        return float("nan")
    ann_ret = float(portfolio_ret.mean() * 252)
    ann_vol = float(portfolio_ret.std() * np.sqrt(252))
    return (ann_ret - rf) / ann_vol if ann_vol else float("nan")


# ── Beta vs SPY ───────────────────────────────────────────────────────────────

def beta_vs_spy(portfolio_ret: pd.Series, spy_returns: pd.Series) -> float:
    aligned = pd.concat([portfolio_ret, spy_returns], axis=1).dropna()
    if len(aligned) < 21:
        return float("nan")
    p, s   = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
    cov    = np.cov(p, s)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")


# ── HHI ──────────────────────────────────────────────────────────────────────

def hhi(holdings: list[dict]) -> float:
    return float(sum(h["weight"] ** 2 for h in holdings))


# ── Max Drawdown ──────────────────────────────────────────────────────────────

def max_drawdown(portfolio_ret: pd.Series) -> float:
    if portfolio_ret.empty:
        return float("nan")
    cum   = (1 + portfolio_ret).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    return float(dd.min())


# ── Component / Marginal VaR ──────────────────────────────────────────────────

def _aligned_returns_weights(price_data: dict, holdings: list[dict],
                             window: int = 252):
    """
    Build an aligned returns DataFrame and matching weight vector for the
    holdings whose price data is usable. Weights are summed per ticker so a
    symbol held across brokers is aggregated. Returns (ret_df, w, tickers) or
    (None, None, None) if fewer than 2 usable assets.
    """
    weights: dict[str, float] = {}
    for h in holdings:
        weights[h["ticker"]] = weights.get(h["ticker"], 0.0) + h["weight"]

    series = {}
    for ticker, w in weights.items():
        pd_obj = price_data.get(ticker)
        if (w == 0 or pd_obj is None or
                pd_obj.status not in ("ok", "reduced") or pd_obj.returns is None):
            continue
        series[ticker] = pd_obj.returns.tail(window)

    if len(series) < 2:
        return None, None, None

    ret_df  = pd.concat(series, axis=1).dropna()
    if ret_df.empty:
        return None, None, None
    tickers = list(ret_df.columns)
    w       = np.array([weights[t] for t in tickers], dtype=float)
    return ret_df, w, tickers


def component_var(price_data: dict, holdings: list[dict], lam: float = 0.94,
                  window: int = 252, confidence: float = 0.95) -> list[dict]:
    """
    Decompose portfolio VaR into per-asset component contributions using the
    EWMA covariance matrix.

      marginal VaR_i = z · (Σw)_i / σ_p          (sensitivity of VaR to weight_i)
      component VaR_i = w_i · marginal VaR_i      (Σ component VaR = portfolio VaR)

    σ_p = √(wᵀΣw). Returns a list of dicts sorted by component_var descending,
    each: {ticker, weight, marginal_var, component_var, pct_of_var}.
    Empty list if fewer than 2 usable assets.
    """
    ret_df, w, tickers = _aligned_returns_weights(price_data, holdings, window)
    if ret_df is None:
        return []

    cov   = ewma_covariance(ret_df, lam)
    sig_p = float(np.sqrt(w @ cov @ w))
    if sig_p <= 0 or np.isnan(sig_p):
        return []

    z          = Z95 if confidence == 0.95 else Z99
    port_var   = z * sig_p
    cov_w      = cov @ w
    marginal   = z * cov_w / sig_p                 # ∂VaR/∂w_i
    component  = w * marginal                       # sums to port_var

    out = []
    for i, t in enumerate(tickers):
        out.append({
            "ticker":        t,
            "weight":        float(w[i]),
            "marginal_var":  float(marginal[i]),
            "component_var": float(component[i]),
            "pct_of_var":    float(component[i] / port_var) if port_var else float("nan"),
        })
    out.sort(key=lambda d: d["component_var"], reverse=True)
    return out


# ── VaR backtesting (Kupiec POF + Christoffersen independence) ─────────────────

def var_backtest(portfolio_returns: pd.Series, lam: float = 0.94,
                 init_window: int = 21, confidence: float = 0.95,
                 method: str = "cf") -> dict:
    """
    Out-of-sample VaR backtest using a causal (one-step-ahead) EWMA forecast.

    For each day t we forecast σ_t from data up to t-1, build the VaR estimate,
    and check whether the realised return breached it. We then run:
      - Kupiec POF test    : LR_uc ~ χ²(1), H0 = exception rate equals 1-c
      - Christoffersen ind.: LR_ind ~ χ²(1), H0 = exceptions are independent
      - joint              : LR_cc = LR_uc + LR_ind ~ χ²(2)

    method "cf" uses the Cornish-Fisher quantile (skew+kurtosis of the trailing
    window); "normal" uses the plain Gaussian quantile.
    Returns a dict of counts, rates, LR stats and p-values (all JSON-safe).
    """
    from scipy.stats import chi2

    n = len(portfolio_returns)
    if n < init_window + 60:
        return {"status": "insufficient_data", "n_obs": int(n)}

    var_series = ewma_variance_series(portfolio_returns, lam, init_window)
    sigma      = np.sqrt(var_series.values)
    r          = portfolio_returns.values
    p          = 1 - confidence
    z          = norm.ppf(p)                       # negative

    start = init_window
    exceptions = []                                # 1 if breach, else 0
    for t in range(start, n):
        s_t = sigma[t - 1]                         # causal: forecast from t-1
        if np.isnan(s_t) or s_t <= 0:
            continue
        if method == "cf":
            win   = portfolio_returns.iloc[max(0, t - 252):t]
            skew  = float(win.skew()) if len(win) > 3 else 0.0
            exk   = float(win.kurt()) if len(win) > 3 else 0.0
            z_use = cornish_fisher_z(z, skew, exk)
        else:
            z_use = z
        var_t = z_use * s_t                        # negative threshold
        exceptions.append(1 if r[t] < var_t else 0)

    exc = np.array(exceptions, dtype=int)
    N   = len(exc)
    if N == 0:
        return {"status": "insufficient_data", "n_obs": int(n)}
    x   = int(exc.sum())
    rate = x / N

    # Kupiec POF: LR_uc = -2 ln[ (1-p)^(N-x) p^x / (1-rate)^(N-x) rate^x ]
    def _safe_ln(a):
        return np.log(a) if a > 0 else 0.0

    ll_null = (N - x) * _safe_ln(1 - p) + x * _safe_ln(p)
    ll_alt  = (N - x) * _safe_ln(1 - rate) + x * _safe_ln(rate)
    lr_uc   = -2.0 * (ll_null - ll_alt)
    p_uc    = float(chi2.sf(lr_uc, 1))

    # Christoffersen independence via transition counts
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(exc[:-1], exc[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) else 0.0
    ll_ind  = (n00 + n10) * _safe_ln(1 - pi) + (n01 + n11) * _safe_ln(pi)
    ll_dep  = (n00 * _safe_ln(1 - pi01) + n01 * _safe_ln(pi01) +
               n10 * _safe_ln(1 - pi11) + n11 * _safe_ln(pi11))
    lr_ind  = -2.0 * (ll_ind - ll_dep)
    p_ind   = float(chi2.sf(lr_ind, 1))

    lr_cc = lr_uc + lr_ind
    p_cc  = float(chi2.sf(lr_cc, 2))

    return {
        "status":          "ok",
        "method":          method,
        "confidence":      float(confidence),
        "n_obs":           int(N),
        "exceptions":      int(x),
        "expected":        float(N * p),
        "exception_rate":  float(rate),
        "expected_rate":   float(p),
        "kupiec_lr":       float(lr_uc),
        "kupiec_p":        p_uc,
        "christ_lr":       float(lr_ind),
        "christ_p":        p_ind,
        "cc_lr":           float(lr_cc),
        "cc_p":            p_cc,
        "pass":            bool(p_uc > 0.05 and p_ind > 0.05),
    }


# ── Full metrics bundle ───────────────────────────────────────────────────────

def compute_all(holdings: list[dict], price_data: dict,
                spy_price_data, vcfg: dict = None,
                thresholds: dict = None) -> dict:
    """
    Run all 10 core metrics. Returns dict of metric_name → value (floats).
    spy_price_data: PriceData for SPY.
    """
    vcfg = vcfg or {}
    lam        = vcfg.get("ewma", {}).get("lambda", 0.94)
    init_win   = vcfg.get("ewma", {}).get("init_window", 21)
    fixed_win  = vcfg.get("windows", {}).get("fixed", 252)
    rf         = vcfg.get("risk_free_rate", 0.035)

    port_ret   = portfolio_returns(price_data, holdings, window=fixed_win)
    nav        = sum(h["market_value_eur"] for h in holdings)
    total_pnl  = sum(h["unrealised_pnl_eur"] for h in holdings)
    daily_ret  = port_ret.iloc[-1] if not port_ret.empty else float("nan")

    var_95_ewma = ewma_var(port_ret, lam=lam, init_window=init_win, confidence=0.95)
    var_99_ewma = ewma_var(port_ret, lam=lam, init_window=init_win, confidence=0.99)
    var_95_hist, cvar_95 = fixed_var_cvar(port_ret, 0.95)

    # Cornish-Fisher (adjusts for skewness + fat tails) and EVT corrections
    var_series = ewma_variance_series(port_ret, lam=lam, init_window=init_win)
    sigma_t    = float(np.sqrt(var_series.iloc[-1])) if len(var_series) > 0 else float("nan")
    var_95_cf  = cf_var(port_ret, sigma_t, confidence=0.95)
    var_99_cf  = cf_var(port_ret, sigma_t, confidence=0.99)
    var_99_evt = evt_var_gpd(port_ret, confidence=0.99)

    spy_ret = spy_price_data.returns if spy_price_data and spy_price_data.returns is not None else pd.Series(dtype=float)

    metrics = {
        "nav_eur":           nav,
        "total_pnl_eur":     total_pnl,
        "daily_return":      daily_ret,
        "var_95_ewma":       var_95_ewma,   # plain normal — kept for comparison
        "var_99_ewma":       var_99_ewma,
        "var_95_cf":         var_95_cf,    # Cornish-Fisher (primary alert trigger)
        "var_99_cf":         var_99_cf,
        "var_99_evt":        var_99_evt,   # EVT / GPD (extreme quantile)
        "var_95_hist":       var_95_hist,
        "cvar_95":           cvar_95,
        "cvar_var_ratio":    cvar_95 / var_95_hist if var_95_hist else float("nan"),
        "max_drawdown":      max_drawdown(port_ret),
        "sharpe":            sharpe_ratio(port_ret, rf=rf),
        "beta":              beta_vs_spy(port_ret, spy_ret),
        "hhi":               hhi(holdings),
        "max_position_wt":   max((h["weight"] for h in holdings), default=0.0),
        "port_sigma_annual": portfolio_sigma(price_data, holdings, vcfg.get("garch", {})),
        "component_var":     component_var(price_data, holdings, lam=lam,
                                           window=fixed_win, confidence=0.95),
        "var_backtest":      {
            "cf":     var_backtest(port_ret, lam=lam, init_window=init_win,
                                   confidence=0.95, method="cf"),
            "normal": var_backtest(port_ret, lam=lam, init_window=init_win,
                                   confidence=0.95, method="normal"),
        },
    }

    # RAG status: RED = breach, AMBER = inside the warning band (within
    # amber_ratio of the threshold), GREEN = comfortably clear, GREY = no data.
    amber_ratio = float((thresholds or {}).get("amber_ratio", 0.8))

    def rag(value, threshold, direction="above"):
        if value is None or value != value:  # nan
            return "GREY"
        if direction == "above":
            if value > threshold:
                return "RED"
            if value > threshold * amber_ratio:
                return "AMBER"
        else:
            if value < threshold:
                return "RED"
            if value < threshold / amber_ratio:
                return "AMBER"
        return "GREEN"

    alerts = {}
    if thresholds:
        t = thresholds.get("alerts", {})
        alerts["var_95"]      = rag(metrics["var_95_cf"],      t.get("var_95_pct",         {}).get("threshold", 0.05))
        alerts["cvar_ratio"]  = rag(metrics["cvar_var_ratio"], t.get("cvar_var_ratio",      {}).get("threshold", 1.8))
        alerts["max_dd"]      = rag(abs(metrics["max_drawdown"]), t.get("max_drawdown",     {}).get("threshold", 0.20))
        alerts["max_pos"]     = rag(metrics["max_position_wt"], t.get("max_single_position",{}).get("threshold", 0.30))
        alerts["hhi"]         = rag(metrics["hhi"],             t.get("hhi",                {}).get("threshold", 0.25))
        alerts["beta"]        = rag(metrics["beta"],            t.get("beta",               {}).get("threshold", 1.5))
        alerts["sharpe"]      = rag(metrics["sharpe"],          t.get("sharpe",             {}).get("threshold", 0.5), direction="below")
        alerts["daily_loss"]  = rag(-metrics["daily_return"],  t.get("single_day_loss",     {}).get("threshold", 0.03))

    overall_rag = "RED" if "RED" in alerts.values() else ("AMBER" if "AMBER" in alerts.values() else "GREEN")
    metrics["alerts"]      = alerts
    metrics["overall_rag"] = overall_rag

    return metrics
