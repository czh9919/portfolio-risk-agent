"""
Unified entry point.
RUN_MODE=portfolio    → portfolio risk pipeline only (no watchlist factor ranking)
RUN_MODE=full         → portfolio pipeline + watchlist FF5 factor ranking (default)
RUN_MODE=alert_check  → portfolio threshold check only (no report)
RUN_MODE=backtest     → walk-forward backtest
RUN_MODE=paper_trade  → paper trading via Alpaca
RUN_MODE=market_brief → pre-market macro/news brief (Treasuries, DXY, VIX, curated news)
RUN_MODE=heatmap      → post-close ETF heatmap (SPY/QQQ + 11 sector SPDRs, treemap + cards)
"""
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml


def setup_logging(level: str = "INFO", log_file: str = "logs/agent.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format=fmt,
        handlers=[
            logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), raw)
    return yaml.safe_load(raw)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_watchlist(path: str = "config/watchlist.csv") -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Portfolio + Factor pipeline ────────────────────────────────────────────────

def run_portfolio_pipeline(run_mode: str = "full", config: dict = None, on_log=None):
    logger = logging.getLogger("portfolio")

    if on_log:
        class _WSHandler(logging.Handler):
            def emit(self, record):
                on_log(self.format(record))
        _h = _WSHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(_h)

    logger.info(f"=== Portfolio Risk System — run_mode={run_mode} ===")

    thresholds = load_yaml("config/thresholds.yaml")
    vol_cfg    = load_yaml("config/volatility.yaml")
    fixed_days = vol_cfg.get("windows", {}).get("fixed", 252)

    from data.price_loader   import load_fx_rates, load_prices
    from data.fetch_holdings import fetch_all_holdings
    from data                import cache

    fx_rates = load_fx_rates()
    holdings = fetch_all_holdings(fx_rates)

    if not holdings:
        cached = cache.load_latest_snapshot()
        if cached:
            snap_data, snap_date = cached
            logger.warning(f"No live holdings — using cache from {snap_date}")
            holdings = snap_data.get("holdings", [])
        else:
            logger.error("No holdings and no cache — skipping portfolio pipeline")
            return

    # Load ALL holding tickers (incl. bonds/gold) so 1-day risk metrics, the
    # optimizer and stress price the whole portfolio — not just the equity sleeve.
    all_tickers    = list({h["ticker"] for h in holdings})
    price_data     = load_prices(all_tickers + ["SPY"], days=fixed_days)
    spy_pd         = price_data.pop("SPY", None)

    from risk.risk_engine import compute_all
    metrics = compute_all(holdings, price_data, spy_pd, vcfg=vol_cfg, thresholds=thresholds)
    logger.info(f"NAV: €{metrics['nav_eur']:,.0f} | VaR(95%): {metrics['var_95_ewma']*100:.1f}% | RAG: {metrics['overall_rag']}")

    from alert import check_and_alert
    check_and_alert(metrics, thresholds)

    if run_mode == "alert_check":
        logger.info("alert_check — skipping full report")
        return {"metrics": metrics, "html_en": "", "html_zh": ""}

    # ── Fama-French 5-factor analysis ─────────────────────────────────────────
    from risk.optimizer        import fetch_ff5
    from strategy.factor_model import (run_factor_regression, portfolio_factor_exposure,
                                       compute_attribution, enrich_suggestions,
                                       compute_stock_attribution, explain_stock_drivers,
                                       compute_robust_signal)

    ff5           = fetch_ff5()
    stock_factors: list[dict] = []
    port_exposure             = None
    attribution               = None
    collinearity              = None
    ic_result                 = None
    signal_weights: list[dict] = []

    if ff5 is not None:
        # Portfolio factor exposure (weighted average betas of current holdings)
        port_exposure = portfolio_factor_exposure(holdings, price_data, ff5)
        # Performance attribution over last 21 trading days
        if port_exposure:
            attribution = compute_attribution(holdings, price_data, ff5, port_exposure)

        if run_mode == "full":
            from strategy.ic_analysis      import compute_universe_ic
            from strategy.factor_ortho     import check_factor_collinearity
            from strategy.weight_allocator import compute_signal_weights

            # Watchlist factor ranking — load prices for tickers not in holdings
            watchlist_equity = [
                w for w in load_watchlist()
                if w.get("asset_class", "equity") == "equity"
            ]
            known_pd = {**price_data, "SPY": spy_pd} if spy_pd else dict(price_data)
            extra_tk = [w["ticker"] for w in watchlist_equity if w["ticker"] not in known_pd]
            factor_pd = {**known_pd, **(load_prices(extra_tk, days=fixed_days) if extra_tk else {})}

            for entry in watchlist_equity:
                tk     = entry["ticker"]
                pd_obj = factor_pd.get(tk)
                if pd_obj and pd_obj.returns is not None:
                    reg = run_factor_regression(tk, pd_obj.returns, ff5)
                    if reg:
                        reg["price"] = (float(pd_obj.closes.iloc[-1])
                                        if pd_obj.closes is not None and not pd_obj.closes.empty
                                        else None)
                        # Robust signal: multi-window + PSR + deflated IR + OOS
                        robust = compute_robust_signal(
                            tk, pd_obj.returns, ff5,
                            n_strategies=len(watchlist_equity),
                        )
                        if robust:
                            reg.update({k: robust[k] for k in (
                                "psr", "ir_deflated", "ir_raw",
                                "alpha_consistency", "n_windows",
                                "oos_hit_rate", "robust_signal",
                                "window_alphas", "window_t_alphas",
                            )})

                        # 63-day factor attribution + driver explanation
                        atr_63 = compute_stock_attribution(tk, pd_obj.returns, ff5, reg, window=63)
                        if atr_63:
                            reg["attribution_63d"] = atr_63
                            reg["driver_en"] = explain_stock_drivers(reg, atr_63, lang="en")
                            reg["driver_zh"] = explain_stock_drivers(reg, atr_63, lang="zh")
                        stock_factors.append(reg)

            stock_factors.sort(
                key=lambda x: x.get("ir") if x.get("ir") == x.get("ir") else float("-inf"),
                reverse=True,
            )

            # Factor collinearity diagnostics
            collinearity = check_factor_collinearity(ff5)
            if collinearity.get("has_warning"):
                logger.warning(
                    f"Factor collinearity detected: "
                    f"high pairs={collinearity.get('high_pairs')}, "
                    f"cond#={collinearity.get('condition_number', 0):.1f}"
                )

            # Cross-sectional IC analysis (temporal validation)
            ic_result = compute_universe_ic(factor_pd, ff5)
            if ic_result:
                logger.info(
                    f"IC analysis: IC_mean={ic_result['ic_mean']:+.3f}  "
                    f"ICIR={ic_result['icir']:.2f}  "
                    f"p={ic_result['pvalue']:.3f}  "
                    f"n_periods={ic_result['n_obs']}"
                )

            # Signal-weighted allocation
            signal_weights = compute_signal_weights(stock_factors)
            if signal_weights:
                logger.info(
                    f"Signal portfolio: {len(signal_weights)} positions, "
                    f"top={signal_weights[0]['ticker']} {signal_weights[0]['weight']*100:.1f}%"
                )

            logger.info(f"Factor ranking: {len(stock_factors)} stocks | "
                        f"top IR: {stock_factors[0]['ticker'] if stock_factors else '—'}")
    else:
        logger.warning("FF5 data unavailable — factor analysis skipped")

    # ── Stress, optimizer, Monte Carlo ────────────────────────────────────────
    from risk.stress        import run_all as run_stress
    from risk.optimizer     import compute_frontier, rebalancing_suggestions, marginal_impact
    from risk.mc_portfolio  import run_mc_portfolio
    from notify.report_gen  import build_report
    from notify.chart       import risk_return_png
    from notify             import mailer

    # Stress tests need multi-year history to reprice through past crises; the
    # 252-day risk window never reaches 2008/2020/2022. Load a long window once.
    stress_days = vol_cfg.get("windows", {}).get("stress", 2200)
    stress_pd   = load_prices(all_tickers, days=stress_days)
    stress      = run_stress(holdings, stress_pd, metrics["nav_eur"], vol_cfg=vol_cfg,
                             port_beta=metrics.get("beta", 1.0))
    snapshot    = {"holdings": holdings, "metrics": {k: v for k, v in metrics.items() if k != "alerts"}}
    cache.save_snapshot(snapshot)
    last_week   = cache.load_week_ago_snapshot()

    rf          = thresholds.get("risk_free_rate", 0.035)
    frontier    = compute_frontier(price_data, holdings, rf=rf)
    suggestions = rebalancing_suggestions(holdings, frontier, nav_eur=metrics["nav_eur"])
    if stock_factors and suggestions:
        enrich_suggestions(suggestions, stock_factors)
    chart_en    = risk_return_png(frontier, "en")
    chart_zh    = risk_return_png(frontier, "zh")

    mc_cfg    = (config or {}).get("monte_carlo", {})
    mc_result = run_mc_portfolio(
        holdings, price_data, rf=rf,
        n_paths=int(mc_cfg.get("n_paths", 5000)),
        horizon=int(mc_cfg.get("horizon", 21)),
        config={"model": mc_cfg.get("model", "hawkes")},
    )
    if mc_result:
        tail = mc_result["tail"]
        dec  = mc_result["decision"]
        stress["monte_carlo"] = {
            "paths":            tail["n_paths"],
            "horizon_days":     tail["horizon_days"],
            "model":            "Hawkes-GARCH",
            "var_95":           tail["var_95_pct"],
            "eur_var_95":       tail["var_95_eur"],
            "var_99":           tail["var_99_pct"],
            "eur_var_99":       tail["var_99_eur"],
            "cvar_95":          tail["cvar_95_pct"],
            "cvar_95_eur":      tail["cvar_95_eur"],
            "max_dd_mean":      tail["max_dd_mean"],
            "max_dd_p95":       tail["max_dd_p95"],
            "prob_loss_10pct":  tail["prob_loss_10pct"],
            "prob_loss_20pct":  tail["prob_loss_20pct"],
            "margin_call_prob": tail["margin_call_prob"],
            "p_positive":       dec["win_rate"],
            "sharpe_median":    dec["sharpe_median"],
            "sortino_median":   dec["sortino_median"],
            "ann_ret_median":   dec["ann_ret_median"],
            "ann_vol_median":   dec["ann_vol_median"],
            "trading_cost_eur": mc_result["costs"]["trading_eur"],
            "cgt_median_eur":   mc_result["costs"]["cgt_median_eur"],
        }
        logger.info(
            f"MC: VaR95={tail['var_95_pct']*100:.1f}%  "
            f"CVaR95={tail['cvar_95_pct']*100:.1f}%  "
            f"P(+)={dec['win_rate']*100:.0f}%"
        )

    # ── Diversification candidates (bonds + gold from watchlist) ──────────────
    watchlist_all = load_watchlist()
    cand_entries  = [w for w in watchlist_all if w.get("asset_class") in ("bond", "gold")]
    div_candidates: list[dict] = []
    if cand_entries:
        new_tickers   = [w["ticker"] for w in cand_entries if w["ticker"] not in price_data]
        cand_pd_extra = load_prices(new_tickers, days=fixed_days) if new_tickers else {}
        for entry in cand_entries:
            tk     = entry["ticker"]
            pd_obj = cand_pd_extra.get(tk) or price_data.get(tk)
            impact = marginal_impact(tk, pd_obj, holdings, price_data, rf=rf)
            if impact:
                impact["currency"]    = entry.get("currency", "USD")
                impact["asset_class"] = entry.get("asset_class", "bond")
                impact["name"]        = entry.get("notes", tk)
                div_candidates.append(impact)
        div_candidates.sort(key=lambda x: x["delta_sharpe"], reverse=True)
        logger.info(f"Diversification candidates: {len(div_candidates)} computed")

    # ── Execution / exit-laddering plan (held + watchlist BUY candidates) ─────
    execution: list[dict] = []
    try:
        from data.price_loader import load_ohlcv
        from strategy.execution import analyze_execution

        exec_cfg   = (config or {}).get("execution", {})
        ohlcv_days = int(exec_cfg.get("ohlcv_days", 180))
        max_pos_pct = float(exec_cfg.get("max_position_pct", 0.15))
        target_eur  = max_pos_pct * float(metrics.get("nav_eur") or 0.0)
        wl_cur = {w["ticker"]: w.get("currency", "USD") for w in load_watchlist()}

        held = {h["ticker"]: h for h in holdings if h.get("asset_class", "equity") == "equity"}
        buy_watch = [
            s["ticker"] for s in (stock_factors or [])
            if (s.get("robust_signal") or s.get("signal")) == "BUY"
            and s["ticker"] not in held
        ]
        exec_tickers = list(held.keys()) + buy_watch
        ohlcv = load_ohlcv(exec_tickers, days=ohlcv_days) if exec_tickers else {}

        for tk, h in held.items():
            df = ohlcv.get(tk)
            if df is None:
                continue
            qty  = float(h.get("quantity") or 0)
            rate = fx_rates.get(f"{h.get('currency','USD')}EUR", 1.0)
            avg_cost_native = (float(h["cost_basis_eur"]) / qty / rate) if qty and rate else None
            entry_dt = None
            ed = h.get("entry_date")
            if ed:
                try:
                    entry_dt = date.fromisoformat(str(ed)[:10])
                except ValueError:
                    entry_dt = None
            plan = analyze_execution(tk, df, side="exit", position_shares=qty,
                                     avg_cost_native=avg_cost_native,
                                     entry_date=entry_dt, cfg=exec_cfg)
            if plan:
                execution.append(plan)

        for tk in buy_watch:
            df = ohlcv.get(tk)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            # Target a max_position_pct slice of NAV → native share count
            rate = fx_rates.get(f"{wl_cur.get(tk, 'USD')}EUR", 1.0)
            price = float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else None
            ref_shares = None
            if price and rate and target_eur > 0:
                ref_shares = (target_eur / rate) / price
            plan = analyze_execution(tk, df, side="entry",
                                     ref_shares=ref_shares, cfg=exec_cfg)
            if plan:
                execution.append(plan)

        logger.info(f"Execution plans: {len(execution)} "
                    f"({len(held)} held, {len(buy_watch)} watchlist BUY)")
    except Exception as e:
        logger.warning(f"Execution analysis skipped — {e}")

    html_en, html_zh = build_report(
        metrics, stress, holdings, price_data, last_week=last_week,
        frontier=frontier, suggestions=suggestions,
        has_chart_en=bool(chart_en), has_chart_zh=bool(chart_zh),
        div_candidates=div_candidates, fx_rates=fx_rates,
        stock_factors=stock_factors if stock_factors else None,
        port_exposure=port_exposure,
        attribution=attribution,
        collinearity=collinearity,
        ic_result=ic_result,
        signal_weights=signal_weights if signal_weights else None,
        execution=execution if execution else None,
    )
    mailer.send_report(html_en, html_zh, rag=metrics["overall_rag"],
                       chart_en=chart_en, chart_zh=chart_zh)
    logger.info("=== Portfolio pipeline complete ===")
    return {"metrics": metrics, "html_en": html_en, "html_zh": html_zh}


# ── Backtest pipeline ──────────────────────────────────────────────────────────

def run_backtest_pipeline(config: dict):
    logger = logging.getLogger("backtest")
    logger.info("=== Walk-Forward Backtest — starting ===")

    vol_cfg    = load_yaml("config/volatility.yaml")
    thresholds = load_yaml("config/thresholds.yaml")
    rf         = thresholds.get("risk_free_rate", 0.035)

    watchlist      = load_watchlist()
    equity_tickers = [w["ticker"] for w in watchlist if w.get("asset_class", "equity") == "equity"]

    from data.spy_universe     import load_universe
    from backtest.walk_forward import run_walk_forward, print_summary

    max_sp500 = int(config.get("backtest", {}).get("max_sp500", 100))
    days      = int(config.get("backtest", {}).get("history_days", 1260))

    price_data = load_universe(equity_tickers, days=days, max_sp500=max_sp500)

    results = run_walk_forward(
        price_data,
        rf=rf,
        train_days=int(config.get("backtest", {}).get("train_days", 756)),
        test_days=int(config.get("backtest", {}).get("test_days", 63)),
    )

    if results:
        print_summary(results["summary"])
    else:
        logger.warning("Walk-forward returned no results")

    logger.info("=== Walk-Forward Backtest complete ===")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg     = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logging(log_cfg.get("level", "INFO"), log_cfg.get("file", "logs/agent.log"))
    logger = logging.getLogger("main")

    mode = os.environ.get("RUN_MODE", "full")

    if mode in ("portfolio", "full", "alert_check"):
        run_portfolio_pipeline(run_mode=mode, config=cfg)

    elif mode == "backtest":
        run_backtest_pipeline(cfg)

    elif mode == "paper_trade":
        from strategy.paper_engine import run_paper_trade_pipeline
        run_paper_trade_pipeline(cfg)

    elif mode == "market_brief":
        from notify.market_brief import run_market_brief_pipeline
        run_market_brief_pipeline(cfg)

    elif mode == "heatmap":
        from notify.heatmap import run_heatmap_pipeline
        run_heatmap_pipeline(cfg)

    else:
        logger.error(
            f"Unknown RUN_MODE: {mode!r}. Use: "
            "portfolio | full | alert_check | backtest | paper_trade | "
            "market_brief | heatmap"
        )
        sys.exit(1)
