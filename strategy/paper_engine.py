"""
Paper trading engine.

Pipeline:
  1. expand_universe()      — permanent watchlist + random S&P 500 fills to target_size
  2. screen_universe()      — FF5 regression on all tickers
  3. prune_watchlist()      — remove stale auto-promoted entries (SELL or HOLD>30d)
  4. promote_to_watchlist() — candidates with BUY/SELL signal appended to watchlist.csv
  5. generate_orders()      — compute + execute Alpaca paper trades

Entry point: run_paper_trade_pipeline(config)
"""
import csv
import logging
import os
import re
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Matches the promotion date embedded in auto-promoted notes, e.g. "[2026-05-21]"
_PROMO_DATE_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]")

_EWMA_LAMBDA = 0.94   # same as main risk engine

logger = logging.getLogger(__name__)

_WATCHLIST_PATH = Path("config/watchlist.csv")


# ── Universe expansion ────────────────────────────────────────────────────────

def expand_universe(target: int = 50) -> tuple[list[str], list[str]]:
    """
    Returns (permanent_equity_tickers, candidate_tickers).

    Permanent = current watchlist equity tickers.
    Candidates = random S&P 500 sample (seeded by date, reproducible within a day)
                 that fills universe up to `target` total tickers.
    """
    from data.spy_universe import get_sp500_tickers
    from main import load_watchlist

    watchlist = load_watchlist()
    permanent = [w["ticker"] for w in watchlist if w.get("asset_class", "equity") == "equity"]
    perm_set  = set(permanent)

    sp500      = [t for t in get_sp500_tickers() if t not in perm_set]
    rng        = random.Random(int(date.today().strftime("%Y%m%d")))
    rng.shuffle(sp500)

    fill_n     = max(0, target - len(permanent))
    candidates = sp500[:fill_n]

    logger.info(
        f"Universe: {len(permanent)} permanent + {len(candidates)} S&P 500 candidates "
        f"= {len(permanent) + len(candidates)} total"
    )
    return permanent, candidates


# ── FF5 screening ─────────────────────────────────────────────────────────────

def screen_universe(
    tickers:     list[str],
    price_data:  dict,
    ff5,
    n_strategies: int,
) -> list[dict]:
    """
    Run FF5 regression + robust signal on each ticker.
    Returns list of result dicts sorted by IR descending.
    """
    from strategy.factor_model import run_factor_regression, compute_robust_signal

    results = []
    for ticker in tickers:
        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.returns is None or pd_obj.status not in ("ok", "reduced"):
            continue

        reg = run_factor_regression(ticker, pd_obj.returns, ff5)
        if not reg:
            continue

        robust = compute_robust_signal(ticker, pd_obj.returns, ff5, n_strategies=n_strategies)
        if robust:
            for key in ("psr", "ir_deflated", "robust_signal",
                        "alpha_consistency", "oos_hit_rate"):
                if key in robust:
                    reg[key] = robust[key]

        if pd_obj.closes is not None and not pd_obj.closes.empty:
            reg["price"] = float(pd_obj.closes.iloc[-1])

        results.append(reg)

    results.sort(key=lambda x: x.get("ir") or float("-inf"), reverse=True)
    logger.info(f"FF5 screen: {len(results)}/{len(tickers)} tickers processed")
    return results


# ── Watchlist promotion ───────────────────────────────────────────────────────

def promote_to_watchlist(
    candidate_results: list[dict],
    watchlist_path: Path = _WATCHLIST_PATH,
) -> list[str]:
    """
    Append candidate tickers that have a BUY or SELL signal to watchlist.csv.
    Idempotent: skips tickers already in the watchlist.
    Returns list of newly added tickers.
    """
    from main import load_watchlist

    existing = {w["ticker"] for w in load_watchlist(str(watchlist_path))}

    # Promote only on the robustly-validated signal (PSR + deflated IR +
    # multi-window consistency + OOS), never the naive single-window t(α).
    # This is what keeps the daily random S&P 500 draw from polluting the
    # permanent watchlist with data-snooped false positives.
    to_add = [
        r for r in candidate_results
        if r.get("robust_signal") in ("BUY", "SELL")
        and r["ticker"] not in existing
    ]
    if not to_add:
        logger.info("Watchlist promotion: no new candidates to promote")
        return []

    with open(watchlist_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "weight", "notes", "asset_class", "currency"]
        )
        for r in to_add:
            t_alpha = r.get("t_alpha", 0) or 0
            writer.writerow({
                "ticker":      r["ticker"],
                "weight":      "1.0",
                "notes":       (
                    f"FF5 signal={r['robust_signal']} "
                    f"t={t_alpha:.1f} "
                    f"IR={r.get('ir', 0) or 0:.2f} "
                    f"[{date.today()}]"
                ),
                "asset_class": "equity",
                "currency":    "USD",
            })

    promoted = [r["ticker"] for r in to_add]
    logger.info(f"Promoted to permanent watchlist: {promoted}")
    return promoted


# ── Watchlist pruning ─────────────────────────────────────────────────────────

def prune_watchlist(
    perm_results: list[dict],
    watchlist_path: Path = _WATCHLIST_PATH,
    hold_grace_days: int = 30,
) -> list[str]:
    """
    Remove auto-promoted entries (notes starts with 'FF5 signal=') that meet
    either removal rule.  Manual entries are never touched.

    Rule A — active SELL: current FF5 signal is SELL (t < -1.5)
    Rule B — signal decay: current signal is not BUY AND entry is older than
              hold_grace_days (date parsed from '[YYYY-MM-DD]' in notes)

    Returns list of removed tickers.
    """
    from main import load_watchlist

    watchlist  = load_watchlist(str(watchlist_path))
    signal_map = {r["ticker"]: r.get("signal", "HOLD") for r in perm_results}
    today      = date.today()

    keep:    list[dict] = []
    removed: list[str]  = []

    for entry in watchlist:
        notes = entry.get("notes", "")

        # Manual entry — never touch
        if not notes.startswith("FF5 signal="):
            keep.append(entry)
            continue

        ticker = entry["ticker"]
        signal = signal_map.get(ticker, "HOLD")

        # Rule A: active SELL → remove immediately
        if signal == "SELL":
            removed.append(ticker)
            logger.info(f"Prune A (SELL signal):  removed {ticker}")
            continue

        # Rule B: not BUY + older than grace period → remove
        m         = _PROMO_DATE_RE.search(notes)
        age_days  = (today - date.fromisoformat(m.group(1))).days if m else 0

        if signal != "BUY" and age_days > hold_grace_days:
            removed.append(ticker)
            logger.info(f"Prune B (HOLD {age_days}d > {hold_grace_days}d): removed {ticker}")
            continue

        keep.append(entry)

    if not removed:
        logger.info("Watchlist pruning: nothing to remove")
        return []

    fieldnames = ["ticker", "weight", "notes", "asset_class", "currency"]
    with open(watchlist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keep)

    logger.info(f"Watchlist pruned — removed: {removed}")
    return removed


# ── Portfolio risk snapshot ───────────────────────────────────────────────────

def _paper_portfolio_risk(
    positions:  dict,
    price_data: dict,
    equity:     float,
) -> dict:
    """
    Lightweight risk snapshot for the current paper portfolio.

    Returns:
      n_positions   – number of open positions with price data
      var_1d_95     – 1-day EWMA VaR at 95% confidence (fraction of equity)
      portfolio_beta – CAPM beta vs SPY; 1.0 if SPY unavailable
      hhi           – Herfindahl-Hirschman Index of position weights
      weights        – {ticker: weight} by market value
    """
    # Only include positions that also have return data
    held = [
        t for t in positions
        if price_data.get(t) and price_data[t].returns is not None
        and not price_data[t].returns.empty
    ]
    if not held:
        return {"n_positions": 0, "var_1d_95": 0.0,
                "portfolio_beta": 1.0, "hhi": 0.0, "weights": {}}

    # Market values from Alpaca position fields
    mv: dict[str, float] = {}
    for t in held:
        pos = positions[t]
        raw = pos.get("market_value") or pos.get("market_value_usd") or 0
        mv[t] = float(raw) if raw else 0.0
    total_mv = sum(mv.values()) or equity
    weights  = {t: mv[t] / total_mv for t in held}

    # Align daily returns into a matrix
    ret_df = pd.concat(
        {t: price_data[t].returns for t in held}, axis=1
    ).dropna()
    if len(ret_df) < 5:
        return {"n_positions": len(held), "var_1d_95": 0.0,
                "portfolio_beta": 1.0, "hhi": sum(v**2 for v in weights.values()),
                "weights": weights}

    w = np.array([weights.get(t, 0.0) for t in ret_df.columns])
    if w.sum() > 1e-9:
        w = w / w.sum()

    # EWMA covariance (shared with the main risk engine — single source of truth)
    from risk.risk_engine import ewma_covariance, cf_var

    cov_ewma  = ewma_covariance(ret_df, _EWMA_LAMBDA)
    port_var  = float(w @ cov_ewma @ w)
    sigma_t   = float(np.sqrt(max(port_var, 0.0)))
    var_1d_95_normal = 1.645 * sigma_t

    # Cornish-Fisher VaR: adjusts σ for the portfolio's skew + fat tails. Used as
    # the primary gate value so a left-skewed book is sized down correctly.
    port_ret = (ret_df * w).sum(axis=1)
    var_cf   = cf_var(port_ret, sigma_t, confidence=0.95)
    var_1d_95 = var_cf if var_cf == var_cf else var_1d_95_normal  # nan-safe

    # HHI
    hhi = float(sum(v ** 2 for v in weights.values()))

    # Beta vs SPY (CAPM)
    beta    = 1.0
    spy_pd  = price_data.get("SPY")
    if spy_pd is not None and spy_pd.returns is not None and not spy_pd.returns.empty:
        aligned  = pd.concat(
            {"port": port_ret, "spy": spy_pd.returns}, axis=1
        ).dropna()
        if len(aligned) > 20:
            cov_mat = np.cov(aligned["port"].values, aligned["spy"].values)
            if cov_mat[1, 1] > 1e-9:
                beta = float(cov_mat[0, 1] / cov_mat[1, 1])

    return {
        "n_positions":      len(held),
        "var_1d_95":        var_1d_95,
        "var_1d_95_normal": var_1d_95_normal,
        "portfolio_beta":   beta,
        "hhi":              hhi,
        "weights":          weights,
    }


def _risk_gates(risk: dict, paper_cfg: dict, equity: float) -> tuple[bool, str]:
    """
    Check all risk gates before opening new positions.
    Returns (allow_new_buys, reason_string).
    SELL / close orders always bypass these gates.
    """
    # Gate 1: max open positions
    max_pos = int(paper_cfg.get("max_positions", 10))
    if risk["n_positions"] >= max_pos:
        return False, f"max positions {risk['n_positions']}/{max_pos}"

    # Gate 2: portfolio 1-day VaR
    var_limit = float(paper_cfg.get("var_limit_pct", 0.05))
    if risk["var_1d_95"] > var_limit:
        return False, (
            f"VaR_95 {risk['var_1d_95']*100:.1f}% > limit {var_limit*100:.0f}%"
        )

    # Gate 3: portfolio beta
    max_beta = float(paper_cfg.get("max_portfolio_beta", 1.5))
    if risk["portfolio_beta"] > max_beta:
        return False, (
            f"beta {risk['portfolio_beta']:.2f} > limit {max_beta:.1f}"
        )

    # Gate 4: drawdown from starting equity
    starting = float(paper_cfg.get("starting_equity", 100_000))
    max_dd   = float(paper_cfg.get("max_drawdown_halt", 0.10))
    if starting > 0 and equity < starting * (1.0 - max_dd):
        dd = (starting - equity) / starting
        return False, (
            f"drawdown {dd*100:.1f}% > halt threshold {max_dd*100:.0f}%"
        )

    return True, ""


# ── Rebalance helper ──────────────────────────────────────────────────────────

def _rebalance_orders(
    results:             list[dict],
    positions:           dict,
    equity:              float,
    price_data:          dict,
    risk_weights:        dict,
    max_pos_pct:         float,
    rebalance_threshold: float,
    gates_allow_buys:    bool,
    remaining_bp:        float,
    dry_run:             bool,
    exiting:             set,
) -> tuple[list[dict], float]:
    """
    Trim overweight and top-up underweight BUY-signal holdings.

    Only acts on tickers where robust_signal == BUY and already held.
    Trims bypass risk gates; top-ups respect them and consume buying power.
    Returns (orders, updated_remaining_bp).
    """
    from brokers.alpaca import place_order

    held_buys = [
        r for r in results
        if r.get("robust_signal") == "BUY"
        and r["ticker"] in positions
        and r["ticker"] not in exiting
    ]
    if not held_buys:
        return [], remaining_bp

    irs      = np.array([max(float(r.get("ir") or 0), 0.01) for r in held_buys])
    raw_w    = irs / irs.sum()
    target_w = np.minimum(raw_w, max_pos_pct)

    orders: list[dict] = []

    for r, tgt in zip(held_buys, target_w):
        ticker  = r["ticker"]
        cur_w   = float(risk_weights.get(ticker, 0.0))
        delta_w = tgt - cur_w   # positive = underweight, negative = overweight

        if abs(delta_w) < rebalance_threshold:
            continue

        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
            continue
        price = float(pd_obj.closes.iloc[-1])
        if price <= 0:
            continue

        if delta_w < 0:
            # Overweight — trim (always allowed)
            trim_qty = int(abs(delta_w) * equity / price)
            if trim_qty < 1:
                continue
            held_qty = int(float(positions[ticker].get("qty") or 0))
            trim_qty = min(trim_qty, held_qty)
            if trim_qty < 1:
                continue
            orders.append({
                "ticker": ticker, "side": "sell", "qty": trim_qty,
                "reason": (
                    f"REBALANCE TRIM  cur={cur_w*100:.1f}%"
                    f"  target={tgt*100:.1f}%  Δ={delta_w*100:+.1f}%"
                ),
            })
            if not dry_run:
                try:
                    place_order(ticker, trim_qty, "sell")
                    logger.info(
                        f"Paper: REBALANCE TRIM {trim_qty}x {ticker}"
                        f"  Δw={delta_w*100:+.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"Paper: rebalance trim {ticker} failed — {e}")

        else:
            # Underweight — top up (only if gates pass and buying power allows)
            if not gates_allow_buys:
                continue
            topup_val = min(delta_w * equity, remaining_bp)
            topup_qty = int(topup_val / price)
            if topup_qty < 1:
                continue
            remaining_bp -= topup_qty * price
            orders.append({
                "ticker": ticker, "side": "buy", "qty": topup_qty,
                "reason": (
                    f"REBALANCE TOPUP  cur={cur_w*100:.1f}%"
                    f"  target={tgt*100:.1f}%  Δ={delta_w*100:+.1f}%"
                ),
            })
            if not dry_run:
                try:
                    place_order(ticker, topup_qty, "buy")
                    logger.info(
                        f"Paper: REBALANCE TOPUP {topup_qty}x {ticker}"
                        f"  Δw={delta_w*100:+.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"Paper: rebalance topup {ticker} failed — {e}")

    return orders, remaining_bp


# ── Order generation + execution ──────────────────────────────────────────────

def generate_orders(
    results:     list[dict],
    positions:   dict,
    equity:      float,
    price_data:  dict,
    max_pos_pct:  float = 0.15,
    dry_run:      bool  = False,
    risk:         dict  = None,
    paper_cfg:    dict  = None,
    buying_power: float = None,
) -> list[dict]:
    """
    Compute and (unless dry_run) execute Alpaca paper orders.

    Position sizing:
      IR-proportional allocation — each new BUY gets equity × (IR_i / ΣIR),
      capped at max_pos_pct.  Ensures higher-conviction names get larger slices.

    Risk gates (pre-trade, applied to BUY orders only):
      • Max open positions count
      • Portfolio 1-day EWMA VaR_95 > limit
      • Portfolio beta vs SPY > limit
      • Equity drawdown > halt threshold
    SELL / close orders always execute regardless of gates.
    """
    from brokers.alpaca import place_order, close_position

    # BUY requires the robust (overfitting-resistant) signal — prevents
    # data-snooped false positives from the daily random S&P 500 draw.
    # SELL fires on either raw OR robust signal for fast de-risking.
    buy_signals  = [r for r in results if r.get("robust_signal") == "BUY"]
    sell_signals = [
        r for r in results
        if r.get("signal") == "SELL" or r.get("robust_signal") == "SELL"
    ]
    orders: list[dict] = []
    exiting: set = set()   # tickers already queued for close (sell signal or stop-loss)

    # ── Close SELL-signaled held positions (gates do NOT apply) ───────────────
    for r in sell_signals:
        ticker = r["ticker"]
        if ticker not in positions or ticker in exiting:
            continue
        exiting.add(ticker)
        orders.append({
            "ticker": ticker, "side": "sell",
            "reason": (
                f"SELL signal  t={r.get('t_alpha', 0) or 0:.1f}"
                f"  IR={r.get('ir', 0) or 0:.2f}"
            ),
        })
        if not dry_run:
            try:
                close_position(ticker)
                logger.info(f"Paper: CLOSED {ticker}")
            except Exception as e:
                logger.warning(f"Paper: close {ticker} failed — {e}")

    # ── Stop-loss: close any held position past its unrealised-loss threshold ──
    stop_loss_pct = float((paper_cfg or {}).get("stop_loss_pct", 0.0))
    if stop_loss_pct > 0:
        for ticker, pos in positions.items():
            if ticker in exiting:
                continue
            try:
                plpc = float(pos.get("unrealized_plpc") or 0.0)
            except (TypeError, ValueError):
                continue
            if plpc >= -stop_loss_pct:
                continue
            exiting.add(ticker)
            orders.append({
                "ticker": ticker, "side": "sell",
                "reason": f"STOP-LOSS  unrealised P&L {plpc*100:.1f}% < -{stop_loss_pct*100:.0f}%",
            })
            if not dry_run:
                try:
                    close_position(ticker)
                    logger.info(f"Paper: STOP-LOSS CLOSED {ticker} ({plpc*100:.1f}%)")
                except Exception as e:
                    logger.warning(f"Paper: stop-loss close {ticker} failed — {e}")

    # ── Risk gates — evaluated once before any new BUY ────────────────────────
    allow_buys = True
    if risk is not None and paper_cfg is not None:
        allow_buys, gate_reason = _risk_gates(risk, paper_cfg, equity)
        if not allow_buys:
            logger.warning(f"Paper: BUY orders blocked — {gate_reason}")

    # Budget shared between rebalance top-ups and new buys
    budget       = float(buying_power) if buying_power is not None else equity
    remaining_bp = budget

    # ── Rebalance existing BUY-signal holdings ────────────────────────────────
    rebal_threshold = float((paper_cfg or {}).get("rebalance_threshold", 0.05))
    if risk is not None and rebal_threshold > 0:
        rebal_orders, remaining_bp = _rebalance_orders(
            results, positions, equity, price_data,
            risk_weights=risk.get("weights", {}),
            max_pos_pct=max_pos_pct,
            rebalance_threshold=rebal_threshold,
            gates_allow_buys=allow_buys,
            remaining_bp=remaining_bp,
            dry_run=dry_run,
            exiting=exiting,
        )
        orders.extend(rebal_orders)

    if not allow_buys:
        return orders

    # ── IR-proportional sizing for new BUY positions ──────────────────────────
    new_buys = [r for r in buy_signals if r["ticker"] not in positions]
    if not new_buys:
        return orders

    # Use IR as conviction weight; floor at 0.01 so tickers missing IR still get a slice
    irs     = np.array([max(float(r.get("ir") or 0), 0.01) for r in new_buys])
    raw_w   = irs / irs.sum()                       # proportional weights
    alloc_w = np.minimum(raw_w, max_pos_pct)        # cap each at max_pos_pct
    # Do NOT renormalise: if total < 1 that's intentional (leaves cash buffer)

    for r, w in zip(new_buys, alloc_w):
        ticker = r["ticker"]
        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
            logger.warning(f"Paper: {ticker} BUY skipped — no price data")
            continue

        price      = float(pd_obj.closes.iloc[-1])
        target_val = min(equity * float(w), remaining_bp)
        if target_val < price:
            logger.info(f"Paper: {ticker} BUY skipped — insufficient buying power")
            continue
        qty = int(target_val / price)
        if qty < 1:
            continue
        remaining_bp -= qty * price

        orders.append({
            "ticker":    ticker,
            "side":      "buy",
            "qty":       qty,
            "weight":    float(w),
            "price_est": price,
            "value_usd": qty * price,
            "reason": (
                f"BUY signal  t={r.get('t_alpha', 0) or 0:.1f}"
                f"  IR={r.get('ir', 0) or 0:.2f}"
                f"  alloc={w*100:.1f}%"
            ),
        })
        if not dry_run:
            try:
                place_order(ticker, qty, "buy")
                logger.info(
                    f"Paper: BUY {qty}x {ticker} @ ~${price:.2f}"
                    f"  (alloc {w*100:.1f}%, IR={r.get('ir', 0) or 0:.2f})"
                )
            except Exception as e:
                logger.warning(f"Paper: buy {ticker} failed — {e}")

    return orders


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_paper_trade_pipeline(config: dict) -> dict:
    """
    Orchestrates the full paper trading pipeline. Called from main.py
    when RUN_MODE=paper_trade.
    """
    from data.price_loader import load_prices
    from risk.optimizer    import fetch_ff5
    from brokers.alpaca    import is_configured, get_account, get_positions

    paper_cfg   = config.get("paper_trade", {})
    target_size = int(paper_cfg.get("universe_size", 50))
    max_pos_pct = float(paper_cfg.get("max_position_pct", 0.15))
    ff_days     = int(config.get("factor_model", {}).get("window_days", 252))
    dry_run     = os.environ.get("PAPER_DRY_RUN", "false").lower() == "true"

    logger.info(
        f"=== Paper Trade Pipeline  universe={target_size}  "
        f"max_pos={max_pos_pct*100:.0f}%  dry_run={dry_run} ==="
    )

    # 1. Expand universe
    permanent, candidates = expand_universe(target_size)
    all_tickers = permanent + candidates

    # 2. Load prices (252-day window); include SPY for beta computation
    fetch_tickers = list(set(all_tickers + ["SPY"]))
    logger.info(f"Loading {len(fetch_tickers)} tickers …")
    price_data = load_prices(fetch_tickers, days=ff_days)

    # 3. FF5 data (uses 20h cache from daily_run if available)
    ff5 = fetch_ff5()
    if ff5 is None:
        logger.error("FF5 data unavailable — aborting paper trade pipeline")
        return {}

    # Multiple-testing deflation must reflect the full effective search space.
    # We draw daily from the entire S&P 500, so floor at 500 rather than the
    # current-day sample size — otherwise deflated IR is far too lenient.
    n_strategies = max(len(all_tickers), 500)

    # 4. Screen permanent watchlist
    logger.info("Screening permanent watchlist …")
    perm_results = screen_universe(permanent, price_data, ff5, n_strategies)

    # 5. Prune stale auto-promoted entries before promoting new ones
    logger.info("Pruning stale auto-promoted watchlist entries …")
    pruned = prune_watchlist(perm_results)

    # 6. Screen S&P 500 candidates
    logger.info("Screening S&P 500 candidates …")
    cand_results = screen_universe(candidates, price_data, ff5, n_strategies)

    # 7. Promote candidates with signal to permanent watchlist
    promoted = promote_to_watchlist(cand_results)

    all_results = perm_results + cand_results

    # 8. Alpaca paper trading
    orders:        list[dict] = []
    account_info:  dict       = {}
    risk_snapshot: dict       = {}

    if is_configured():
        try:
            account   = get_account()
            equity    = float(account.get("equity", 100_000))
            positions = get_positions()
            account_info = {
                "equity":       equity,
                "cash":         float(account.get("cash", 0)),
                "n_positions":  len(positions),
                "buying_power": float(account.get("buying_power", 0)),
            }
            logger.info(
                f"Alpaca paper account: equity=${equity:,.0f}  "
                f"cash=${account_info['cash']:,.0f}  "
                f"positions={account_info['n_positions']}"
            )

            # Compute portfolio risk snapshot for gate evaluation
            risk_snapshot = _paper_portfolio_risk(positions, price_data, equity)
            logger.info(
                f"Paper portfolio risk: "
                f"VaR_95={risk_snapshot['var_1d_95']*100:.1f}%  "
                f"beta={risk_snapshot['portfolio_beta']:.2f}  "
                f"HHI={risk_snapshot['hhi']:.3f}  "
                f"n={risk_snapshot['n_positions']}"
            )

            orders = generate_orders(
                all_results, positions, equity, price_data,
                max_pos_pct=max_pos_pct, dry_run=dry_run,
                risk=risk_snapshot, paper_cfg=paper_cfg,
                buying_power=account_info.get("buying_power"),
            )
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")
    else:
        logger.warning("ALPACA_API_KEY not set — screening only, no paper trades placed")
        risk_snapshot = {}

    buys  = sum(1 for o in orders if o["side"] == "buy")
    sells = sum(1 for o in orders if o["side"] == "sell")
    logger.info(
        f"=== Paper Trade complete: {buys} buys  {sells} sells  "
        f"{len(promoted)} promoted  {len(pruned)} pruned ==="
    )

    return {
        "perm_results":      perm_results,
        "candidate_results": cand_results,
        "all_results":       all_results,
        "promoted":          promoted,
        "pruned":            pruned,
        "orders":            orders,
        "account":           account_info,
        "risk":              risk_snapshot,
    }
