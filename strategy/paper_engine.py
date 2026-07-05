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

_WATCHLIST_PATH  = Path("config/watchlist.csv")
# Durable paper-trade state lives under data/ (committed by CI) — NOT under
# cache/, which is .gitignored and evicted between GitHub Actions runs. Keeping
# these here is what makes NAV history, trade log and trailing-stop watermarks
# survive across runs so the performance metrics are meaningful.
_PAPER_STATE_DIR  = Path("data/paper_state")
_WATERMARKS_PATH  = _PAPER_STATE_DIR / "paper_highs.json"
_TRADE_LOG_PATH   = _PAPER_STATE_DIR / "paper_trade_log.json"
_NAV_HISTORY_PATH = _PAPER_STATE_DIR / "paper_nav_history.json"
# Earnings lookups are an ephemeral per-day cache — fine to keep in cache/.
_EARNINGS_CACHE_PATH = Path("cache/paper_earnings_cache.json")


# ── Watermark / trade-log helpers ─────────────────────────────────────────────

def _load_watermarks() -> dict:
    import json
    try:
        if _WATERMARKS_PATH.exists():
            return json.loads(_WATERMARKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_watermarks(wm: dict) -> None:
    import json
    _WATERMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WATERMARKS_PATH.write_text(json.dumps(wm, indent=2), encoding="utf-8")


def _append_trade_log(orders: list[dict], account: dict) -> None:
    import json
    import datetime as dt
    log: list[dict] = []
    if _TRADE_LOG_PATH.exists():
        try:
            log = json.loads(_TRADE_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for o in orders:
        log.append({
            "timestamp": ts,
            "ticker":    o.get("ticker"),
            "side":      o.get("side"),
            "qty":       o.get("qty"),
            "price_est": o.get("price_est"),
            "value_usd": o.get("value_usd"),
            "reason":    o.get("reason", ""),
            "equity":    account.get("equity"),
        })
    _TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRADE_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


def _record_nav_snapshot(account: dict) -> None:
    """Append today's equity/cash to the NAV history file."""
    import json
    import datetime as dt
    try:
        history: list[dict] = []
        if _NAV_HISTORY_PATH.exists():
            history = json.loads(_NAV_HISTORY_PATH.read_text(encoding="utf-8"))
        today_str = dt.date.today().isoformat()
        # Overwrite same-day entry if we run multiple times
        history = [h for h in history if h.get("date") != today_str]
        history.append({
            "date":   today_str,
            "equity": float(account.get("equity", 0) or 0),
            "cash":   float(account.get("cash", 0) or 0),
        })
        _NAV_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _NAV_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Paper: NAV snapshot failed — {e}")


def _compute_performance(history: list[dict]) -> dict:
    """Return {total_return, sharpe, sortino, max_drawdown, n_days} from NAV history."""
    if len(history) < 2:
        return {"total_return": 0.0, "sharpe": None, "sortino": None,
                "max_drawdown": 0.0, "n_days": len(history)}
    equities = np.array([h["equity"] for h in history], dtype=float)
    returns  = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1.0)
    total_return = float(equities[-1] / equities[0] - 1.0) if equities[0] > 0 else 0.0
    mean_r  = float(np.mean(returns))
    std_r   = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe  = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-10 else None
    neg_r   = returns[returns < 0]
    dstd    = float(np.std(neg_r, ddof=1)) if len(neg_r) > 1 else 0.0
    sortino = float(mean_r / dstd * np.sqrt(252)) if dstd > 1e-10 else None
    peak    = np.maximum.accumulate(equities)
    max_dd  = float(np.max((peak - equities) / np.where(peak > 0, peak, 1.0)))
    return {
        "total_return": total_return,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": max_dd,
        "n_days":       len(history),
    }


def _load_earnings_cache() -> dict:
    import json
    try:
        if _EARNINGS_CACHE_PATH.exists():
            return json.loads(_EARNINGS_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_earnings_cache(cache: dict) -> None:
    import json
    try:
        _EARNINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EARNINGS_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _coerce_date(v):
    """Best-effort coercion of yfinance date values to a datetime.date."""
    import datetime as dt
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        try:
            return dt.date.fromisoformat(str(v)[:10])
        except Exception:
            return None


def _fetch_next_earnings_date(ticker: str):
    """
    Return the next upcoming earnings date (datetime.date) or None.

    Uses Ticker.calendar — far more reliable than the heavily rate-limited
    .info dict. Returns None on both 'no upcoming date found' AND on lookup
    failure; the caller logs the failure case distinctly so a silently broken
    gate is visible in the logs.
    """
    import datetime as dt
    import yfinance as yf
    tk    = yf.Ticker(ticker)
    cal   = tk.calendar
    dates = []
    if isinstance(cal, dict):                       # newer yfinance
        ed = cal.get("Earnings Date") or cal.get("Earnings Date Start")
        if ed is not None:
            for d in (ed if isinstance(ed, (list, tuple)) else [ed]):
                dates.append(_coerce_date(d))
    elif cal is not None and hasattr(cal, "empty") and not cal.empty:  # older DataFrame
        try:
            if "Earnings Date" in list(cal.index):
                for v in cal.loc["Earnings Date"].values:
                    dates.append(_coerce_date(v))
        except Exception:
            pass
    today  = dt.date.today()
    future = [d for d in dates if d is not None and d >= today]
    return min(future) if future else None


def _near_earnings(ticker: str, avoid_days: int) -> bool:
    """
    Return True if ticker reports earnings within avoid_days calendar days.

    Results are cached per calendar day to avoid hammering yfinance across the
    BUY candidate loop. On a *lookup failure* we log a WARNING (so a broken gate
    is observable) and return False so a transient yfinance outage doesn't block
    all trading.
    """
    if avoid_days <= 0:
        return False
    import datetime as dt
    today = dt.date.today()
    key   = ticker.upper()
    cache = _load_earnings_cache()
    entry = cache.get(key)

    if entry and entry.get("date") == today.isoformat():
        nd = entry.get("next_earnings")
        if nd is None:
            return False
        return 0 <= (dt.date.fromisoformat(nd) - today).days <= avoid_days

    try:
        next_date = _fetch_next_earnings_date(ticker)
    except Exception as e:
        logger.warning(
            f"Paper: earnings lookup FAILED for {ticker} — gate skipped this run ({e})"
        )
        return False

    cache[key] = {
        "date":          today.isoformat(),
        "next_earnings": next_date.isoformat() if next_date else None,
    }
    _save_earnings_cache(cache)
    if next_date is None:
        return False
    return 0 <= (next_date - today).days <= avoid_days


def _fetch_vix() -> float | None:
    """Fetch the latest VIX close from yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Paper: VIX fetch failed — {e}")
    return None


def _send_paper_summary(
    orders: list[dict], account: dict, risk: dict, config: dict,
    perf: dict = None, market_open: bool = True,
) -> None:
    """Send a brief HTML email summarising today's paper trade orders and risk snapshot."""
    import datetime as dt
    try:
        from notify.mailer import _smtp_send
    except Exception as e:
        logger.warning(f"Paper: mailer import failed — {e}")
        return

    buys  = [o for o in orders if o.get("side") == "buy"]
    sells = [o for o in orders if o.get("side") == "sell"]

    def _order_rows() -> str:
        if not orders:
            return (
                "<tr><td colspan='4' style='padding:14px;text-align:center;"
                "color:#bdc3c7;font-size:13px'>No orders today / 今日无订单</td></tr>"
            )
        rows = ""
        _side_style = {
            "buy":   ("#27ae60", "#eafaf1", "买入"),
            "sell":  ("#e74c3c", "#fdf0ef", "卖出"),
            "short": ("#8e44ad", "#f5eef8", "做空"),
            "cover": ("#2980b9", "#ebf5fb", "回补"),
        }
        for o in orders:
            side = o.get("side", "")
            sc, sc_bg, side_zh = _side_style.get(side, ("#7f8c8d", "#f4f6f6", side))
            px        = o.get("price_est")
            price_str = f"${px:.2f}" if px is not None else "—"
            reason    = o.get("reason", "")
            rows += f"""
<tr style="background:{sc_bg}">
  <td style="padding:10px 10px 4px;font-size:15px;font-weight:bold;
             color:#2c3e50;border-top:1px solid #eee">
    {o.get('ticker','')}
  </td>
  <td style="padding:10px 10px 4px;text-align:center;border-top:1px solid #eee">
    <span style="background:{sc};color:#fff;font-size:11px;font-weight:bold;
                 padding:3px 8px;border-radius:3px">
      {side.upper()} / {side_zh}
    </span>
  </td>
  <td style="padding:10px 10px 4px;text-align:center;font-size:14px;
             border-top:1px solid #eee;color:#2c3e50">
    {o.get('qty','—')}
  </td>
  <td style="padding:10px 10px 4px;text-align:right;font-size:14px;
             border-top:1px solid #eee;color:#2c3e50;white-space:nowrap">
    {price_str}
  </td>
</tr>
<tr style="background:{sc_bg}">
  <td colspan="4" style="padding:2px 10px 10px;font-size:11px;
                          color:#7f8c8d;word-break:break-word">
    {reason}
  </td>
</tr>"""
        return rows

    equity = account.get("equity", 0) or 0
    cash   = account.get("cash", 0) or 0
    n_pos  = account.get("n_positions", 0) or 0
    var_95 = (risk.get("var_1d_95", 0) or 0) * 100
    beta   = risk.get("portfolio_beta", 0) or 0
    hhi    = risk.get("hhi", 0) or 0

    def _stat(label_en: str, label_zh: str, value: str, bold: bool = False) -> str:
        v = f"<b>{value}</b>" if bold else value
        return (
            f"<tr>"
            f"<td style='padding:8px 12px;font-size:13px'>"
            f"  <span style='color:#555'>{label_en}</span><br>"
            f"  <span style='color:#aaa;font-size:11px'>{label_zh}</span>"
            f"</td>"
            f"<td style='padding:8px 12px;font-size:14px;text-align:right'>{v}</td>"
            f"</tr>"
        )

    buys_zh  = f"{len(buys)} 买入"
    sells_zh = f"{len(sells)} 卖出"

    # Performance section
    def _perf_rows() -> str:
        if not perf or perf.get("n_days", 0) < 2:
            return ""
        tr = perf.get("total_return", 0) * 100
        tr_color = "#27ae60" if tr >= 0 else "#e74c3c"
        sharpe_str  = f"{perf['sharpe']:.2f}"  if perf.get("sharpe")  is not None else "—"
        sortino_str = f"{perf['sortino']:.2f}" if perf.get("sortino") is not None else "—"
        dd_str = f"{perf.get('max_drawdown', 0)*100:.1f}%"
        return f"""
        <tr style="background:#f8f9fa">
          <td colspan="2" style="padding:10px 12px;font-size:11px;font-weight:bold;
                                  color:#95a5a6;letter-spacing:.5px">
            PERFORMANCE ({perf.get('n_days','?')} days) / 累计表现
          </td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-size:13px">
            <span style="color:#555">Total Return / 累计收益</span><br>
            <span style="color:#aaa;font-size:11px"></span>
          </td>
          <td style="padding:8px 12px;font-size:14px;text-align:right;
                     font-weight:bold;color:{tr_color}">
            {tr:+.2f}%
          </td>
        </tr>
        {_stat("Sharpe (ann.)", "夏普比率（年化）", sharpe_str)}
        {_stat("Sortino (ann.)", "索提诺比率（年化）", sortino_str)}
        {_stat("Max Drawdown", "最大回撤", dd_str)}"""

    mkt_notice = ""
    if not market_open:
        mkt_notice = (
            "<tr><td colspan='2' style='background:#fef9e7;padding:10px 12px;"
            "font-size:12px;color:#d68910;border-bottom:1px solid #f9e79f'>"
            "Market closed — screening only, no new orders placed. / "
            "市场已休市，仅完成筛选，未下单。</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Paper Trade Summary</title>
</head>
<body style="margin:0;padding:0;background:#f5f6fa;
             font-family:Arial,Helvetica,sans-serif">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f6fa">
<tr><td align="center" style="padding:16px 8px">

  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:580px;background:#fff;border-radius:6px;
                overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)">

    <!-- Header -->
    <tr><td style="background:#2c3e50;padding:20px 20px 16px">
      <p style="margin:0;font-size:20px;font-weight:bold;color:#fff">
        Paper Trade Summary &nbsp;<span style="font-size:14px;font-weight:normal;
        color:rgba(255,255,255,.7)">/ 模拟交易摘要</span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {dt.date.today()} &nbsp;·&nbsp;
        {len(buys)} buy / {buys_zh} &nbsp;·&nbsp;
        {len(sells)} sell / {sells_zh}
      </p>
    </td></tr>

    <!-- Account + Risk stats -->
    <tr><td style="padding:0">
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-bottom:2px solid #ecf0f1">
        {mkt_notice}
        <!-- Account header -->
        <tr style="background:#f8f9fa">
          <td colspan="2" style="padding:10px 12px;font-size:11px;font-weight:bold;
                                  color:#95a5a6;letter-spacing:.5px">
            ACCOUNT / 账户
          </td>
        </tr>
        {_stat("Equity", "净值", f"${equity:,.0f}", bold=True)}
        {_stat("Cash", "现金", f"${cash:,.0f}")}
        {_stat("Open positions", "持仓数", str(n_pos))}
        <!-- Risk header -->
        <tr style="background:#f8f9fa">
          <td colspan="2" style="padding:10px 12px;font-size:11px;font-weight:bold;
                                  color:#95a5a6;letter-spacing:.5px">
            RISK SNAPSHOT / 风险快照
          </td>
        </tr>
        {_stat("VaR₉₅ (1-day)", "单日风险价值", f"{var_95:.1f}%", bold=True)}
        {_stat("Beta vs SPY", "市场贝塔", f"{beta:.2f}")}
        {_stat("HHI concentration", "集中度指数", f"{hhi:.3f}")}
        {_perf_rows()}
      </table>
    </td></tr>

    <!-- Orders table -->
    <tr><td style="padding:0">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr style="background:#2c3e50">
          <th style="padding:10px;text-align:left;font-size:12px;
                     color:#fff;font-weight:bold;width:30%">
            Ticker
          </th>
          <th style="padding:10px;text-align:center;font-size:12px;
                     color:#fff;font-weight:bold;width:22%">
            Side / 方向
          </th>
          <th style="padding:10px;text-align:center;font-size:12px;
                     color:#fff;font-weight:bold;width:18%">
            Qty / 数量
          </th>
          <th style="padding:10px;text-align:right;font-size:12px;
                     color:#fff;font-weight:bold;width:30%">
            Price / 价格
          </th>
        </tr>
        {_order_rows()}
      </table>
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center">
        Paper account only — no real money &nbsp;·&nbsp; 仅模拟账户，非真实资金
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>
"""

    subject = (
        f"[Paper Trade] {dt.date.today()} — "
        f"{len(buys)}B {len(sells)}S  equity=${equity:,.0f}"
    )
    try:
        _smtp_send(html, subject)
        logger.info("Paper: summary email sent")
    except Exception as e:
        logger.warning(f"Paper: summary email failed — {e}")


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

    # nan-safe sort: treat NaN IR as -inf, but keep a legitimate IR of 0.0
    results.sort(
        key=lambda x: x.get("ir") if x.get("ir") == x.get("ir") else float("-inf"),
        reverse=True,
    )
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

    # Guard against a file missing its final newline — appending would
    # otherwise merge the first new row into the last existing line.
    needs_nl = False
    if watchlist_path.exists():
        raw = watchlist_path.read_bytes()
        needs_nl = bool(raw) and not raw.endswith(b"\n")

    with open(watchlist_path, "a", newline="", encoding="utf-8") as f:
        if needs_nl:
            f.write("\r\n")
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


def _risk_gates(
    risk: dict, paper_cfg: dict, equity: float, vix: float = None
) -> tuple[bool, str]:
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

    # Gate 5: VIX spike
    vix_threshold = float(paper_cfg.get("vix_halt_threshold", 0))
    if vix is not None and vix_threshold > 0 and vix > vix_threshold:
        return False, f"VIX {vix:.1f} > halt threshold {vix_threshold:.0f}"

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
    cgt_rate:            float,
    gates_allow_buys:    bool,
    remaining_bp:        float,
    dry_run:             bool,
    exiting:             set,
) -> tuple[list[dict], float]:
    """
    Trim overweight and top-up underweight BUY-signal holdings.

    Only acts on tickers where robust_signal == BUY and already held.
    Trims bypass risk gates; top-ups respect them and consume buying power.

    CGT friction (Irish 33%): for profitable positions, the effective trim
    threshold is raised by gain_fraction × cgt_rate so that small drifts on
    large winners are left alone rather than triggering a taxable disposal.
    Loss positions have zero friction and are trimmed at the base threshold.

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

    # Targets must share the same basis as risk_weights (fraction of the WHOLE
    # portfolio). The BUY sleeve keeps its current total weight and is merely
    # redistributed IR-proportionally within itself. Normalising over the
    # sleeve alone (summing to 1) would make every BUY name look underweight
    # whenever non-BUY positions exist → perpetual top-ups each run.
    sleeve_total = sum(float(risk_weights.get(r["ticker"], 0.0)) for r in held_buys)
    if sleeve_total <= 0:
        return [], remaining_bp

    irs      = np.array([max(float(r.get("ir") or 0), 0.01) for r in held_buys])
    raw_w    = irs / irs.sum()
    target_w = np.minimum(raw_w * sleeve_total, max_pos_pct)

    orders: list[dict] = []

    for r, tgt in zip(held_buys, target_w):
        ticker  = r["ticker"]
        cur_w   = float(risk_weights.get(ticker, 0.0))
        delta_w = tgt - cur_w   # positive = underweight, negative = overweight

        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
            continue
        price = float(pd_obj.closes.iloc[-1])
        if price <= 0:
            continue

        if delta_w < 0:
            # Overweight — trim (always allowed, but CGT raises effective threshold)
            try:
                plpc = float(positions[ticker].get("unrealized_plpc") or 0.0)
            except (TypeError, ValueError):
                plpc = 0.0
            gain_frac       = max(0.0, plpc / (1.0 + plpc)) if plpc > 0 else 0.0
            cgt_friction    = gain_frac * cgt_rate
            effective_thresh = rebalance_threshold + cgt_friction
            if abs(delta_w) < effective_thresh:
                logger.debug(
                    f"Paper: REBALANCE skip {ticker}  |Δw|={abs(delta_w)*100:.1f}%"
                    f" < threshold {effective_thresh*100:.1f}% (incl. CGT friction"
                    f" {cgt_friction*100:.1f}%)"
                )
                continue
            trim_qty = round(abs(delta_w) * equity / price, 6)
            if trim_qty < 0.001:
                continue
            held_qty = float(positions[ticker].get("qty") or 0)
            trim_qty = min(trim_qty, held_qty)
            if trim_qty < 0.001:
                continue
            orders.append({
                "ticker":    ticker, "side": "sell", "qty": trim_qty,
                "price_est": price,
                "value_usd": trim_qty * price,
                "reason": (
                    f"REBALANCE TRIM  cur={cur_w*100:.1f}%"
                    f"  target={tgt*100:.1f}%  delta={delta_w*100:+.1f}%"
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
            # Underweight — top up (buying has no CGT impact; gate + BP check only)
            if abs(delta_w) < rebalance_threshold:
                continue
            if not gates_allow_buys:
                continue
            topup_val = min(delta_w * equity, remaining_bp)
            topup_qty = round(topup_val / price, 6)
            if topup_qty < 0.001:
                continue
            remaining_bp -= topup_qty * price
            orders.append({
                "ticker":    ticker, "side": "buy", "qty": topup_qty,
                "price_est": price,
                "value_usd": topup_qty * price,
                "reason": (
                    f"REBALANCE TOPUP  cur={cur_w*100:.1f}%"
                    f"  target={tgt*100:.1f}%  delta={delta_w*100:+.1f}%"
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
    max_pos_pct:        float = 0.15,
    dry_run:            bool  = False,
    risk:               dict  = None,
    paper_cfg:          dict  = None,
    buying_power:       float = None,
    vix:                float = None,
    market_open:        bool  = True,
    cash:               float = None,
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

    Core parking (paper_cfg["core_parking_ticker"], e.g. SPY):
      Idle cash above core_cash_buffer_pct is swept into the core ETF at the
      end of each run, and the core is sold to fund new signal BUYs. The core
      is exempt from signal trading, per-position caps, rebalancing, and
      stop-losses — market-level de-risking is governed by the VIX/drawdown
      gates (which stop new sweeps but never force-liquidate the core).

    Short overlay (paper_cfg["short_max_total_pct"] > 0):
      Robust SELL signals on non-held names open whole-share shorts of
      short_pos_pct each, capped at short_max_total_pct total exposure.
      Shorts are covered when the robust signal fades and are protected by
      the same stop-loss as longs. Order sides: "short" / "cover".
      Short-sale proceeds are collateral, not sweepable cash — the caller
      passes cash net of short market value and shorts/covers are cash-neutral
      on that basis.
    """
    from brokers.alpaca import place_order, close_position

    core_ticker = str((paper_cfg or {}).get("core_parking_ticker", "") or "").upper()
    core_buffer = float((paper_cfg or {}).get("core_cash_buffer_pct", 0.02))
    short_pos_pct   = float((paper_cfg or {}).get("short_pos_pct", 0.04))
    short_max_total = float((paper_cfg or {}).get("short_max_total_pct", 0.0))

    def _pos_qty(pos: dict) -> float:
        try:
            return float(pos.get("qty") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    short_held = {t for t, p in positions.items() if _pos_qty(p) < 0}

    # BUY requires the robust (overfitting-resistant) signal — prevents
    # data-snooped false positives from the daily random S&P 500 draw.
    # SELL fires on either raw OR robust signal for fast de-risking.
    # The core parking ETF never trades on its own FF5 signal.
    buy_signals  = [
        r for r in results
        if r.get("robust_signal") == "BUY" and r["ticker"] != core_ticker
    ]
    sell_signals = [
        r for r in results
        if (r.get("signal") == "SELL" or r.get("robust_signal") == "SELL")
        and r["ticker"] != core_ticker
    ]
    orders: list[dict] = []
    exiting: set = set()   # tickers already queued for close (sell signal or stop-loss)

    # ── Close SELL-signaled held LONG positions (gates do NOT apply) ──────────
    for r in sell_signals:
        ticker = r["ticker"]
        if ticker not in positions or ticker in exiting or ticker in short_held:
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

    # ── Cover shorts whose robust SELL signal has faded (gates do NOT apply) ──
    result_by_ticker = {r["ticker"]: r for r in results}
    for ticker in sorted(short_held):
        if ticker in exiting:
            continue
        r = result_by_ticker.get(ticker)
        # No screening result today (data gap / rotated out) → hold the short
        if r is None or r.get("robust_signal") == "SELL":
            continue
        exiting.add(ticker)
        pos = positions[ticker]
        _qty = abs(_pos_qty(pos))
        try:
            _mv = abs(float(pos.get("market_value") or 0.0))
        except (TypeError, ValueError):
            _mv = 0.0
        orders.append({
            "ticker": ticker, "side": "cover", "qty": _qty,
            "value_usd": _mv,
            "reason": f"COVER  robust signal now {r.get('robust_signal', 'HOLD')}",
        })
        if not dry_run:
            try:
                close_position(ticker)
                logger.info(f"Paper: COVERED short {ticker}")
            except Exception as e:
                logger.warning(f"Paper: cover {ticker} failed — {e}")

    # ── Stop-loss: close any held position past its unrealised-loss threshold ──
    stop_loss_pct = float((paper_cfg or {}).get("stop_loss_pct", 0.0))
    if stop_loss_pct > 0:
        for ticker, pos in positions.items():
            if ticker in exiting or ticker == core_ticker:
                continue
            try:
                plpc = float(pos.get("unrealized_plpc") or 0.0)
            except (TypeError, ValueError):
                continue
            if plpc >= -stop_loss_pct:
                continue
            exiting.add(ticker)
            _pd  = price_data.get(ticker)
            _px  = float(_pd.closes.iloc[-1]) if (_pd and _pd.closes is not None and not _pd.closes.empty) else None
            _qty = abs(int(_pos_qty(pos)))
            orders.append({
                "ticker":    ticker,
                "side":      "cover" if ticker in short_held else "sell",
                "qty":       _qty,
                "price_est": _px,
                "value_usd": _qty * _px if _px else None,
                "reason":    f"STOP-LOSS  unrealised P&L {plpc*100:.1f}% < -{stop_loss_pct*100:.0f}%",
            })
            if not dry_run:
                try:
                    close_position(ticker)
                    logger.info(f"Paper: STOP-LOSS CLOSED {ticker} ({plpc*100:.1f}%)")
                except Exception as e:
                    logger.warning(f"Paper: stop-loss close {ticker} failed — {e}")

    # ── Trailing stop: close positions that have fallen >X% from their high ─────
    trailing_pct = float((paper_cfg or {}).get("trailing_stop_pct", 0.0))
    if trailing_pct > 0:
        watermarks = _load_watermarks()
        for ticker, pos in positions.items():
            if ticker in exiting or ticker == core_ticker or ticker in short_held:
                continue
            pd_obj = price_data.get(ticker)
            if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
                continue
            price = float(pd_obj.closes.iloc[-1])
            if price <= 0:
                continue
            prev_wm = float(watermarks.get(ticker, price))
            new_wm  = max(prev_wm, price)
            watermarks[ticker] = new_wm
            drop = (new_wm - price) / new_wm
            if drop >= trailing_pct:
                exiting.add(ticker)
                del watermarks[ticker]
                _qty = int(float(pos.get("qty") or 0))
                orders.append({
                    "ticker":    ticker, "side": "sell", "qty": _qty,
                    "price_est": price,
                    "value_usd": _qty * price,
                    "reason":    f"TRAILING STOP  {drop*100:.1f}% below high ${new_wm:.2f}",
                })
                if not dry_run:
                    try:
                        close_position(ticker)
                        logger.info(
                            f"Paper: TRAILING STOP CLOSED {ticker}"
                            f" ({drop*100:.1f}% from high ${new_wm:.2f})"
                        )
                    except Exception as e:
                        logger.warning(f"Paper: trailing stop close {ticker} failed — {e}")
        # Clear watermarks for positions no longer held after exits
        held = set(positions) - exiting
        watermarks = {t: wm for t, wm in watermarks.items() if t in held}
        _save_watermarks(watermarks)

    # ── Risk gates — evaluated once before any new BUY ────────────────────────
    allow_buys = True
    if not market_open:
        allow_buys = False
        logger.info("Paper: BUY orders skipped — market closed")
    elif risk is not None and paper_cfg is not None:
        gate_risk = risk
        if core_ticker and core_ticker in positions and risk.get("n_positions", 0) > 0:
            # The parking ETF is cash-equivalent, not an alpha bet — it must
            # not consume a max_positions slot.
            gate_risk = {**risk, "n_positions": risk["n_positions"] - 1}
        allow_buys, gate_reason = _risk_gates(gate_risk, paper_cfg, equity, vix=vix)
        if not allow_buys:
            logger.warning(f"Paper: BUY orders blocked — {gate_reason}")

    # Budget shared between rebalance top-ups and new buys
    budget       = float(buying_power) if buying_power is not None else equity
    remaining_bp = budget

    # Estimated free cash, updated as orders are planned (sells add, buys drain).
    # Shorts/covers are neutral: `cash` is net of short market value, and on
    # that basis opening or covering a short at market leaves it unchanged.
    cash_avail = float(cash) if cash is not None else max(0.0, budget)

    def _planned_cash_delta() -> float:
        delta = 0.0
        for o in orders:
            if o["side"] not in ("buy", "sell"):
                continue
            val = o.get("value_usd")
            if val is None:
                try:
                    val = float(positions.get(o["ticker"], {}).get("market_value") or 0.0)
                except (TypeError, ValueError):
                    val = 0.0
            delta += val if o["side"] == "sell" else -val
        return delta

    # ── Rebalance existing BUY-signal holdings ────────────────────────────────
    rebal_threshold = float((paper_cfg or {}).get("rebalance_threshold", 0.05))
    cgt_rate        = float((paper_cfg or {}).get("cgt_rate", 0.33))
    if risk is not None and rebal_threshold > 0:
        rebal_orders, remaining_bp = _rebalance_orders(
            results, positions, equity, price_data,
            risk_weights=risk.get("weights", {}),
            max_pos_pct=max_pos_pct,
            rebalance_threshold=rebal_threshold,
            cgt_rate=cgt_rate,
            gates_allow_buys=allow_buys,
            remaining_bp=remaining_bp,
            dry_run=dry_run,
            exiting=exiting | short_held | ({core_ticker} if core_ticker else set()),
        )
        orders.extend(rebal_orders)

    if not allow_buys:
        return orders

    # ── IR-proportional sizing for new BUY positions ──────────────────────────
    new_buys = [r for r in buy_signals if r["ticker"] not in positions]
    if not new_buys:
        remaining_bp = _open_short_orders(
            results, positions, exiting, orders, price_data, equity,
            paper_cfg, remaining_bp, dry_run, core_ticker, short_held,
        )
        _core_sweep(
            core_ticker, core_buffer, cash_avail + _planned_cash_delta(),
            orders, positions, price_data, equity, dry_run,
        )
        return orders

    earnings_avoid_days = int((paper_cfg or {}).get("earnings_avoid_days", 0))

    # ADV participation cap: a single day's BUY must not exceed participation_pct
    # of the name's average daily share volume, regardless of conviction sizing.
    participation_pct = float((paper_cfg or {}).get("participation_pct", 0) or 0)
    adv_cap_shares: dict = {}
    if participation_pct > 0:
        try:
            from data.price_loader import load_ohlcv
            from strategy.execution import participation_cap_shares
            adv_window = int((paper_cfg or {}).get("adv_window", 20))
            ohlcv = load_ohlcv([r["ticker"] for r in new_buys], days=max(adv_window * 3, 60))
            for tk, df in ohlcv.items():
                cap = participation_cap_shares(df, adv_window, participation_pct)
                if cap:
                    adv_cap_shares[tk] = cap
        except Exception as e:
            logger.warning(f"Paper: ADV participation cap unavailable — {e}")

    # Use IR as conviction weight; floor at 0.01 so tickers missing IR still get a slice
    irs     = np.array([max(float(r.get("ir") or 0), 0.01) for r in new_buys])
    raw_w   = irs / irs.sum()                       # proportional weights
    alloc_w = np.minimum(raw_w, max_pos_pct)        # cap each at max_pos_pct
    # Do NOT renormalise: if total < 1 that's intentional (leaves cash buffer)

    # ── Fund new BUYs from the core parking position when cash is short ───────
    core_pos = positions.get(core_ticker) if core_ticker else None
    if core_pos is not None:
        intended  = float(equity * alloc_w.sum())
        cash_now  = cash_avail + _planned_cash_delta()
        shortfall = intended + core_buffer * equity - cash_now
        core_pd   = price_data.get(core_ticker)
        core_px   = (float(core_pd.closes.iloc[-1])
                     if core_pd is not None and core_pd.closes is not None
                     and not core_pd.closes.empty else 0.0)
        try:
            core_mv = float(core_pos.get("market_value") or 0.0)
        except (TypeError, ValueError):
            core_mv = 0.0
        if shortfall > 0 and core_mv > 0 and core_px > 0:
            fund_val = min(shortfall, core_mv)
            fund_qty = round(fund_val / core_px, 6)
            if fund_qty >= 0.001:
                remaining_bp += fund_val
                orders.append({
                    "ticker":    core_ticker, "side": "sell", "qty": fund_qty,
                    "price_est": core_px,
                    "value_usd": fund_qty * core_px,
                    "reason": (
                        f"CORE FUNDING  sell parked {core_ticker} "
                        f"${fund_val:,.0f} to fund signal BUYs"
                    ),
                })
                if not dry_run:
                    try:
                        place_order(core_ticker, fund_qty, "sell")
                        logger.info(
                            f"Paper: CORE FUNDING sold {fund_qty}x {core_ticker}"
                            f" (${fund_val:,.0f})"
                        )
                    except Exception as e:
                        logger.warning(f"Paper: core funding sell failed — {e}")

    for r, w in zip(new_buys, alloc_w):
        ticker = r["ticker"]

        # Skip if earnings are imminent (binary event risk)
        if _near_earnings(ticker, earnings_avoid_days):
            logger.info(
                f"Paper: {ticker} BUY skipped — earnings within {earnings_avoid_days}d"
            )
            continue

        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
            logger.warning(f"Paper: {ticker} BUY skipped — no price data")
            continue

        price      = float(pd_obj.closes.iloc[-1])
        target_val = min(equity * float(w), remaining_bp)
        if target_val < price * 0.001:
            logger.info(f"Paper: {ticker} BUY skipped — insufficient buying power")
            continue
        qty = round(target_val / price, 6)
        if qty < 0.001:
            continue

        # Hard liquidity constraint: never exceed participation_pct of ADV/day
        cap = adv_cap_shares.get(ticker)
        capped = False
        if cap is not None and qty > cap:
            logger.info(
                f"Paper: {ticker} BUY capped {qty:.4f} → {cap:.4f} shares "
                f"({participation_pct*100:.0f}% ADV liquidity limit)"
            )
            qty = round(cap, 6)
            capped = True
            if qty < 0.001:
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
                + ("  [ADV-capped]" if capped else "")
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

    remaining_bp = _open_short_orders(
        results, positions, exiting, orders, price_data, equity,
        paper_cfg, remaining_bp, dry_run, core_ticker, short_held,
    )
    _core_sweep(
        core_ticker, core_buffer, cash_avail + _planned_cash_delta(),
        orders, positions, price_data, equity, dry_run,
    )
    return orders


def _open_short_orders(
    results:      list[dict],
    positions:    dict,
    exiting:      set,
    orders:       list[dict],
    price_data:   dict,
    equity:       float,
    paper_cfg:    dict,
    remaining_bp: float,
    dry_run:      bool,
    core_ticker:  str,
    short_held:   set,
) -> float:
    """
    Open whole-share shorts (Alpaca disallows fractional shorting) on robust
    SELL names not currently held, short_pos_pct each, capped at
    short_max_total_pct gross short exposure. Appends to `orders`, executes
    unless dry_run, returns updated remaining buying power. Callers only
    invoke this when risk gates allow new positions.
    """
    from brokers.alpaca import place_order

    short_pos_pct   = float((paper_cfg or {}).get("short_pos_pct", 0.04))
    short_max_total = float((paper_cfg or {}).get("short_max_total_pct", 0.0))
    if short_max_total <= 0 or short_pos_pct <= 0:
        return remaining_bp

    earnings_avoid_days = int((paper_cfg or {}).get("earnings_avoid_days", 0))

    # Gross short exposure already on the book (minus covers queued this run)
    gross_short = 0.0
    for t in short_held:
        if t in exiting:
            continue
        try:
            gross_short += abs(float(positions[t].get("market_value") or 0.0))
        except (TypeError, ValueError):
            pass

    cand = [
        r for r in results
        if r.get("robust_signal") == "SELL"
        and r["ticker"] != core_ticker
        and r["ticker"] not in positions
        and r["ticker"] not in exiting
    ]
    # Most negative IR first = highest-conviction shorts
    cand.sort(key=lambda r: r.get("ir") if r.get("ir") == r.get("ir") else 0.0)

    cap = short_max_total * equity
    for r in cand:
        ticker = r["ticker"]
        if gross_short >= cap - 1e-9:
            break
        if _near_earnings(ticker, earnings_avoid_days):
            logger.info(
                f"Paper: {ticker} SHORT skipped — earnings within {earnings_avoid_days}d"
            )
            continue
        pd_obj = price_data.get(ticker)
        if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
            continue
        price = float(pd_obj.closes.iloc[-1])
        if price <= 0:
            continue
        target = min(short_pos_pct * equity, cap - gross_short, remaining_bp)
        qty = int(target / price)
        if qty < 1:
            continue
        value = qty * price
        gross_short  += value
        remaining_bp -= value
        orders.append({
            "ticker": ticker, "side": "short", "qty": qty,
            "price_est": price, "value_usd": value,
            "reason": (
                f"SHORT signal  t={r.get('t_alpha', 0) or 0:.1f}"
                f"  IR={r.get('ir', 0) or 0:.2f}"
                f"  size={value/equity*100:.1f}%"
            ),
        })
        if not dry_run:
            try:
                place_order(ticker, qty, "sell")
                logger.info(f"Paper: SHORT {qty}x {ticker} @ ~${price:.2f}")
            except Exception as e:
                logger.warning(f"Paper: short {ticker} failed — {e}")
    return remaining_bp


def _core_sweep(
    core_ticker: str,
    core_buffer: float,
    cash_est:    float,
    orders:      list[dict],
    positions:   dict,
    price_data:  dict,
    equity:      float,
    dry_run:     bool,
) -> None:
    """
    Sweep estimated post-trade idle cash (above the buffer) into the core
    parking ETF. Appends the order to `orders` and executes unless dry_run.
    Callers only invoke this when risk gates allow buys and the market is open.
    """
    from brokers.alpaca import place_order

    if not core_ticker:
        return
    pd_obj = price_data.get(core_ticker)
    if pd_obj is None or pd_obj.closes is None or pd_obj.closes.empty:
        logger.warning(f"Paper: core sweep skipped — no price data for {core_ticker}")
        return
    price = float(pd_obj.closes.iloc[-1])
    if price <= 0:
        return

    sweep_val = cash_est - core_buffer * equity
    # Minimum ticket avoids daily dust orders as NAV drifts around the buffer
    if sweep_val < max(100.0, 0.005 * equity):
        return
    qty = round(sweep_val / price, 6)
    if qty < 0.001:
        return

    orders.append({
        "ticker":    core_ticker, "side": "buy", "qty": qty,
        "price_est": price,
        "value_usd": qty * price,
        "reason": (
            f"CORE SWEEP  idle cash ${sweep_val:,.0f} → {core_ticker}"
            f"  (buffer {core_buffer*100:.0f}%)"
        ),
    })
    if not dry_run:
        try:
            place_order(core_ticker, qty, "buy")
            logger.info(
                f"Paper: CORE SWEEP bought {qty}x {core_ticker} (${sweep_val:,.0f})"
            )
        except Exception as e:
            logger.warning(f"Paper: core sweep buy failed — {e}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_paper_trade_pipeline(config: dict) -> dict:
    """
    Orchestrates the full paper trading pipeline. Called from main.py
    when RUN_MODE=paper_trade.
    """
    from data.price_loader import load_prices
    from risk.optimizer    import fetch_ff5
    from brokers.alpaca    import is_configured, get_account, get_positions, is_market_open

    paper_cfg   = config.get("paper_trade", {})
    target_size = int(paper_cfg.get("universe_size", 50))
    max_pos_pct = float(paper_cfg.get("max_position_pct", 0.15))
    ff_days     = int(config.get("factor_model", {}).get("window_days", 252))
    dry_run     = os.environ.get("PAPER_DRY_RUN", "false").lower() == "true"

    market_open = is_market_open()
    if not market_open:
        logger.warning("Paper: US market is closed — screening will run but no orders placed")

    # Fetch VIX for the VIX gate
    vix = _fetch_vix()
    if vix is not None:
        logger.info(f"Paper: VIX = {vix:.1f}")

    logger.info(
        f"=== Paper Trade Pipeline  universe={target_size}  "
        f"max_pos={max_pos_pct*100:.0f}%  dry_run={dry_run}  "
        f"market_open={market_open}  VIX={vix} ==="
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

    # Multiple-testing deflation reflects the per-decision search space (the
    # ~50 tickers actually screened each day). The old floor of 500 (full
    # S&P 500) pushed E[max IR] to ~3.0 annualized — beyond any stock-level
    # FF5 α — so the strategy never opened a position in its first live month.
    n_strategies = max(len(all_tickers), 50)

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

            # Idempotency guard: cancel any stale unfilled day-orders from an
            # earlier run before reading positions / placing new ones. Two
            # workflows (daily_run 13:45 UTC and paper_trade 22:00 UTC) plus
            # manual dispatch can fire the same day; clearing open orders first
            # prevents duplicate/contradictory orders stacking up.
            if not dry_run and market_open:
                try:
                    from brokers.alpaca import cancel_all_orders
                    cancel_all_orders()
                    logger.info("Paper: cleared stale open orders before trading")
                except Exception as e:
                    logger.warning(f"Paper: cancel_all_orders failed — {e}")

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

            # Record daily NAV for performance tracking
            _record_nav_snapshot(account_info)

            orders = generate_orders(
                all_results, positions, equity, price_data,
                max_pos_pct=max_pos_pct, dry_run=dry_run,
                risk=risk_snapshot, paper_cfg=paper_cfg,
                buying_power=account_info.get("buying_power"),
                vix=vix, market_open=market_open,
                # Net of short market value: short-sale proceeds are collateral,
                # not sweepable cash (smv is negative or 0)
                cash=(
                    float(account.get("cash", 0) or 0)
                    + float(account.get("short_market_value", 0) or 0)
                ),
            )
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")
    else:
        logger.warning("ALPACA_API_KEY not set — screening only, no paper trades placed")
        risk_snapshot = {}

    buys   = sum(1 for o in orders if o["side"] == "buy")
    sells  = sum(1 for o in orders if o["side"] == "sell")
    shorts = sum(1 for o in orders if o["side"] == "short")
    covers = sum(1 for o in orders if o["side"] == "cover")
    logger.info(
        f"=== Paper Trade complete: {buys} buys  {sells} sells  "
        f"{shorts} shorts  {covers} covers  "
        f"{len(promoted)} promoted  {len(pruned)} pruned ==="
    )

    # ── Persist trade history and send email summary ───────────────────────────
    if orders:
        _append_trade_log(orders, account_info)

    # Load NAV history for performance section in email
    import json
    nav_history: list[dict] = []
    if _NAV_HISTORY_PATH.exists():
        try:
            nav_history = json.loads(_NAV_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    perf = _compute_performance(nav_history)

    _send_paper_summary(
        orders, account_info, risk_snapshot, config,
        perf=perf, market_open=market_open,
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
