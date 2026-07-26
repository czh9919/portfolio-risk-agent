"""
AI Sector Panel — daily deep-dive on AI leaders.

Fires at 22:45 UTC weekdays via .github/workflows/ai_panel.yml
(RUN_MODE=ai_panel). Pipeline:

  1. Load AI leader list from config/ai_leaders.yaml (~25 tickers,
     grouped into chips/cloud/models/enterprise)
  2. Batch-download 60d OHLCV via yfinance for all tickers + SPY
  3. Per-ticker .info pull for valuation ratios (fwd PE, PS, PEG,
     EV/EBITDA, market cap, margins, growth, beta)
  4. Per-ticker .insider_transactions → parse 'Text' for buy/sell
     price, filter last 30 days, exclude 0-price grants
  5. Compute AI Thermometer — five sub-scores composited to 0-100:
     breadth (25%), relative strength vs SPY (25%), momentum (20%),
     insider net-buy (15%), volume vs 20d (15%)
  6. Sonnet 4.6 writes a bilingual sector analysis
  7. Bilingual HTML email with embedded PNG charts (thermometer gauge,
     valuation scatter, insider bar) plus a long insider-trade table

Fails gracefully — each yfinance call wrapped so a per-ticker outage
never blocks the whole panel.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

AI_CONFIG_PATH = Path("config/ai_leaders.yaml")
LLM_MODEL      = "claude-sonnet-4-6"

# Regex to extract price from insider transaction "Text" field.
# Examples matched:  "Sale at price 402.84 per share."
#                    "Purchase at price 12.50 per share."
#                    "Stock Award(Grant) at price 0.00 per share."
_PRICE_RE = re.compile(r"at\s+price\s+([\d,]+\.?\d*)\s+per\s+share", re.IGNORECASE)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_ai_config(path: Path = AI_CONFIG_PATH) -> dict:
    """Return the parsed YAML. Falls back to a minimal default if missing."""
    if not path.exists():
        logger.warning(f"AIPanel: {path} missing — using minimal default list")
        return {
            "groups": {
                "chips_hardware": {
                    "label_en": "Chips & Hardware",
                    "label_zh": "芯片与硬件",
                    "tickers": ["NVDA", "AMD", "AVGO"],
                },
                "cloud_infra": {
                    "label_en": "Cloud",
                    "label_zh": "云计算",
                    "tickers": ["MSFT", "GOOGL", "AMZN"],
                },
            }
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def all_tickers(cfg: dict) -> list[str]:
    """Flat sorted list of every ticker in the AI groups."""
    seen: set[str] = set()
    for g in cfg.get("groups", {}).values():
        for tk in g.get("tickers", []):
            seen.add(tk)
    return sorted(seen)


# ── Price + valuation fetch ───────────────────────────────────────────────────

def fetch_price_metrics(tickers: list[str]) -> tuple[dict, dict]:
    """Batch 60d OHLCV, plus SPY reference for relative strength.

    Returns:
        (per_ticker_metrics, spy_metrics)
    """
    if not tickers:
        return {}, {}

    import yfinance as yf

    all_syms = list(tickers) + ["SPY"]
    end   = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=90)

    try:
        df = yf.download(all_syms, start=start, end=end,
                         progress=False, auto_adjust=False, group_by="ticker")
    except Exception as e:
        logger.warning(f"AIPanel: price batch failed — {e}")
        return {}, {}

    def _extract(tkr: str) -> dict | None:
        try:
            close = df[tkr]["Close"] if (tkr, "Close") in df.columns else df["Close"]
            vol   = df[tkr]["Volume"] if (tkr, "Volume") in df.columns else df["Volume"]
        except (KeyError, TypeError):
            return None
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if hasattr(vol, "squeeze"):
            vol = vol.squeeze()
        if hasattr(close, "dropna"):
            close = close.dropna()
        if hasattr(vol, "dropna"):
            vol = vol.dropna()

        if close is None or len(close) < 2:
            return None

        curr = float(close.iloc[-1])
        d1   = float(close.iloc[-2])
        d5   = float(close.iloc[-6])  if len(close) >= 6  else d1
        d21  = float(close.iloc[-22]) if len(close) >= 22 else d5

        window52 = close.tail(252)
        hi52 = float(window52.max()) if len(window52) else curr
        lo52 = float(window52.min()) if len(window52) else curr

        vol_today = float(vol.iloc[-1]) if len(vol) else 0.0
        vol_20    = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else vol_today
        vol_ratio = (vol_today / vol_20) if vol_20 > 0 else 1.0

        return {
            "curr":       curr,
            "chg_1d_pct": ((curr - d1)  / d1)  * 100 if d1 else 0.0,
            "chg_5d_pct": ((curr - d5)  / d5)  * 100 if d5 else 0.0,
            "chg_1m_pct": ((curr - d21) / d21) * 100 if d21 else 0.0,
            "dist_52wh":  ((curr - hi52) / hi52) * 100 if hi52 else 0.0,
            "dist_52wl":  ((curr - lo52) / lo52) * 100 if lo52 else 0.0,
            "vol_ratio":  vol_ratio,
        }

    per_ticker = {}
    for tk in tickers:
        m = _extract(tk)
        if m is not None:
            per_ticker[tk] = m
    spy_m = _extract("SPY") or {}
    return per_ticker, spy_m


def fetch_valuations(tickers: list[str]) -> dict[str, dict]:
    """Per-ticker .info valuation ratios via yfinance. Slow (1 call each).

    Returns {tkr: {market_cap, fwd_pe, ps, peg, ev_ebitda, margin, growth, beta}}
    with any missing fields as None.
    """
    if not tickers:
        return {}

    import yfinance as yf

    out: dict[str, dict] = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info or {}
        except Exception as e:
            logger.debug(f"AIPanel: {tk} .info failed — {e}")
            info = {}
        out[tk] = {
            "market_cap":     info.get("marketCap"),
            "trailing_pe":    info.get("trailingPE"),
            "fwd_pe":         info.get("forwardPE"),
            "ps":             info.get("priceToSalesTrailing12Months"),
            "pb":             info.get("priceToBook"),
            "peg":            info.get("pegRatio"),
            "ev_ebitda":      info.get("enterpriseToEbitda"),
            "profit_margin":  info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth":info.get("earningsGrowth"),
            "beta":           info.get("beta"),
        }
    return out


# ── Insider transactions ──────────────────────────────────────────────────────

def _parse_price(text: str) -> float | None:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _classify_action(text: str, shares: float, value: float) -> str:
    t = (text or "").lower()
    if "sale" in t:
        return "sell"
    if "purchase" in t or "buy" in t:
        return "buy"
    if "gift" in t:
        return "gift"
    if "grant" in t or "award" in t:
        return "grant"
    if "exercise" in t or "option" in t:
        return "option_exercise"
    return "other"


def fetch_insider_transactions(
    tickers: list[str], lookback_days: int = 30,
) -> tuple[list[dict], dict[str, dict]]:
    """Aggregate insider Form 4 activity across all AI leaders.

    Returns:
        (trades, per_ticker_summary)
        - trades: flat list of dicts, filtered to the lookback window and
                  excluding zero-price grants; each includes ticker,
                  insider name, position, action (buy/sell/gift/grant/etc),
                  shares, price, value_usd, date.
        - per_ticker_summary: {tkr: {n_trades_30d, buys_shares, sells_shares,
                                     net_shares, buyers_names, sellers_names}}
    """
    if not tickers:
        return [], {}

    import yfinance as yf

    cutoff = dt.date.today() - dt.timedelta(days=lookback_days)

    trades: list[dict] = []
    summary: dict[str, dict] = {}

    for tk in tickers:
        try:
            df = yf.Ticker(tk).insider_transactions
        except Exception as e:
            logger.debug(f"AIPanel: {tk} insider fetch failed — {e}")
            df = None

        if df is None or len(df) == 0:
            summary[tk] = {"n_trades_30d": 0, "buys_shares": 0,
                           "sells_shares": 0, "net_shares": 0}
            continue

        buys_sh = 0
        sells_sh = 0
        n_in_window = 0
        for _, row in df.iterrows():
            raw_date = row.get("Start Date")
            if raw_date is None:
                continue
            try:
                d = raw_date.date() if hasattr(raw_date, "date") else \
                    dt.date.fromisoformat(str(raw_date)[:10])
            except (ValueError, AttributeError):
                continue
            if d < cutoff:
                continue

            text   = str(row.get("Text", ""))
            shares = float(row.get("Shares", 0) or 0)
            value  = float(row.get("Value", 0) or 0)
            price  = _parse_price(text)
            action = _classify_action(text, shares, value)

            # Exclude zero-price grants/gifts from the trade list — they're
            # noise for a buy/sell signal read. Keep them in the counts.
            if action in ("buy", "sell"):
                n_in_window += 1
                if action == "buy":
                    buys_sh += shares
                else:
                    sells_sh += shares
                trades.append({
                    "ticker":    tk,
                    "insider":   str(row.get("Insider", "")).strip(),
                    "position":  str(row.get("Position", "")).strip(),
                    "action":    action,
                    "shares":    shares,
                    "price":     price,
                    "value_usd": value if value else (shares * price if price else None),
                    "date":      d.isoformat(),
                })

        summary[tk] = {
            "n_trades_30d": n_in_window,
            "buys_shares":  buys_sh,
            "sells_shares": sells_sh,
            "net_shares":   buys_sh - sells_sh,
        }

    # Newest first, then by absolute USD value
    trades.sort(key=lambda t: (t["date"], t.get("value_usd") or 0), reverse=True)
    return trades, summary


# ── AI Thermometer ────────────────────────────────────────────────────────────

def compute_thermometer(
    metrics: dict[str, dict], spy: dict, insider_summary: dict[str, dict],
) -> dict:
    """Five sub-scores combined into a 0-100 AI sector temperature.

    Component definitions (each returned as 0..1 for scale clarity):
      breadth:      pct of leaders with 1D chg > 0
      rel_strength: sector avg 1D - SPY 1D, mapped ±5% → 0..1
      momentum:     sector avg 5D return, mapped ±10% → 0..1
      insider:      pct of tickers with net-shares purchased > 0 (30d)
      volume:       sector avg volume ratio, mapped 0.5x..2x → 0..1
    Composite = weighted sum × 100.

    Also returns raw metrics for the caller to display alongside.
    """
    weights = {"breadth": 0.25, "rel_strength": 0.25,
               "momentum": 0.20, "insider": 0.15, "volume": 0.15}

    def _clip(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

    if not metrics:
        return {"composite": 50.0, "label_en": "N/A", "label_zh": "无数据",
                "components": {}, "raw": {}}

    changes_1d = [m["chg_1d_pct"] for m in metrics.values()]
    changes_5d = [m["chg_5d_pct"] for m in metrics.values()]
    vol_ratios = [m["vol_ratio"] for m in metrics.values()]

    n = len(metrics)
    pct_green = sum(1 for c in changes_1d if c > 0) / n
    avg_1d    = sum(changes_1d) / n
    avg_5d    = sum(changes_5d) / n
    avg_vol   = sum(vol_ratios) / n

    spy_1d = spy.get("chg_1d_pct", 0.0)

    # Relative strength: sector avg 1D minus SPY 1D, ±5% span → 0..1
    rs_raw = avg_1d - spy_1d
    rs_norm = _clip(0.5 + rs_raw / 10.0)

    # Momentum: avg 5D, ±10% span → 0..1
    mom_norm = _clip(0.5 + avg_5d / 20.0)

    # Volume: 20d ratio, 1.0x = neutral, 2.0x = full, 0.5x = zero
    vol_norm = _clip((avg_vol - 0.5) / 1.5)

    # Insider: pct of tickers with net_shares > 0 in last 30d
    net_positive = sum(1 for s in insider_summary.values()
                       if s.get("net_shares", 0) > 0)
    ins_norm = net_positive / max(1, len(insider_summary))

    components = {
        "breadth":      pct_green,
        "rel_strength": rs_norm,
        "momentum":     mom_norm,
        "insider":      ins_norm,
        "volume":       vol_norm,
    }
    composite = 100.0 * sum(components[k] * w for k, w in weights.items())

    if   composite >= 75: en, zh = "Overheated / Hot",  "过热"
    elif composite >= 60: en, zh = "Hot",               "偏热"
    elif composite >= 45: en, zh = "Neutral",           "中性"
    elif composite >= 30: en, zh = "Cool",              "偏冷"
    else:                  en, zh = "Cold",              "冷淡"

    return {
        "composite":  composite,
        "label_en":   en,
        "label_zh":   zh,
        "components": components,
        "raw": {
            "pct_green":     pct_green * 100,
            "avg_1d_pct":    avg_1d,
            "avg_5d_pct":    avg_5d,
            "spy_1d_pct":    spy_1d,
            "rel_strength":  rs_raw,
            "avg_vol_ratio": avg_vol,
            "pct_net_buy":   ins_norm * 100,
            "n_leaders":     n,
        },
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def _mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _to_png(fig, dpi: int = 130) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def chart_thermometer(thermo: dict) -> bytes:
    """Horizontal gauge from 0 to 100 with a marker at composite."""
    plt = _mpl_setup()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 1.6), dpi=130)
    ax.set_axis_off()

    # Gradient bar from cold blue to hot red
    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 100, 0, 1], aspect="auto",
              cmap="RdYlBu_r")

    v = thermo["composite"]
    ax.axvline(v, ymin=0, ymax=1, color="#1a1a1a", linewidth=3)
    ax.annotate(f"{v:.0f}", xy=(v, 1.05), ha="center", fontsize=16,
                fontweight="bold", color="#1a1a1a")

    # Band labels
    for pos, lbl in ((15, "Cold"), (37.5, "Cool"), (52.5, "Neutral"),
                     (67.5, "Hot"), (87.5, "Overheated")):
        ax.text(pos, -0.15, lbl, ha="center", va="top",
                fontsize=8, color="#5d6d7e")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.3, 1.4)
    ax.set_title(
        f"AI Sector Thermometer  ({thermo['label_en']})",
        fontsize=12, fontweight="bold", loc="left", pad=8,
    )
    return _to_png(fig)


def chart_valuation_scatter(
    metrics: dict, valuations: dict, groups: dict,
) -> bytes:
    """Forward PE (x) vs 1D % change (y), bubble size = market cap,
    color = group. Skips tickers missing fwd PE or price."""
    plt = _mpl_setup()
    import matplotlib.pyplot as plt

    group_colors = {
        "chips_hardware": "#e67e22",
        "cloud_infra":    "#2980b9",
        "models_apps":    "#8e44ad",
        "enterprise_ai":  "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    plotted = 0
    for gkey, ginfo in groups.items():
        xs, ys, sizes, labels = [], [], [], []
        for tk in ginfo.get("tickers", []):
            m = metrics.get(tk); v = valuations.get(tk)
            if not m or not v or v.get("fwd_pe") is None:
                continue
            if v["fwd_pe"] <= 0 or v["fwd_pe"] > 200:
                continue  # discard unhelpful outliers
            xs.append(v["fwd_pe"])
            ys.append(m["chg_1d_pct"])
            mc = v.get("market_cap") or 1e10
            sizes.append(max(40, min(1200, mc / 5e9)))
            labels.append(tk)
        if not xs:
            continue
        ax.scatter(xs, ys, s=sizes, alpha=0.55,
                   color=group_colors.get(gkey, "#7f8c8d"),
                   edgecolors="#2c3e50", linewidths=0.7,
                   label=ginfo.get("label_en", gkey))
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=8, color="#2c3e50")
            plotted += 1

    ax.axhline(0, color="#95a5a6", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Forward P/E", fontsize=10)
    ax.set_ylabel("1-Day Return %", fontsize=10)
    ax.set_title("Valuation vs 1D Move  (bubble = market cap)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if plotted:
        ax.legend(loc="best", fontsize=9)
    return _to_png(fig)


def chart_insider_bar(insider_summary: dict, top_n: int = 15) -> bytes:
    """Horizontal bar of net insider shares (buy - sell) over 30d,
    sorted by absolute magnitude."""
    plt = _mpl_setup()
    import matplotlib.pyplot as plt

    items = [(tk, s["net_shares"]) for tk, s in insider_summary.items()
             if s.get("n_trades_30d", 0) > 0]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    items = items[:top_n]
    if not items:
        return b""

    items.reverse()  # so the largest appears at top
    tickers, values = zip(*items)
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(items))), dpi=130)
    ax.barh(range(len(items)), values, color=colors, edgecolor="#2c3e50",
            linewidth=0.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(tickers, fontsize=9)
    ax.axvline(0, color="#7f8c8d", linewidth=0.5)
    ax.set_xlabel("Net Shares (Buy − Sell), last 30d", fontsize=10)
    ax.set_title("Insider Net Buying / Selling  (30-day)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_png(fig)


# ── LLM analysis ──────────────────────────────────────────────────────────────

def analyze_with_llm(
    thermo: dict, top_movers: list[dict], insider_headlines: list[dict],
    valuations: dict, api_key: str,
) -> dict:
    """Sonnet 4.6 writes bilingual analysis of AI sector state.

    Falls back to a template summary if the key is missing.
    """
    fallback = {
        "narrative_en": _fallback_en(thermo, top_movers),
        "narrative_zh": _fallback_zh(thermo, top_movers),
    }
    if not api_key:
        return fallback

    try:
        from anthropic import Anthropic
    except ImportError:
        return fallback

    mover_lines = []
    for m in top_movers[:8]:
        mover_lines.append(f"  {m['ticker']:6s} {m['chg_1d_pct']:+6.2f}% "
                           f"(5D {m['chg_5d_pct']:+6.2f}%)")

    insider_lines = []
    for h in insider_headlines[:6]:
        insider_lines.append(
            f"  {h['date']}  {h['ticker']:6s} {h['action']:5s} "
            f"{h['shares']:>10,.0f} sh @ ${h.get('price') or 0:.2f} — "
            f"{h['insider']} ({h['position']})"
        )

    val_lines = []
    for tk, v in list(valuations.items())[:8]:
        if v.get("fwd_pe") is None:
            continue
        val_lines.append(
            f"  {tk:6s} fwd P/E {v['fwd_pe']:.1f}  "
            f"PEG {v.get('peg') or '?'}  "
            f"rev-growth {(v.get('revenue_growth') or 0) * 100:+.1f}%"
        )

    prompt = (
        "You are a senior tech-sector analyst writing a daily bilingual "
        "(English + 简体中文) briefing on the AI sector for a quant "
        "portfolio manager. In 3-4 sentences per language, cover:\n"
        "  (a) sector sentiment (thermometer reading, breadth, RS vs SPY)\n"
        "  (b) any standout movers and their likely driver\n"
        "  (c) valuation stance (rich / fair / cheap on fwd P/E and PEG)\n"
        "  (d) any notable insider transaction pattern\n"
        "Do NOT recommend specific trades.\n\n"

        f"AI THERMOMETER: {thermo['composite']:.1f} "
        f"({thermo['label_en']} / {thermo['label_zh']})\n"
        f"  breadth: {thermo['raw']['pct_green']:.0f}% green  |  "
        f"sector 1D avg {thermo['raw']['avg_1d_pct']:+.2f}% "
        f"(SPY {thermo['raw']['spy_1d_pct']:+.2f}%, "
        f"RS {thermo['raw']['rel_strength']:+.2f}%)\n\n"

        "TOP MOVERS TODAY:\n" + "\n".join(mover_lines) + "\n\n"
        "VALUATION SNAPSHOT:\n" + "\n".join(val_lines) + "\n\n"
        "INSIDER HEADLINES (last 30d):\n"
        + ("\n".join(insider_lines) if insider_lines else "  (none material)") + "\n\n"

        "Return ONLY valid JSON, no fences, no prose outside:\n"
        "{\n"
        '  "narrative_en": "3-4 sentence English read",\n'
        '  "narrative_zh": "3-4 句中文分析"\n'
        "}\n"
        "IMPORTANT: do NOT use the double-quote character inside your string "
        "values — use single quotes or 「」 for emphasis."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        parsed = _safe_json(text) or fallback
        for k in ("narrative_en", "narrative_zh"):
            parsed.setdefault(k, fallback[k])
        return parsed
    except Exception as e:
        logger.warning(f"AIPanel: LLM analysis failed ({e}) — fallback")
        return fallback


def _safe_json(text: str) -> dict | None:
    """Robust extraction — direct parse, then outermost {...}, then regex."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip("` \n")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    def _grab(k):
        pat = rf'"{k}"\s*:\s*"(.*?)(?<!\\)"\s*(?:,|\}})'
        mm = re.search(pat, stripped, re.DOTALL)
        return mm.group(1).replace("\\n", " ").strip() if mm else None
    en = _grab("narrative_en"); zh = _grab("narrative_zh")
    if en or zh:
        return {"narrative_en": en or "", "narrative_zh": zh or ""}
    return None


def _fallback_en(thermo: dict, top_movers: list[dict]) -> str:
    parts = [f"AI Sector Thermometer at {thermo['composite']:.1f} "
             f"({thermo['label_en']})."]
    parts.append(f"{thermo['raw']['pct_green']:.0f}% of leaders green; "
                 f"sector avg 1D {thermo['raw']['avg_1d_pct']:+.2f}% "
                 f"vs SPY {thermo['raw']['spy_1d_pct']:+.2f}%.")
    if top_movers:
        best = top_movers[0]
        parts.append(f"Top mover: {best['ticker']} "
                     f"{best['chg_1d_pct']:+.2f}%.")
    return " ".join(parts)


def _fallback_zh(thermo: dict, top_movers: list[dict]) -> str:
    parts = [f"AI 板块温度 {thermo['composite']:.1f}"
             f"（{thermo['label_zh']}）。"]
    parts.append(f"{thermo['raw']['pct_green']:.0f}% 龙头上涨，"
                 f"板块日均 {thermo['raw']['avg_1d_pct']:+.2f}%（SPY "
                 f"{thermo['raw']['spy_1d_pct']:+.2f}%）。")
    if top_movers:
        best = top_movers[0]
        parts.append(f"领涨：{best['ticker']} "
                     f"{best['chg_1d_pct']:+.2f}%。")
    return "".join(parts)


# ── HTML rendering ────────────────────────────────────────────────────────────

def _pct_color(v: float | None) -> str:
    if v is None: return "#7f8c8d"
    if v >=  3: return "#0a8f39"
    if v >=  1: return "#28a745"
    if v > -1:  return "#7f8c8d"
    if v > -3:  return "#dc3545"
    return "#8b1a1a"


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def _fmt_mcap(v: float | None) -> str:
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


def render_html(
    thermo: dict, metrics: dict, valuations: dict, groups: dict,
    trades: list[dict], insider_summary: dict, narrative: dict,
    chart_cids: dict[str, str],
) -> str:
    today = dt.date.today().isoformat()

    # Sector heatmap rows — group by group
    sector_html = ""
    for gkey, ginfo in groups.items():
        tickers = ginfo.get("tickers", [])
        rows = ""
        for tk in tickers:
            m = metrics.get(tk)
            v = valuations.get(tk, {})
            s = insider_summary.get(tk, {})
            if not m:
                continue
            p1 = m["chg_1d_pct"]
            bg1 = _pct_color(p1)
            fg1 = "#ffffff" if p1 >= 3 or p1 <= -3 else \
                  ("#ffffff" if abs(p1) >= 1 else "#1a1a1a")
            fwd_pe = v.get("fwd_pe")
            peg    = v.get("peg")
            mcap   = _fmt_mcap(v.get("market_cap"))
            net    = s.get("net_shares", 0) or 0
            net_txt = ("↑" if net > 0 else "↓" if net < 0 else "·")
            net_col = ("#27ae60" if net > 0 else "#e74c3c" if net < 0 else "#95a5a6")

            rows += f"""
<tr>
  <td style="padding:6px 8px;font-size:12px;font-weight:bold;
             color:#2c3e50;border-top:1px solid #f0f0f0">{tk}</td>
  <td style="padding:6px 8px;background:{bg1};color:{fg1};text-align:right;
             font-size:12px;font-weight:bold;border-top:1px solid #f0f0f0">
    {_fmt_pct(p1)}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:{_pct_color(m['chg_5d_pct'])};
             border-top:1px solid #f0f0f0">{_fmt_pct(m['chg_5d_pct'])}</td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:{_pct_color(m['chg_1m_pct'])};
             border-top:1px solid #f0f0f0">{_fmt_pct(m['chg_1m_pct'])}</td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f0f0f0">
    {mcap}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f0f0f0">
    {f'{fwd_pe:.1f}' if fwd_pe else '—'}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f0f0f0">
    {f'{peg:.2f}' if peg else '—'}
  </td>
  <td style="padding:6px 8px;text-align:center;font-size:12px;
             color:{net_col};font-weight:bold;border-top:1px solid #f0f0f0">
    {net_txt}
  </td>
</tr>"""

        if not rows:
            continue
        sector_html += f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="margin:0 0 14px;border:1px solid #ecf0f1;border-radius:6px;
              overflow:hidden">
  <tr style="background:#2c3e50">
    <td colspan="8" style="padding:10px 12px;color:#fff">
      <span style="font-size:13px;font-weight:bold">{ginfo.get('label_en', gkey)}</span>
      <span style="font-size:11px;opacity:.7"> / {ginfo.get('label_zh', '')}</span>
    </td>
  </tr>
  <tr style="background:#f8f9fa">
    <th style="padding:5px 8px;text-align:left;font-size:10px;color:#95a5a6">TKR</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">1D</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">5D</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">1M</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">MCAP</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">FWD PE</th>
    <th style="padding:5px 8px;text-align:right;font-size:10px;color:#95a5a6">PEG</th>
    <th style="padding:5px 8px;text-align:center;font-size:10px;color:#95a5a6">INSIDER</th>
  </tr>
  {rows}
</table>"""

    # Insider trade table
    insider_html = ""
    if trades:
        rows_ins = ""
        for t in trades:
            side_bg = "#eafaf1" if t["action"] == "buy" else "#fdf0ef"
            side_c  = "#27ae60" if t["action"] == "buy" else "#e74c3c"
            side_zh = "买入" if t["action"] == "buy" else "卖出"
            val_str = ("—" if not t.get("value_usd")
                       else _fmt_mcap(t["value_usd"]))
            price_str = ("—" if t.get("price") is None
                         else f"${t['price']:,.2f}")
            rows_ins += f"""
<tr style="background:{side_bg}">
  <td style="padding:6px 8px;font-size:11px;color:#5d6d7e;
             border-top:1px solid #ecf0f1">{t['date']}</td>
  <td style="padding:6px 8px;font-size:12px;font-weight:bold;color:#2c3e50;
             border-top:1px solid #ecf0f1">{t['ticker']}</td>
  <td style="padding:6px 8px;font-size:11px;color:#2c3e50;
             border-top:1px solid #ecf0f1">
    {t['insider']}
    <div style="color:#95a5a6;font-size:10px">{t['position']}</div>
  </td>
  <td style="padding:6px 8px;text-align:center;
             border-top:1px solid #ecf0f1">
    <span style="background:{side_c};color:#fff;font-size:10px;
                 font-weight:bold;padding:2px 6px;border-radius:3px">
      {t['action'].upper()} / {side_zh}
    </span>
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#2c3e50;
             border-top:1px solid #ecf0f1">{t['shares']:,.0f}</td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#2c3e50;
             border-top:1px solid #ecf0f1">{price_str}</td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;
             font-weight:bold;color:#2c3e50;
             border-top:1px solid #ecf0f1">{val_str}</td>
</tr>"""
        insider_html = f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="margin:0 0 14px;border:1px solid #ecf0f1;border-radius:6px;
              overflow:hidden">
  <tr style="background:#2c3e50">
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">DATE</th>
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">TKR</th>
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">INSIDER</th>
    <th style="padding:8px;text-align:center;font-size:11px;color:#fff">SIDE</th>
    <th style="padding:8px;text-align:right;font-size:11px;color:#fff">SHARES</th>
    <th style="padding:8px;text-align:right;font-size:11px;color:#fff">PRICE</th>
    <th style="padding:8px;text-align:right;font-size:11px;color:#fff">VALUE</th>
  </tr>
  {rows_ins}
</table>"""
    else:
        insider_html = ('<p style="margin:0;font-size:12px;color:#95a5a6;'
                        'text-align:center;padding:20px">'
                        'No material insider buys or sells in the last 30 days. '
                        '/ 过去 30 天无重要内部交易。</p>')

    # Chart imgs
    def _img(cid):
        return f'<img src="cid:{cid}" style="max-width:100%;height:auto;border:1px solid #ecf0f1;border-radius:4px" alt="chart">' if cid else ""

    thermometer_img = _img(chart_cids.get("thermometer", ""))
    valuation_img   = _img(chart_cids.get("valuation", ""))
    insider_img     = _img(chart_cids.get("insider", ""))

    n_leaders = thermo["raw"].get("n_leaders", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Sector Panel</title>
</head>
<body style="margin:0;padding:0;background:#f5f6fa;
             font-family:Arial,Helvetica,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f6fa">
<tr><td align="center" style="padding:16px 8px">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:720px;background:#fff;border-radius:6px;
                overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)">

    <!-- Header -->
    <tr><td style="background:#2c3e50;padding:20px">
      <p style="margin:0;font-size:20px;font-weight:bold;color:#fff">
        AI Sector Panel &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / AI 板块检测
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; {n_leaders} AI leaders monitored / 监控 {n_leaders} 只龙头
      </p>
    </td></tr>

    <!-- LLM narrative -->
    <tr><td style="padding:16px 20px;background:#f8f9fa">
      <p style="margin:0 0 8px;font-size:14px;color:#2c3e50;line-height:1.5">
        {narrative.get('narrative_en', '')}
      </p>
      <p style="margin:0;font-size:13px;color:#5d6d7e;line-height:1.5">
        {narrative.get('narrative_zh', '')}
      </p>
    </td></tr>

    <!-- Thermometer -->
    <tr><td style="padding:16px 20px 4px;text-align:center">
      {thermometer_img}
    </td></tr>

    <!-- Sector heatmap tables -->
    <tr><td style="padding:16px 20px 0">
      <p style="margin:0 0 10px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        SECTOR HEATMAP · VALUATION / 板块+估值
      </p>
      {sector_html}
    </td></tr>

    <!-- Valuation scatter -->
    <tr><td style="padding:0 20px 4px;text-align:center">
      {valuation_img}
    </td></tr>

    <!-- Insider chart -->
    <tr><td style="padding:12px 20px 4px;text-align:center">
      {insider_img}
    </td></tr>

    <!-- Insider trade table -->
    <tr><td style="padding:12px 20px">
      <p style="margin:0 0 10px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        INSIDER TRADES · LAST 30 DAYS / 内部交易明细（近 30 天）
      </p>
      {insider_html}
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center;line-height:1.5">
        Sources: yfinance (prices, .info valuation, .insider_transactions). Analysis: Claude Sonnet 4.6.
        Insider data is aggregated from SEC Form 4 filings by yfinance. Informational only. /
        仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_ai_panel_pipeline(config: dict | None = None) -> dict:
    logger.info("=== AI Sector Panel pipeline start ===")

    cfg = load_ai_config()
    groups = cfg.get("groups", {})
    tickers = all_tickers(cfg)
    if not tickers:
        logger.error("AIPanel: no tickers configured")
        return {}
    logger.info(f"AIPanel: monitoring {len(tickers)} leaders across "
                f"{len(groups)} groups")

    metrics, spy = fetch_price_metrics(tickers)
    logger.info(f"AIPanel: prices for {len(metrics)}/{len(tickers)} tickers")

    valuations = fetch_valuations(tickers)
    logger.info(f"AIPanel: valuations for {sum(1 for v in valuations.values() if v.get('fwd_pe'))} tickers with fwd PE")

    trades, ins_summary = fetch_insider_transactions(tickers, lookback_days=30)
    logger.info(f"AIPanel: {len(trades)} insider buy/sell events in 30d "
                f"across {sum(1 for s in ins_summary.values() if s['n_trades_30d'] > 0)} tickers")

    thermo = compute_thermometer(metrics, spy, ins_summary)
    logger.info(f"AIPanel: thermometer {thermo['composite']:.1f} "
                f"({thermo['label_en']})")

    top_movers = sorted(
        [{"ticker": t, **m} for t, m in metrics.items()],
        key=lambda x: abs(x["chg_1d_pct"]), reverse=True,
    )

    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    narrative = analyze_with_llm(thermo, top_movers, trades[:6], valuations, ant_key)

    # Charts
    charts = {
        "thermometer": chart_thermometer(thermo),
        "valuation":   chart_valuation_scatter(metrics, valuations, groups),
        "insider":     chart_insider_bar(ins_summary),
    }
    chart_cids = {k: f"ai_{k}" for k, v in charts.items() if v}
    logger.info(f"AIPanel: {len(chart_cids)}/3 charts rendered, "
                f"{sum(len(v) for v in charts.values())} bytes total")

    html = render_html(thermo, metrics, valuations, groups, trades,
                       ins_summary, narrative, chart_cids)

    try:
        from notify.mailer import _smtp_send
        images = [(cid, charts[k]) for k, cid in chart_cids.items()]
        subject = (f"[AI Panel] {dt.date.today()} — "
                   f"thermometer {thermo['composite']:.0f} "
                   f"{thermo['label_en']}")
        _smtp_send(html, subject, images=images)
        logger.info("AIPanel: email sent")
    except Exception as e:
        logger.warning(f"AIPanel: email send failed — {e}")

    logger.info("=== AI Sector Panel pipeline complete ===")
    return {
        "thermometer":     thermo,
        "metrics":         metrics,
        "valuations":      valuations,
        "insider_trades":  trades,
        "insider_summary": ins_summary,
        "narrative":       narrative,
    }
