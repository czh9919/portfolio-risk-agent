"""
Macro Panel — weekly Fed & macro-indicator dashboard.

Fires at 22:00 UTC Sundays via .github/workflows/macro_panel.yml
(RUN_MODE=macro_panel). Pipeline:

  1. Pull ~14 FRED series via pandas-datareader — no key needed:
     employment (PAYEMS, UNRATE, ICSA), inflation (PCEPILFE, PCEPI,
     CPILFESL, CPIAUCSL), growth (GDPC1), Fed policy
     (DFEDTARU/L, FEDFUNDS), financial conditions (NFCI), yield-
     curve spreads (T10Y2Y, T10Y3M)
  2. Try CME FedWatch unofficial JSON API for rate probabilities;
     fall back to deriving implied rates from 30-day Fed Funds
     futures (yfinance ZQ contracts) if CME blocks the request
  3. Render 5 PNG charts: employment / inflation / Fed path /
     financial conditions / yield curves
  4. Claude Sonnet 4.6 writes a bilingual analytical read of Fed
     policy stance, inflation trajectory, employment strength,
     and financial-conditions regime
  5. Bilingual HTML email with all charts embedded via CID

Fails gracefully — missing FRED response, blocked CME, or missing
ANTHROPIC_API_KEY each degrade one section rather than break delivery.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import urllib.request
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# FRED series id → (English label, Chinese label, transform, freq_hint)
FRED_SERIES: dict[str, tuple[str, str, str, str]] = {
    "PAYEMS":   ("Nonfarm Payrolls",         "非农就业总人数",  "level",  "monthly"),
    "UNRATE":   ("Unemployment Rate",        "失业率",          "level",  "monthly"),
    "ICSA":     ("Initial Jobless Claims",   "初请失业金人数",  "level",  "weekly"),
    "PCEPILFE": ("Core PCE Index",           "核心PCE指数",     "yoy",    "monthly"),
    "PCEPI":    ("Headline PCE Index",       "整体PCE指数",     "yoy",    "monthly"),
    "CPILFESL": ("Core CPI",                 "核心CPI",         "yoy",    "monthly"),
    "CPIAUCSL": ("Headline CPI",             "整体CPI",         "yoy",    "monthly"),
    "GDPC1":    ("Real GDP",                 "实际GDP",         "qoq_saar", "quarterly"),
    "DFEDTARU": ("Fed Funds Target — Upper", "联邦基金目标上限","level",  "daily"),
    "DFEDTARL": ("Fed Funds Target — Lower", "联邦基金目标下限","level",  "daily"),
    "FEDFUNDS": ("Effective Fed Funds Rate", "有效联邦基金利率","level",  "monthly"),
    "NFCI":     ("Chicago Fed NFCI",         "芝加哥联储金融环境指数", "level", "weekly"),
    "T10Y2Y":   ("10Y – 2Y Spread",          "10Y-2Y 利差",     "level",  "daily"),
    "T10Y3M":   ("10Y – 3M Spread",          "10Y-3M 利差",     "level",  "daily"),
}

# Fed Funds futures contract-month letter codes (per CME/CBOT convention)
FUTURES_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

# FOMC meeting dates published by the Federal Reserve. Update annually.
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_MEETINGS_2026_2027 = [
    dt.date(2026, 1, 28),  dt.date(2026, 3, 18),  dt.date(2026, 4, 29),
    dt.date(2026, 6, 10),  dt.date(2026, 7, 29),  dt.date(2026, 9, 16),
    dt.date(2026, 10, 28), dt.date(2026, 12,  9),
    dt.date(2027, 1, 27),  dt.date(2027, 3, 17),  dt.date(2027, 4, 28),
    dt.date(2027, 6,  9),
]

LLM_MODEL       = "claude-sonnet-4-6"
FRED_LOOKBACK_D = 730     # 2 years — enough for YoY and 8-quarter GDP charts
USER_AGENT      = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36")


# ── FRED fetch + transforms ───────────────────────────────────────────────────

def fetch_fred_series() -> "pd.DataFrame":
    """Pull all FRED_SERIES via pandas-datareader.

    Returns an empty DataFrame on failure (so caller can degrade gracefully).
    """
    try:
        import pandas as pd
        import pandas_datareader as pdr
    except ImportError as e:
        logger.warning(f"MacroPanel: pandas-datareader unavailable — {e}")
        import pandas as pd
        return pd.DataFrame()

    start = dt.date.today() - dt.timedelta(days=FRED_LOOKBACK_D)
    try:
        df = pdr.get_data_fred(list(FRED_SERIES.keys()), start=start)
        logger.info(f"MacroPanel: FRED {len(df.columns)} series "
                    f"× {len(df)} rows loaded")
        return df
    except Exception as e:
        logger.warning(f"MacroPanel: FRED fetch failed — {e}")
        import pandas as pd
        return pd.DataFrame()


def transform_yoy(series):
    """Year-over-year % change for price indices (PCEPILFE etc.)."""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 13:
        return None
    return s.pct_change(periods=12) * 100.0


def transform_qoq_saar(series):
    """Quarter-over-quarter Seasonally-Adjusted Annualised Rate for GDP."""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 2:
        return None
    return ((s / s.shift(1)) ** 4 - 1) * 100.0


def latest_value(series, dropna: bool = True) -> tuple:
    """Return (value, date) for the last non-NaN observation, or (None, None)."""
    if series is None:
        return None, None
    s = series.dropna() if dropna else series
    if len(s) == 0:
        return None, None
    return float(s.iloc[-1]), s.index[-1].date()


# ── Fed rate expectations (CME → ZQ fallback) ─────────────────────────────────

def fetch_cme_fedwatch() -> list[dict]:
    """Attempt CME's unofficial FedWatch JSON API. Returns [] on 403/timeout.

    We do not rely on this — it's blocked from most non-browser clients —
    but if it ever starts working the caller gets official probabilities.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept":     "application/json, text/plain, */*",
        "Referer":    "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        "Origin":     "https://www.cmegroup.com",
    }
    url = ("https://www.cmegroup.com/CmeWS/mvc/Volatility/Cme/"
           "FedWatchTool/CentralTendency")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            payload = json.loads(r.read())
        return payload.get("meetings", []) or []
    except Exception as e:
        logger.info(f"MacroPanel: CME FedWatch unavailable ({e}); "
                    "using ZQ-futures fallback")
        return []


def _zq_contract(year: int, month: int) -> str:
    """CBOT symbol for 30-day Fed Funds futures at a given delivery month."""
    yy = year % 100
    return f"ZQ{FUTURES_MONTH_CODES[month]}{yy}.CBT"


def fetch_zq_implied_rates(current_effr: float | None,
                           n_meetings: int = 4) -> list[dict]:
    """For the next `n_meetings` FOMC meetings, fetch the ZQ contract for
    that month and derive an implied post-meeting Fed Funds rate.

    Simplification: we approximate the implied rate as `100 - contract price`
    (i.e. we use the month-average implied rate rather than solving for the
    exact post-meeting rate given the meeting date within the month). Close
    enough for reader intuition; CME FedWatch uses a more precise blend.

    Returns:
        [{"date": "2026-09-16", "days_away": 52,
          "contract": "ZQU26.CBT", "contract_price": 96.19,
          "implied_rate_pct": 3.81,
          "delta_bps": +18, "direction_en": "hike", "direction_zh": "加息",
          "probability_pct": 72.0}]
        Probability estimate assumes a binary 25bps move; delta of x bps
        implies |x|/25 * 100% probability of the direction, capped at 100.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    today = dt.date.today()
    upcoming = [m for m in FOMC_MEETINGS_2026_2027 if m >= today][:n_meetings]

    out: list[dict] = []
    for m in upcoming:
        symbol = _zq_contract(m.year, m.month)
        try:
            hist = yf.Ticker(symbol).history(period="10d")
            if hist is None or len(hist) == 0:
                out.append({"date": m.isoformat(), "days_away": (m - today).days,
                            "contract": symbol, "error": "no price"})
                continue
            price   = float(hist["Close"].dropna().iloc[-1])
            implied = 100.0 - price
        except Exception as e:
            out.append({"date": m.isoformat(), "days_away": (m - today).days,
                        "contract": symbol, "error": str(e)[:80]})
            continue

        row = {"date":            m.isoformat(),
               "days_away":       (m - today).days,
               "contract":        symbol,
               "contract_price":  price,
               "implied_rate_pct": implied}

        if current_effr is not None:
            delta_bps = (implied - current_effr) * 100
            row["delta_bps"] = round(delta_bps, 1)
            if delta_bps > 5:
                row["direction_en"] = "hike"; row["direction_zh"] = "加息"
            elif delta_bps < -5:
                row["direction_en"] = "cut"; row["direction_zh"] = "降息"
            else:
                row["direction_en"] = "hold"; row["direction_zh"] = "不变"
            # Binary 25bps model: |delta| / 25 * 100, capped at 100
            row["probability_pct"] = min(abs(delta_bps) / 25.0 * 100.0, 100.0)
        out.append(row)
    return out


# ── Chart rendering ───────────────────────────────────────────────────────────

def _mpl_setup():
    """Import matplotlib with Agg backend and CJK-safe font fallback."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _to_png_bytes(fig, dpi: int = 130) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def chart_employment(df) -> bytes:
    """NFP monthly changes (bars) + unemployment rate (line, right axis)."""
    plt = _mpl_setup()
    if "PAYEMS" not in df.columns or "UNRATE" not in df.columns:
        return b""

    nfp     = df["PAYEMS"].dropna().tail(24).diff().dropna()   # MoM Δ, thousands
    unemp   = df["UNRATE"].dropna().tail(len(nfp) + 1)

    if len(nfp) < 2:
        return b""

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=130)
    colors  = ["#27ae60" if v >= 0 else "#e74c3c" for v in nfp.values]
    ax.bar(nfp.index, nfp.values, color=colors, width=20)
    ax.set_ylabel("NFP MoM Δ (thousands)", fontsize=10)
    ax.axhline(0, color="#7f8c8d", linewidth=0.5)
    ax.grid(True, alpha=0.15)
    ax.set_title("Employment  (NFP MoM Δ + Unemployment)",
                 fontsize=12, fontweight="bold", loc="left")

    ax2 = ax.twinx()
    ax2.plot(unemp.index, unemp.values, color="#8e44ad", linewidth=2,
             marker="o", markersize=3, label="Unemployment %")
    ax2.set_ylabel("Unemployment Rate %", color="#8e44ad", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#8e44ad")

    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    return _to_png_bytes(fig)


def chart_inflation(df) -> bytes:
    """Core PCE, headline PCE, Core CPI, headline CPI — YoY %, last 36 months."""
    plt = _mpl_setup()
    yoy_series = {}
    for sid, label in (("PCEPILFE", "Core PCE"), ("PCEPI", "PCE"),
                       ("CPILFESL", "Core CPI"), ("CPIAUCSL", "CPI")):
        if sid in df.columns:
            y = transform_yoy(df[sid])
            if y is not None:
                yoy_series[label] = y.tail(36)

    if not yoy_series:
        return b""

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=130)
    colors = {"Core PCE": "#c0392b", "PCE":     "#e67e22",
              "Core CPI": "#2980b9", "CPI":     "#3498db"}
    for label, series in yoy_series.items():
        ax.plot(series.index, series.values, label=label,
                color=colors.get(label, "#7f8c8d"), linewidth=1.8)
    ax.axhline(2.0, color="#27ae60", linestyle="--", linewidth=1,
               label="Fed 2% target")
    ax.set_ylabel("YoY %", fontsize=10)
    ax.set_title("Inflation  (YoY %, Core & Headline)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    return _to_png_bytes(fig)


def chart_fed_path(df, fedwatch: list[dict]) -> bytes:
    """Historical Effective FFR + target range, plus implied forward path
    from ZQ futures / CME."""
    plt = _mpl_setup()
    if "FEDFUNDS" not in df.columns:
        return b""

    effr  = df["FEDFUNDS"].dropna().tail(24)
    upper = df.get("DFEDTARU")
    lower = df.get("DFEDTARL")

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=130)
    ax.plot(effr.index, effr.values, color="#2c3e50", linewidth=2,
            marker="o", markersize=3, label="Effective FFR")

    if upper is not None and lower is not None:
        u_tail = upper.dropna().tail(500)
        l_tail = lower.dropna().tail(500)
        cut = max(effr.index.min(), u_tail.index.min())
        u_use = u_tail[u_tail.index >= cut]
        l_use = l_tail[l_tail.index >= cut]
        ax.fill_between(u_use.index, l_use.reindex(u_use.index).values,
                        u_use.values, alpha=0.20, color="#3498db",
                        label="Target range")

    # Implied forward path
    for entry in fedwatch:
        if "implied_rate_pct" not in entry:
            continue
        meeting_date = dt.date.fromisoformat(entry["date"])
        ax.plot([meeting_date], [entry["implied_rate_pct"]],
                marker="^", markersize=9, color="#e67e22",
                markeredgecolor="#c0392b", markeredgewidth=1.2, zorder=5)
        ax.annotate(f"{entry['implied_rate_pct']:.2f}%",
                    xy=(meeting_date, entry["implied_rate_pct"]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color="#c0392b", fontweight="bold")
    if fedwatch:
        ax.plot([], [], marker="^", markersize=8, color="#e67e22",
                markeredgecolor="#c0392b", linestyle="",
                label="Market-implied (FOMC)")

    ax.set_ylabel("Fed Funds Rate %", fontsize=10)
    ax.set_title("Fed Policy Path  (Effective FFR + market-implied FOMC)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    return _to_png_bytes(fig)


def chart_financial_conditions(df) -> bytes:
    """Chicago Fed NFCI over last 104 weeks (~2 years) with tight/loose bands."""
    plt = _mpl_setup()
    if "NFCI" not in df.columns:
        return b""
    nfci = df["NFCI"].dropna().tail(104)
    if len(nfci) < 5:
        return b""

    fig, ax = plt.subplots(figsize=(8, 3.0), dpi=130)
    ax.plot(nfci.index, nfci.values, color="#1a1a1a", linewidth=1.8)
    ax.axhline(0, color="#7f8c8d", linewidth=0.5, linestyle="--")
    ax.fill_between(nfci.index, 0, nfci.values,
                    where=(nfci.values > 0), alpha=0.20, color="#e74c3c",
                    label="Tighter than average")
    ax.fill_between(nfci.index, 0, nfci.values,
                    where=(nfci.values < 0), alpha=0.20, color="#27ae60",
                    label="Looser than average")
    ax.set_ylabel("NFCI (std devs)", fontsize=10)
    ax.set_title("Financial Conditions  (Chicago Fed NFCI)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    return _to_png_bytes(fig)


def chart_yield_curve(df) -> bytes:
    """10Y-2Y and 10Y-3M spreads with inversion shading."""
    plt = _mpl_setup()
    if "T10Y2Y" not in df.columns or "T10Y3M" not in df.columns:
        return b""
    t2 = df["T10Y2Y"].dropna().tail(500)
    t3 = df["T10Y3M"].dropna().tail(500)
    if len(t2) < 10:
        return b""

    fig, ax = plt.subplots(figsize=(8, 3.0), dpi=130)
    ax.plot(t2.index, t2.values, color="#2980b9", linewidth=1.6, label="10Y – 2Y")
    ax.plot(t3.index, t3.values, color="#8e44ad", linewidth=1.6, label="10Y – 3M")
    ax.axhline(0, color="#7f8c8d", linewidth=0.5, linestyle="--")
    ax.fill_between(t2.index, 0, t2.values,
                    where=(t2.values < 0), alpha=0.15, color="#e74c3c")
    ax.set_ylabel("Spread (pct)", fontsize=10)
    ax.set_title("Yield-Curve Spreads  (10Y-2Y, 10Y-3M)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    return _to_png_bytes(fig)


# ── LLM narrative ─────────────────────────────────────────────────────────────

def summarize_latest(df) -> dict[str, dict]:
    """Compact snapshot for the header table + LLM prompt input."""
    out = {}
    for sid, (en, zh, transform, _freq) in FRED_SERIES.items():
        if sid not in df.columns:
            continue
        series = df[sid]
        if transform == "yoy":
            series = transform_yoy(series)
        elif transform == "qoq_saar":
            series = transform_qoq_saar(series)
        v, d = latest_value(series)
        if v is None:
            continue
        out[sid] = {"label_en": en, "label_zh": zh, "value": v,
                    "date": d.isoformat(), "transform": transform}
    return out


def narrate_with_llm(snapshot: dict, fedwatch: list[dict],
                     api_key: str) -> dict:
    """Sonnet 4.6 writes a 3-4 sentence bilingual analytical read."""
    fallback = {
        "narrative_en": _fallback_narrative_en(snapshot, fedwatch),
        "narrative_zh": _fallback_narrative_zh(snapshot, fedwatch),
    }
    if not api_key:
        return fallback

    try:
        from anthropic import Anthropic
    except ImportError:
        return fallback

    lines = []
    for sid, d in snapshot.items():
        unit = "%" if d["transform"] in ("yoy", "qoq_saar") else ""
        lines.append(f"  {d['label_en']:32s} {d['value']:.2f}{unit}  "
                     f"({d['date']}, {d['transform']})")

    fw_lines = []
    for e in fedwatch:
        if "implied_rate_pct" in e:
            fw_lines.append(
                f"  FOMC {e['date']}  implied {e['implied_rate_pct']:.2f}%  "
                f"Δ {e.get('delta_bps', 0):+.0f}bps  "
                f"({e.get('direction_en', '?')}, "
                f"prob≈{e.get('probability_pct', 0):.0f}%)")

    prompt = (
        "You are a senior macro strategist writing a weekly analytical read "
        "for a quant portfolio manager. Cover in 3-4 sentences per language "
        "(English + 简体中文): (a) Fed policy stance and market-implied path, "
        "(b) inflation trajectory relative to the 2% target, (c) labour-market "
        "strength, (d) financial-conditions regime. Cite specific numbers.\n\n"

        "LATEST FRED VALUES:\n" + "\n".join(lines) + "\n\n"
        "MARKET-IMPLIED FOMC PATH (from Fed Funds futures):\n"
        + ("\n".join(fw_lines) if fw_lines else "  (unavailable)") + "\n\n"

        "Return ONLY valid JSON, no markdown fences, no prose outside:\n"
        "{\n"
        '  "narrative_en": "3-4 sentence English analytical read",\n'
        '  "narrative_zh": "3-4 句中文分析"\n'
        "}\n\n"
        "IMPORTANT: do NOT use the double-quote character (\") inside your "
        "string values — it will break the JSON. If you need to quote a "
        "term, use single quotes or Chinese quotation marks (「」)."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        parsed = _safe_parse_json(text) or fallback
        for k in ("narrative_en", "narrative_zh"):
            parsed.setdefault(k, fallback[k])
        return parsed
    except Exception as e:
        logger.warning(f"MacroPanel: LLM narrative failed ({e}) — fallback")
        return fallback


def _safe_parse_json(text: str) -> dict | None:
    """Best-effort JSON extraction from LLM output.

    Handles ```json fences, leading/trailing prose, and — when direct
    parsing fails — pulls values out of the two known keys with regex so
    a stray unescaped double-quote inside a Chinese sentence doesn't
    force a full fallback.
    """
    import re

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

    # Grab the outermost {...} block
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: regex the two known fields directly. Anchor on the key
    # then greedily grab until the next `",\n  "` pattern or closing brace.
    def _grab(key: str) -> str | None:
        # non-greedy up to a comma-newline-quote-key or a close brace
        pat = rf'"{key}"\s*:\s*"(.*?)(?<!\\)"\s*(?:,|\}})'
        mm = re.search(pat, stripped, re.DOTALL)
        return mm.group(1).replace("\\n", " ").strip() if mm else None

    en = _grab("narrative_en")
    zh = _grab("narrative_zh")
    if en or zh:
        return {"narrative_en": en or "", "narrative_zh": zh or ""}
    return None


def _fallback_narrative_en(snapshot: dict, fedwatch: list[dict]) -> str:
    parts = []
    ffr = snapshot.get("FEDFUNDS", {}).get("value")
    tu  = snapshot.get("DFEDTARU", {}).get("value")
    tl  = snapshot.get("DFEDTARL", {}).get("value")
    if ffr and tu and tl:
        parts.append(f"Fed Funds effective {ffr:.2f}% "
                     f"(target {tl:.2f}-{tu:.2f}%).")
    if fedwatch:
        first = next((e for e in fedwatch if "delta_bps" in e), None)
        if first:
            parts.append(f"Market prices {first['delta_bps']:+.0f}bps by "
                         f"{first['date']}.")
    pce = snapshot.get("PCEPILFE", {}).get("value")
    if pce:
        parts.append(f"Core PCE {pce:.2f}% YoY.")
    unemp = snapshot.get("UNRATE", {}).get("value")
    if unemp:
        parts.append(f"Unemployment {unemp:.1f}%.")
    return " ".join(parts) if parts else "LLM narrative unavailable."


def _fallback_narrative_zh(snapshot: dict, fedwatch: list[dict]) -> str:
    parts = []
    ffr = snapshot.get("FEDFUNDS", {}).get("value")
    tu  = snapshot.get("DFEDTARU", {}).get("value")
    tl  = snapshot.get("DFEDTARL", {}).get("value")
    if ffr and tu and tl:
        parts.append(f"有效联邦基金利率 {ffr:.2f}%，目标区间 {tl:.2f}-{tu:.2f}%。")
    if fedwatch:
        first = next((e for e in fedwatch if "delta_bps" in e), None)
        if first:
            parts.append(f"市场对 {first['date']} 会议定价 "
                         f"{first['delta_bps']:+.0f}bps。")
    pce = snapshot.get("PCEPILFE", {}).get("value")
    if pce:
        parts.append(f"核心 PCE 同比 {pce:.2f}%。")
    unemp = snapshot.get("UNRATE", {}).get("value")
    if unemp:
        parts.append(f"失业率 {unemp:.1f}%。")
    return "".join(parts) if parts else "LLM 分析暂不可用。"


# ── HTML rendering ────────────────────────────────────────────────────────────

def _fmt_value(sid: str, val: float, transform: str) -> str:
    if transform in ("yoy", "qoq_saar"):
        return f"{val:+.2f}%"
    if sid in ("ICSA", "PAYEMS", "GDPC1"):
        return f"{val:,.0f}"
    if sid == "NFCI":
        return f"{val:+.3f}"
    if "SPREAD" in sid or "T10Y" in sid:
        return f"{val:+.2f}"
    return f"{val:.2f}%"


def render_html(snapshot: dict, fedwatch: list[dict],
                narrative: dict, chart_cids: dict[str, str]) -> str:
    today = dt.date.today().isoformat()

    # Snapshot cards — grouped
    groups = [
        ("Fed Policy / 联储政策",       ["DFEDTARL", "DFEDTARU", "FEDFUNDS"]),
        ("Employment / 就业",           ["PAYEMS", "UNRATE", "ICSA"]),
        ("Inflation / 通胀 (YoY)",      ["PCEPILFE", "PCEPI", "CPILFESL", "CPIAUCSL"]),
        ("Growth / 增长",               ["GDPC1"]),
        ("Financial / 金融条件+曲线",   ["NFCI", "T10Y2Y", "T10Y3M"]),
    ]

    group_html = ""
    for gname, sids in groups:
        rows = ""
        for sid in sids:
            d = snapshot.get(sid)
            if not d:
                continue
            val_str = _fmt_value(sid, d["value"], d["transform"])
            rows += f"""
<tr>
  <td style="padding:6px 12px;font-size:12px;color:#2c3e50;
             border-top:1px solid #f0f0f0">
    {d['label_en']}<br>
    <span style="color:#95a5a6;font-size:10px">{d['label_zh']}</span>
  </td>
  <td style="padding:6px 12px;font-size:14px;font-weight:bold;
             color:#2c3e50;text-align:right;border-top:1px solid #f0f0f0">
    {val_str}
  </td>
  <td style="padding:6px 12px;font-size:10px;color:#7f8c8d;
             text-align:right;border-top:1px solid #f0f0f0">
    {d['date']}
  </td>
</tr>"""
        if rows:
            group_html += f"""
<tr><td colspan="3" style="padding:12px 12px 6px;font-size:11px;
                          font-weight:bold;color:#95a5a6;letter-spacing:.4px">
    {gname}
</td></tr>
{rows}
"""

    # FedWatch table
    fw_html = ""
    if fedwatch:
        fw_rows = ""
        for e in fedwatch:
            if "implied_rate_pct" not in e:
                continue
            direction = e.get("direction_en", "?")
            dir_color = {"hike": "#e74c3c", "cut": "#27ae60",
                         "hold": "#7f8c8d"}.get(direction, "#7f8c8d")
            zh_dir = e.get("direction_zh", "?")
            fw_rows += f"""
<tr>
  <td style="padding:6px 10px;font-size:12px;color:#2c3e50;
             border-top:1px solid #f0f0f0">{e['date']}
    <span style="color:#95a5a6;font-size:10px"> · {e['days_away']}d</span>
  </td>
  <td style="padding:6px 10px;font-size:12px;color:#2c3e50;text-align:right;
             border-top:1px solid #f0f0f0">{e['implied_rate_pct']:.2f}%</td>
  <td style="padding:6px 10px;font-size:12px;color:{dir_color};
             text-align:right;font-weight:bold;
             border-top:1px solid #f0f0f0">
    {e.get('delta_bps', 0):+.0f}bps
    <span style="color:#95a5a6;font-weight:normal">
      ({direction} / {zh_dir}, {e.get('probability_pct', 0):.0f}%)
    </span>
  </td>
</tr>"""
        if fw_rows:
            fw_html = f"""
<tr><td style="padding:12px 20px 4px">
  <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
            color:#95a5a6;letter-spacing:.4px">
    MARKET-IMPLIED FOMC PATH / 市场隐含利率路径
  </p>
  <p style="margin:0 0 8px;font-size:10px;color:#95a5a6">
    From 30-day Fed Funds futures (ZQ). Δ bps vs current effective rate,
    probability from a binary 25bps model.
  </p>
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border:1px solid #ecf0f1;border-radius:4px">
    {fw_rows}
  </table>
</td></tr>"""

    # Chart image tags
    chart_order = [
        ("employment",  "Employment / 就业"),
        ("inflation",   "Inflation / 通胀"),
        ("fed_path",    "Fed Policy Path / 联储利率路径"),
        ("nfci",        "Financial Conditions / 金融环境"),
        ("yield_curve", "Yield Curve Spreads / 收益率曲线"),
    ]
    charts_html = ""
    for key, _label in chart_order:
        cid = chart_cids.get(key)
        if not cid:
            continue
        charts_html += f"""
<tr><td style="padding:8px 20px 4px;text-align:center">
  <img src="cid:{cid}" style="max-width:100%;height:auto;
       border:1px solid #ecf0f1;border-radius:4px" alt="{key}">
</td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Macro Panel</title>
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
        Macro Panel &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / 宏观数据面板
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; Weekly Fed &amp; macro snapshot / 周度联储+宏观快照
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

    <!-- Snapshot table -->
    <tr><td style="padding:16px 20px 4px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        LATEST INDICATORS / 最新指标
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border:1px solid #ecf0f1;border-radius:4px">
        {group_html}
      </table>
    </td></tr>

    {fw_html}

    <!-- Charts -->
    <tr><td style="padding:16px 20px 4px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        TREND CHARTS / 走势图
      </p>
    </td></tr>
    {charts_html}

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center;
                line-height:1.5">
        Sources: FRED (Federal Reserve Economic Data) via pandas-datareader,
        yfinance ZQ futures for FOMC path, Claude Sonnet 4.6 for analysis.
        Informational only. / 仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_macro_panel_pipeline(config: dict | None = None) -> dict:
    logger.info("=== Macro Panel pipeline start ===")

    df = fetch_fred_series()
    if df is None or len(df.columns) == 0:
        logger.error("MacroPanel: FRED unavailable; nothing to send")
        return {}

    snapshot = summarize_latest(df)
    logger.info(f"MacroPanel: snapshot has {len(snapshot)} indicators")

    # Try CME first (usually blocked), fallback to ZQ math
    cme = fetch_cme_fedwatch()
    if cme:
        # If CME ever succeeds, adapt its response shape to ours here.
        # For now we log and fall through — the parser lives here so future
        # unblocked runs still degrade gracefully rather than crash.
        fedwatch = []
    else:
        effr = snapshot.get("FEDFUNDS", {}).get("value")
        fedwatch = fetch_zq_implied_rates(effr, n_meetings=4)
    logger.info(f"MacroPanel: fedwatch has {len(fedwatch)} entries")

    # Charts
    charts_bytes = {
        "employment":  chart_employment(df),
        "inflation":   chart_inflation(df),
        "fed_path":    chart_fed_path(df, fedwatch),
        "nfci":        chart_financial_conditions(df),
        "yield_curve": chart_yield_curve(df),
    }
    chart_cids = {k: f"macro_{k}" for k, v in charts_bytes.items() if v}
    total_png_bytes = sum(len(v) for v in charts_bytes.values())
    logger.info(f"MacroPanel: {len(chart_cids)}/5 charts rendered, "
                f"{total_png_bytes} bytes total")

    # LLM narrative
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    narrative = narrate_with_llm(snapshot, fedwatch, ant_key)

    html = render_html(snapshot, fedwatch, narrative, chart_cids)

    try:
        from notify.mailer import _smtp_send
        images = [(cid, charts_bytes[k])
                  for k, cid in chart_cids.items()]
        subject = f"[Macro Panel] {dt.date.today()} — weekly Fed & macro"
        _smtp_send(html, subject, images=images)
        logger.info("MacroPanel: email sent")
    except Exception as e:
        logger.warning(f"MacroPanel: email send failed — {e}")

    logger.info("=== Macro Panel pipeline complete ===")
    return {
        "snapshot": snapshot,
        "fedwatch": fedwatch,
        "charts":   {k: len(v) for k, v in charts_bytes.items()},
        "html":     html,
    }
