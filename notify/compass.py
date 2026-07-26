"""
Bull-Bear Compass — 6-factor market regime score.

Fires at 22:30 UTC weekdays via .github/workflows/compass.yml
(RUN_MODE=compass). Each of six market dimensions is normalised to a
score in [-1, +1]; the weighted composite is scaled to [-100, +100]
where:

  > +50   Bull
  20..50  Mildly bullish
  -20..20 Neutral
  -50..-20 Mildly bearish
  < -50   Bear

Dimensions and default weights (config/settings.yaml → compass.weights):
  Trend    (0.20) — SPY vs 200SMA plus 50-vs-200 golden/death cross
  Breadth  (0.20) — % of S&P 500 members above 200SMA
  Vol      (0.20) — VIX level + VIX9D/VIX term structure
  Credit   (0.15) — HYG (junk) vs IEF (Treasuries) 20-day divergence
  Curve    (0.10) — 10Y - 3M Treasury slope
  Momentum (0.15) — SPY 3M/6M returns + Gold/Copper safe-haven ratio

Pipeline: fetch → score → composite → persist to history →
render 60-day trend PNG → Sonnet 4.6 narrative → bilingual HTML email.

Every stage fails gracefully:
  - Missing SP500 list → breadth scored zero, other dims still computed
  - Missing ANTHROPIC_API_KEY → template narrative
  - matplotlib unavailable → HTML-only email
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

HISTORY_DIR  = Path("data/compass_history")
HISTORY_FILE = HISTORY_DIR / "history.json"

LLM_MODEL = "claude-sonnet-4-6"

DEFAULT_WEIGHTS = {
    "trend":    0.20,
    "breadth":  0.20,
    "vol":      0.20,
    "credit":   0.15,
    "curve":    0.10,
    "momentum": 0.15,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_series(df, tkr: str, field: str = "Close"):
    """Extract a 1-D Series from a group_by='ticker' yfinance DataFrame."""
    try:
        s = df[tkr][field] if (tkr, field) in df.columns else df[field]
    except (KeyError, TypeError):
        return None
    if hasattr(s, "squeeze"):
        s = s.squeeze()
    if hasattr(s, "dropna"):
        s = s.dropna()
    if s is None or len(s) == 0:
        return None
    return s


def _pct_return(series, lookback_bars: int) -> float | None:
    if series is None or len(series) < lookback_bars + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-lookback_bars - 1]) - 1.0)


# ── Individual factor scores ───────────────────────────────────────────────────

def score_trend(spy_close) -> tuple[float, dict]:
    """SPY position vs 200SMA plus golden-cross confirmation.

    Returns (score in [-1,+1], raw metrics dict).
    """
    if spy_close is None or len(spy_close) < 200:
        return 0.0, {"reason": "insufficient SPY history"}

    curr   = float(spy_close.iloc[-1])
    sma50  = float(spy_close.tail(50).mean())
    sma200 = float(spy_close.tail(200).mean())

    # Position vs 200SMA: ±10% away = full ±1
    pos_score = _clip((curr - sma200) / sma200 / 0.10)
    # 50-vs-200 crossover contributes a discrete ±1
    cross_score = 1.0 if sma50 > sma200 else -1.0

    score = 0.7 * pos_score + 0.3 * cross_score
    return _clip(score), {
        "spy":    curr,
        "sma50":  sma50,
        "sma200": sma200,
        "pct_above_200sma": (curr - sma200) / sma200 * 100,
        "golden_cross": sma50 > sma200,
    }


def score_breadth(constituent_series: dict) -> tuple[float, dict]:
    """% of S&P 500 constituents above their own 200SMA.

    Historical range in bull markets: 55–80%. In sharp bears: 20-40%.
    Score maps 25% → -1, 75% → +1.
    """
    if not constituent_series:
        return 0.0, {"reason": "no breadth data"}

    above_count = 0
    total = 0
    for tkr, s in constituent_series.items():
        if s is None or len(s) < 200:
            continue
        curr   = float(s.iloc[-1])
        sma200 = float(s.tail(200).mean())
        total += 1
        if curr > sma200:
            above_count += 1

    if total == 0:
        return 0.0, {"reason": "no valid constituents"}

    pct_above = above_count / total
    # Center at 50%, scale so 25%→-1 and 75%→+1
    score = _clip((pct_above - 0.50) * 4)
    return score, {
        "constituents_ok": total,
        "pct_above_200sma": pct_above * 100,
        "above_count": above_count,
    }


def score_volatility(vix_close, vix9d_close) -> tuple[float, dict]:
    """VIX level buckets + VIX9D/VIX term structure (backwardation penalty)."""
    if vix_close is None or len(vix_close) == 0:
        return 0.0, {"reason": "no VIX data"}

    vix = float(vix_close.iloc[-1])

    if   vix < 12: level_score =  1.0
    elif vix < 15: level_score =  0.6
    elif vix < 18: level_score =  0.2
    elif vix < 22: level_score = -0.2
    elif vix < 28: level_score = -0.6
    else:          level_score = -1.0

    term_penalty = 0.0
    term_ratio = None
    if vix9d_close is not None and len(vix9d_close) > 0:
        vix9d = float(vix9d_close.iloc[-1])
        term_ratio = vix9d / vix if vix else None
        if term_ratio is not None and term_ratio > 1.0:
            # Backwardation (front-month richer than 9-day) → risk-off signal
            term_penalty = -0.3 * _clip((term_ratio - 1.0) * 10)

    score = _clip(level_score + term_penalty)
    return score, {
        "vix": vix,
        "vix9d": float(vix9d_close.iloc[-1]) if vix9d_close is not None
                 and len(vix9d_close) > 0 else None,
        "term_structure_ratio": term_ratio,
        "backwardation": term_ratio is not None and term_ratio > 1.0,
    }


def score_credit(hyg_close, ief_close) -> tuple[float, dict]:
    """20-day divergence between HYG (junk) and IEF (Treasuries).

    HYG outperforms → spreads tightening → risk-on. IEF outperforms
    → junk sold, safe-haven bid → risk-off.
    """
    if hyg_close is None or ief_close is None:
        return 0.0, {"reason": "no credit data"}

    hyg_ret = _pct_return(hyg_close, 20)
    ief_ret = _pct_return(ief_close, 20)
    if hyg_ret is None or ief_ret is None:
        return 0.0, {"reason": "insufficient credit history"}

    divergence = hyg_ret - ief_ret
    # ±5% relative move → full ±1
    score = _clip(divergence / 0.05)
    return score, {
        "hyg_20d_pct": hyg_ret * 100,
        "ief_20d_pct": ief_ret * 100,
        "divergence_pct": divergence * 100,
    }


def score_curve(tnx_close, irx_close) -> tuple[float, dict]:
    """10Y - 3M slope in basis points. Deep inversion is bearish."""
    if tnx_close is None or irx_close is None:
        return 0.0, {"reason": "no yield data"}

    tnx = float(tnx_close.iloc[-1])
    irx = float(irx_close.iloc[-1])
    slope_bps = (tnx - irx) * 100

    # 200bps normal steep → +1; -200bps deep inversion → -1
    score = _clip(slope_bps / 200.0)
    return score, {
        "tnx_yield": tnx,
        "irx_yield": irx,
        "slope_bps": slope_bps,
        "inverted": slope_bps < 0,
    }


def score_momentum(spy_close, gold_close, copper_close) -> tuple[float, dict]:
    """SPY 3M/6M returns + Gold/Copper 20-day change (inverted).

    Gold/Copper rising = safe-haven bid = risk-off, hence sign flip.
    """
    if spy_close is None:
        return 0.0, {"reason": "no SPY data"}

    ret_63  = _pct_return(spy_close, 63)   # ~3 months of trading days
    ret_126 = _pct_return(spy_close, 126)  # ~6 months

    s_3m  = _clip((ret_63  or 0) / 0.10)   # 10% in 3M = full
    s_6m  = _clip((ret_126 or 0) / 0.20)   # 20% in 6M = full

    gc_ratio_score = 0.0
    gc_ratio_now = None
    gc_ratio_20d = None
    if gold_close is not None and copper_close is not None:
        try:
            gc_now = float(gold_close.iloc[-1]) / float(copper_close.iloc[-1])
            gc_20  = float(gold_close.iloc[-21]) / float(copper_close.iloc[-21])
            gc_change = (gc_now / gc_20) - 1.0
            # Rising Gold/Copper = risk-off, so negate; ±5% = full
            gc_ratio_score = _clip(-gc_change / 0.05)
            gc_ratio_now, gc_ratio_20d = gc_now, gc_20
        except (IndexError, ZeroDivisionError):
            pass

    score = _clip(0.4 * s_3m + 0.4 * s_6m + 0.2 * gc_ratio_score)
    return score, {
        "spy_3m_pct": (ret_63  or 0) * 100,
        "spy_6m_pct": (ret_126 or 0) * 100,
        "gold_copper_now": gc_ratio_now,
        "gold_copper_20d_ago": gc_ratio_20d,
    }


# ── Compass composite ──────────────────────────────────────────────────────────

def compute_compass(scores: dict[str, float], weights: dict[str, float]) -> dict:
    """Weighted composite scaled to [-100, +100] with a labelled regime."""
    total_w = sum(weights.values())
    if total_w <= 0:
        return {"composite": 0.0, "label_en": "Neutral", "label_zh": "中性"}

    composite = 100.0 * sum(scores.get(k, 0.0) * w
                            for k, w in weights.items()) / total_w

    if   composite >  50: en, zh = "Bull",           "强牛"
    elif composite >  20: en, zh = "Mildly Bullish", "偏牛"
    elif composite > -20: en, zh = "Neutral",        "中性"
    elif composite > -50: en, zh = "Mildly Bearish", "偏熊"
    else:                 en, zh = "Bear",           "强熊"

    return {"composite": composite, "label_en": en, "label_zh": zh}


# ── History persistence ────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Compass: history load failed — {e}; starting fresh")
        return []


def save_history(entries: list[dict], keep_days: int = 180) -> None:
    """Append + rotate, keeping the newest `keep_days` entries."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    # Dedup by date, keeping the last write per calendar date
    by_date: dict[str, dict] = {}
    for e in entries:
        by_date[e["date"]] = e
    sorted_entries = sorted(by_date.values(), key=lambda x: x["date"])
    trimmed = sorted_entries[-keep_days:]
    HISTORY_FILE.write_text(
        json.dumps(trimmed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_today(prev: list[dict], today_entry: dict,
                 keep_days: int = 180) -> list[dict]:
    """Return a new list with today's entry appended/replaced + rotated."""
    combined = list(prev) + [today_entry]
    by_date: dict[str, dict] = {}
    for e in combined:
        by_date[e["date"]] = e
    return sorted(by_date.values(), key=lambda x: x["date"])[-keep_days:]


# ── Trend chart PNG ────────────────────────────────────────────────────────────

def render_trend_chart_png(
    history: list[dict], width_in: float = 9.0, height_in: float = 3.5,
    dpi: int = 130,
) -> bytes:
    """60-day compass composite line chart with regime bands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("Compass: matplotlib unavailable — no chart")
        return b""

    if len(history) < 2:
        return b""

    tail = history[-60:]
    dates  = [dt.datetime.strptime(e["date"], "%Y-%m-%d") for e in tail]
    values = [e["composite"] for e in tail]

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")

    # Regime bands
    ax.axhspan( 50,  100, facecolor="#0a8f39", alpha=0.10)
    ax.axhspan( 20,   50, facecolor="#7dc98e", alpha=0.10)
    ax.axhspan(-20,   20, facecolor="#bfbfbf", alpha=0.10)
    ax.axhspan(-50,  -20, facecolor="#e6a5a2", alpha=0.10)
    ax.axhspan(-100, -50, facecolor="#dc3545", alpha=0.10)

    ax.plot(dates, values, color="#1a1a1a", linewidth=1.6, marker="o",
            markersize=3, markerfacecolor="#2c3e50")
    ax.axhline(0, color="#95a5a6", linewidth=0.5, linestyle="--")

    ax.set_ylim(-100, 100)
    ax.set_ylabel("Compass Score", fontsize=10)
    ax.set_title(f"Bull-Bear Compass — last {len(tail)} sessions",
                 fontsize=12, fontweight="bold", loc="left", pad=6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    fig.autofmt_xdate(rotation=0, ha="center")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.15)

    # Annotate today
    ax.annotate(f"{values[-1]:+.1f}", xy=(dates[-1], values[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", va="center",
                color=("#0a8f39" if values[-1] >= 0 else "#dc3545"))

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


# ── LLM narrative ──────────────────────────────────────────────────────────────

def narrate_with_llm(compass: dict, scores: dict, raw: dict,
                     prev_composite: float | None,
                     api_key: str) -> dict:
    """Sonnet 4.6 writes a bilingual 2-3 sentence read of today's compass.

    Falls back to a deterministic template if the key is missing or the
    call fails.
    """
    fallback = {
        "narrative_en": _fallback_narrative_en(compass, scores, prev_composite),
        "narrative_zh": _fallback_narrative_zh(compass, scores, prev_composite),
    }
    if not api_key:
        return fallback

    try:
        from anthropic import Anthropic
    except ImportError:
        return fallback

    delta_line = ""
    if prev_composite is not None:
        d = compass["composite"] - prev_composite
        delta_line = f"Change vs previous session: {d:+.1f} points.\n"

    factor_lines = []
    for k, v in scores.items():
        factor_lines.append(f"  {k:8s} score={v:+.2f}")
    raw_lines = [f"  {k}: {v}" for k, v in raw.items()]

    prompt = (
        "You are a market strategist summarising a daily Bull-Bear Compass "
        "reading for a quantitative portfolio manager. Produce a bilingual "
        "(English + 简体中文) narrative of 2-3 sentences per language. Focus "
        "on: which factor is dominant, any notable shift vs prior session, "
        "and one concrete risk to watch. Do NOT recommend specific trades.\n\n"

        f"COMPOSITE: {compass['composite']:+.1f}  "
        f"({compass['label_en']} / {compass['label_zh']})\n"
        f"{delta_line}"
        "\nFACTOR SCORES (each ∈ [-1, +1]):\n"
        + "\n".join(factor_lines) + "\n\n"
        "RAW METRICS:\n"
        + "\n".join(raw_lines) + "\n\n"

        "Return ONLY valid JSON, no fences, no prose outside:\n"
        "{\n"
        '  "narrative_en": "2-3 sentence English read of today\'s regime",\n'
        '  "narrative_zh": "2-3 句中文解读"\n'
        "}\n"
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")
        parsed = json.loads(text)
        for k in ("narrative_en", "narrative_zh"):
            parsed.setdefault(k, fallback[k])
        return parsed
    except Exception as e:
        logger.warning(f"Compass: LLM narrative failed ({e}) — falling back")
        return fallback


def _fallback_narrative_en(compass, scores, prev):
    delta = ""
    if prev is not None:
        delta = f" (change {compass['composite'] - prev:+.1f})"
    dom_k, dom_v = max(scores.items(), key=lambda kv: abs(kv[1]))
    direction = "supports" if dom_v * compass["composite"] > 0 else "opposes"
    return (f"Compass at {compass['composite']:+.1f} — {compass['label_en']}"
            f"{delta}. Dominant factor: {dom_k} ({dom_v:+.2f}) which "
            f"{direction} the current read.")


def _fallback_narrative_zh(compass, scores, prev):
    delta = ""
    if prev is not None:
        delta = f"（变化 {compass['composite'] - prev:+.1f}）"
    dom_k, dom_v = max(scores.items(), key=lambda kv: abs(kv[1]))
    zh_names = {"trend": "趋势", "breadth": "广度", "vol": "波动",
                "credit": "信用", "curve": "曲线", "momentum": "动量"}
    return (f"罗盘读数 {compass['composite']:+.1f} — {compass['label_zh']}"
            f"{delta}。主导因子：{zh_names.get(dom_k, dom_k)}（{dom_v:+.2f}）。")


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _band_color(score: float) -> tuple[str, str]:
    """Cell colour + text colour for a factor score ∈ [-1, +1]."""
    if score >=  0.5: return ("#0a8f39", "#ffffff")
    if score >=  0.2: return ("#28a745", "#ffffff")
    if score >  -0.2: return ("#bfbfbf", "#1a1a1a")
    if score > -0.5:  return ("#dc3545", "#ffffff")
    return ("#8b1a1a", "#ffffff")


def render_html(compass: dict, scores: dict, raw: dict,
                narrative: dict, chart_cid: str) -> str:
    today = dt.date.today().isoformat()

    # Composite banner colour
    banner_bg, banner_fg = _band_color(compass["composite"] / 100)

    factor_rows = ""
    zh_names = {"trend": ("Trend", "趋势"), "breadth": ("Breadth", "广度"),
                "vol": ("Volatility", "波动"), "credit": ("Credit", "信用"),
                "curve": ("Yield Curve", "利率曲线"),
                "momentum": ("Momentum", "动量")}
    for k, v in scores.items():
        bg, fg = _band_color(v)
        en, zh = zh_names.get(k, (k, k))
        factor_rows += f"""
<tr>
  <td style="padding:8px 12px;font-size:13px;font-weight:bold;
             color:#2c3e50;border-top:1px solid #f0f0f0">
    {en} <span style="color:#95a5a6;font-weight:normal">/ {zh}</span>
  </td>
  <td style="padding:8px 12px;text-align:right;background:{bg};color:{fg};
             font-size:14px;font-weight:bold;border-top:1px solid #f0f0f0;
             width:80px">
    {v:+.2f}
  </td>
</tr>"""

    # Raw metric table for the detail-oriented
    raw_rows = ""
    for k, v in raw.items():
        if v is None:
            continue
        vs = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
        raw_rows += f"""
<tr>
  <td style="padding:5px 12px;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f5f5f5">{k}</td>
  <td style="padding:5px 12px;text-align:right;font-size:11px;color:#2c3e50;
             border-top:1px solid #f5f5f5">{vs}</td>
</tr>"""

    chart_html = ""
    if chart_cid:
        chart_html = f"""
    <tr><td style="padding:12px 20px;background:#fff;text-align:center">
      <img src="cid:{chart_cid}" alt="Bull-Bear Compass Trend"
           style="max-width:100%;height:auto;border:1px solid #ecf0f1;
                  border-radius:4px">
    </td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Bull-Bear Compass</title>
</head>
<body style="margin:0;padding:0;background:#f5f6fa;
             font-family:Arial,Helvetica,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f6fa">
<tr><td align="center" style="padding:16px 8px">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:640px;background:#fff;border-radius:6px;
                overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)">

    <!-- Header -->
    <tr><td style="background:#2c3e50;padding:20px">
      <p style="margin:0;font-size:20px;font-weight:bold;color:#fff">
        Bull-Bear Compass &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / 牛熊罗盘
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; Post-close market regime / 收盘后市场状态
      </p>
    </td></tr>

    <!-- Composite banner -->
    <tr><td style="background:{banner_bg};padding:24px 20px;text-align:center">
      <p style="margin:0;font-size:14px;color:{banner_fg};opacity:.85">
        COMPOSITE / 综合评分
      </p>
      <p style="margin:6px 0 4px;font-size:42px;font-weight:bold;
                color:{banner_fg};line-height:1">
        {compass['composite']:+.1f}
      </p>
      <p style="margin:0;font-size:16px;color:{banner_fg}">
        {compass['label_en']} &nbsp;·&nbsp; {compass['label_zh']}
      </p>
    </td></tr>

    <!-- LLM narrative -->
    <tr><td style="padding:16px 20px;background:#f8f9fa">
      <p style="margin:0 0 6px;font-size:14px;color:#2c3e50">
        {narrative.get('narrative_en', '')}
      </p>
      <p style="margin:0;font-size:13px;color:#5d6d7e">
        {narrative.get('narrative_zh', '')}
      </p>
    </td></tr>

    {chart_html}

    <!-- Factor scores -->
    <tr><td style="padding:16px 20px 4px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        FACTOR SCORES / 分维度评分
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border:1px solid #ecf0f1;border-radius:4px">
        {factor_rows}
      </table>
    </td></tr>

    <!-- Raw metrics -->
    <tr><td style="padding:16px 20px 8px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        UNDERLYING METRICS / 底层指标
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border:1px solid #ecf0f1;border-radius:4px">
        {raw_rows}
      </table>
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center">
        Sources: yfinance (SPY/VIX/HYG/IEF/Treasuries/GC/HG) · Narrative:
        Claude Sonnet 4.6. Informational only. / 仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Data fetch ────────────────────────────────────────────────────────────────

_MACRO_TICKERS = ["SPY", "^VIX", "^VIX9D", "^TNX", "^IRX", "HYG", "IEF",
                  "GC=F", "HG=F"]


def _fetch_macro() -> dict:
    """Batch-download 12 months for all macro tickers, return dict of Series."""
    import yfinance as yf

    end   = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=380)

    try:
        df = yf.download(
            _MACRO_TICKERS, start=start, end=end,
            progress=False, auto_adjust=False, group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"Compass: macro batch failed — {e}")
        return {}

    return {tk: _safe_series(df, tk) for tk in _MACRO_TICKERS}


def _fetch_sp500_closes(limit: int | None = None) -> dict:
    """Fetch 12M daily closes for S&P 500 constituents (for breadth score).

    Falls back to a sample or [] if the Wikipedia scrape / batch fails.
    """
    try:
        from data.spy_universe import get_sp500_tickers
        tickers = get_sp500_tickers()
    except Exception as e:
        logger.warning(f"Compass: SP500 list unavailable — {e}")
        return {}

    if limit and len(tickers) > limit:
        tickers = tickers[:limit]

    import yfinance as yf
    end   = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=380)

    try:
        df = yf.download(
            tickers, start=start, end=end,
            progress=False, auto_adjust=False, group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"Compass: SP500 batch failed — {e}")
        return {}

    return {tk: _safe_series(df, tk) for tk in tickers}


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_compass_pipeline(config: dict | None = None) -> dict:
    logger.info("=== Bull-Bear Compass pipeline start ===")

    cfg = (config or {}).get("compass", {}) if config else {}
    weights = cfg.get("weights", DEFAULT_WEIGHTS)
    keep_days = int(cfg.get("history_days", 180))

    macro = _fetch_macro()
    logger.info(f"Compass: fetched {sum(1 for v in macro.values() if v is not None)}"
                f"/{len(_MACRO_TICKERS)} macro series")

    sp500 = _fetch_sp500_closes()
    logger.info(f"Compass: fetched {sum(1 for v in sp500.values() if v is not None)}"
                f"/{len(sp500)} SP500 series")

    trend_s,    trend_raw    = score_trend(macro.get("SPY"))
    breadth_s,  breadth_raw  = score_breadth(sp500)
    vol_s,      vol_raw      = score_volatility(macro.get("^VIX"),
                                                 macro.get("^VIX9D"))
    credit_s,   credit_raw   = score_credit(macro.get("HYG"), macro.get("IEF"))
    curve_s,    curve_raw    = score_curve(macro.get("^TNX"), macro.get("^IRX"))
    mom_s,      mom_raw      = score_momentum(macro.get("SPY"),
                                              macro.get("GC=F"),
                                              macro.get("HG=F"))

    scores = {
        "trend":    trend_s,
        "breadth":  breadth_s,
        "vol":      vol_s,
        "credit":   credit_s,
        "curve":    curve_s,
        "momentum": mom_s,
    }
    raw = {**trend_raw, **breadth_raw, **vol_raw, **credit_raw,
           **curve_raw, **mom_raw}

    compass = compute_compass(scores, weights)
    logger.info(f"Compass: composite={compass['composite']:+.1f} "
                f"({compass['label_en']})")

    # History persistence
    prev = load_history()
    prev_composite = prev[-1]["composite"] if prev else None
    today_entry = {
        "date":       dt.date.today().isoformat(),
        "composite":  compass["composite"],
        "label_en":   compass["label_en"],
        "label_zh":   compass["label_zh"],
        **{f"score_{k}": v for k, v in scores.items()},
    }
    history = append_today(prev, today_entry, keep_days=keep_days)
    save_history(history, keep_days=keep_days)

    # 60-day trend chart
    png = render_trend_chart_png(history)
    cid = "compassTrend" if png else ""

    # LLM narrative
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    narrative = narrate_with_llm(compass, scores, raw, prev_composite, ant_key)

    html = render_html(compass, scores, raw, narrative, chart_cid=cid)

    try:
        from notify.mailer import _smtp_send
        subject = (f"[Compass] {dt.date.today()} — {compass['composite']:+.1f} "
                   f"{compass['label_en']}")
        _smtp_send(html, subject, chart_bytes=png, cid=cid)
        logger.info("Compass: email sent")
    except Exception as e:
        logger.warning(f"Compass: email send failed — {e}")

    logger.info("=== Bull-Bear Compass pipeline complete ===")
    return {
        "compass":   compass,
        "scores":    scores,
        "raw":       raw,
        "narrative": narrative,
        "history":   history,
        "png_len":   len(png),
    }
