"""
Market Brief — daily pre-market macro/news snapshot.

Fires at 12:00 UTC weekdays via .github/workflows/market_brief.yml
(RUN_MODE=market_brief). Pipeline:

  1. Fetch Treasury yields, DXY, VIX, SPY/QQQ via yfinance
  2. Compute regime (SPY vs SMA20/50/200, VIX bucket, yield-curve slope)
  3. Pull raw headlines from ~8 free RSS feeds (CNBC / WSJ Markets /
     MarketWatch / Yahoo Finance / Investing.com / SeekingAlpha / FT)
  4. Ask Claude Sonnet 4.6 to curate 6-8 most macro-relevant headlines
     and produce a bilingual summary + 3 takeaways
  5. Render bilingual HTML and send via SMTP

Fails gracefully:
  - Each RSS feed try/except independently; partial outages don't block
  - Missing ANTHROPIC_API_KEY → LLM curation skipped, first 8 raw headlines shown
  - yfinance failures per-ticker → that row omitted, brief still sent
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time as _time
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# yfinance ticker → (English label, Chinese label, format hint)
MACRO_TICKERS: dict[str, tuple[str, str, str]] = {
    "^IRX":     ("3-Month T-Bill",  "3个月国债",  "yield"),
    "^TNX":     ("10-Year Treasury","10年国债",   "yield"),
    "^TYX":     ("30-Year Treasury","30年国债",   "yield"),
    "DX-Y.NYB": ("US Dollar Index", "美元指数",   "index"),
    "^VIX":     ("VIX",             "波动率指数", "index"),
    "SPY":      ("S&P 500",         "标普500",    "price"),
    "QQQ":      ("Nasdaq 100",      "纳指100",    "price"),
}

# Free RSS feeds. Probed 2026-07 — all return valid <item> lists. Ordered by
# quality/signal density; each feed is fetched independently so partial outages
# never block the brief.
RSS_FEEDS: dict[str, str] = {
    "CNBC Top News":  "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC Economy":   "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "WSJ Markets":    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "MarketWatch":    "https://www.marketwatch.com/rss/marketpulse",
    "Yahoo Finance":  "https://finance.yahoo.com/news/rssindex",
    "Investing.com":  "https://www.investing.com/rss/news_25.rss",
    "Seeking Alpha":  "https://seekingalpha.com/market_currents.xml",
    "Financial Times":"https://www.ft.com/rss/home",
}

LLM_MODEL       = "claude-sonnet-4-6"
LLM_USER_AGENT  = "Mozilla/5.0 (compatible; StockAI-Brief/1.0)"


# ── Macro data ─────────────────────────────────────────────────────────────────

def fetch_macro_data() -> dict[str, dict]:
    """Batch-download 3 months of daily closes for macro tickers.

    Returns a dict keyed by ticker: {label_en, label_zh, fmt, curr, chg_1d_pct,
    chg_1w_pct, chg_1m_pct}. Tickers that fail silently drop out.
    """
    import yfinance as yf

    tickers = list(MACRO_TICKERS.keys())
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=100)

    try:
        df = yf.download(
            tickers, start=start, end=end,
            progress=False, auto_adjust=False, group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"MacroBrief: yfinance batch failed — {e}")
        return {}

    out: dict[str, dict] = {}
    for tkr in tickers:
        try:
            series = df[tkr]["Close"] if (tkr, "Close") in df.columns else df["Close"]
        except (KeyError, TypeError):
            continue
        if hasattr(series, "dropna"):
            series = series.dropna()
        if series is None or len(series) < 2:
            continue

        curr = float(series.iloc[-1])
        d1   = float(series.iloc[-2])
        d5   = float(series.iloc[-6])  if len(series) >= 6  else d1
        d21  = float(series.iloc[-22]) if len(series) >= 22 else d5
        label_en, label_zh, fmt = MACRO_TICKERS[tkr]
        out[tkr] = {
            "label_en":    label_en,
            "label_zh":    label_zh,
            "fmt":         fmt,
            "curr":        curr,
            "chg_1d_pct":  ((curr - d1)  / d1)  * 100 if d1  else 0.0,
            "chg_1w_pct":  ((curr - d5)  / d5)  * 100 if d5  else 0.0,
            "chg_1m_pct":  ((curr - d21) / d21) * 100 if d21 else 0.0,
        }
    return out


def compute_market_status(macro: dict[str, dict]) -> dict:
    """Derive regime flags from macro data.

    - VIX bucket:  <15 calm / 15-20 normal / 20-25 elevated / >25 stressed
    - Yield curve: 10Y - 3M in basis points; inverted iff negative
    - SPY trend:   above or below its 20/50/200-day SMA (needs longer history)
    """
    status: dict = {}

    vix = macro.get("^VIX", {}).get("curr")
    if vix is not None:
        if vix < 15:
            status["vix_regime"] = ("calm", "低波动")
        elif vix < 20:
            status["vix_regime"] = ("normal", "正常")
        elif vix < 25:
            status["vix_regime"] = ("elevated", "偏高")
        else:
            status["vix_regime"] = ("stressed", "承压")
        status["vix_level"] = float(vix)

    tnx = macro.get("^TNX", {}).get("curr")
    irx = macro.get("^IRX", {}).get("curr")
    if tnx is not None and irx is not None:
        slope_bps = (tnx - irx) * 100
        status["curve_10y_3m_bps"] = float(slope_bps)
        status["curve_inverted"]   = bool(slope_bps < 0)

    dxy = macro.get("DX-Y.NYB", {}).get("chg_1m_pct")
    if dxy is not None:
        if dxy > 1.5:
            status["usd_trend"] = ("strengthening", "走强")
        elif dxy < -1.5:
            status["usd_trend"] = ("weakening",     "走弱")
        else:
            status["usd_trend"] = ("range-bound",   "震荡")

    status.update(_spy_trend_status())
    return status


def _spy_trend_status() -> dict:
    """Fetch 320d of SPY closes and place today's price vs its 20/50/200 SMAs."""
    try:
        import yfinance as yf
        end   = dt.date.today() + dt.timedelta(days=1)
        start = end - dt.timedelta(days=320)
        raw   = yf.download("SPY", start=start, end=end,
                            progress=False, auto_adjust=False)
        # yfinance returns a DataFrame with a "Close" column; for single-ticker
        # downloads newer versions still return a DataFrame — squeeze to 1D.
        hist = raw["Close"]
        if hasattr(hist, "squeeze"):
            hist = hist.squeeze()
        if hasattr(hist, "dropna"):
            hist = hist.dropna()
        if hist is None or len(hist) < 200:
            return {}
        curr = float(hist.iloc[-1])
        sma20  = float(hist.tail(20).mean())
        sma50  = float(hist.tail(50).mean())
        sma200 = float(hist.tail(200).mean())
        return {
            "spy_curr":   curr,
            "spy_sma20":  sma20,
            "spy_sma50":  sma50,
            "spy_sma200": sma200,
            "spy_above_20":  curr > sma20,
            "spy_above_50":  curr > sma50,
            "spy_above_200": curr > sma200,
        }
    except Exception as e:
        logger.warning(f"MacroBrief: SPY trend fetch failed — {e}")
        return {}


# ── News fetch (RSS) ───────────────────────────────────────────────────────────

def fetch_news_rss(
    feeds: dict[str, str] | None = None,
    lookback_hours: int = 24,
    max_per_source: int = 15,
) -> list[dict]:
    """Pull headlines from every configured RSS feed. Each feed failure is
    logged and skipped; a totally-empty result is still returned safely.

    Returns items sorted newest-first, deduplicated by (case-insensitive) title.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("MacroBrief: feedparser not installed — no news")
        return []

    feeds = feeds if feeds is not None else RSS_FEEDS
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=lookback_hours)
    items: list[dict] = []

    for source_name, url in feeds.items():
        try:
            # feedparser.parse accepts an `agent` param that sets User-Agent
            d = feedparser.parse(url, agent=LLM_USER_AGENT)
        except Exception as e:
            logger.warning(f"MacroBrief: RSS {source_name} failed — {e}")
            continue

        n = 0
        for entry in d.entries:
            if n >= max_per_source:
                break

            pub_struct = (entry.get("published_parsed")
                          or entry.get("updated_parsed"))
            if not pub_struct:
                continue
            try:
                # published_parsed is time.struct_time in UTC per RSS/Atom spec
                pub_dt = dt.datetime.utcfromtimestamp(_time.mktime(pub_struct))
            except (TypeError, ValueError, OverflowError):
                continue
            if pub_dt < cutoff:
                continue

            title = (entry.get("title") or "").strip()
            if not title:
                continue

            items.append({
                "title":     title,
                "url":       entry.get("link", ""),
                "source":    source_name,
                "published": pub_dt.isoformat(),
                "summary":   (entry.get("summary") or "")[:400],
                "_ts":       pub_dt,   # sort key, stripped before return
            })
            n += 1

    # Dedup by lowered title prefix (RSS repeats across sibling feeds)
    seen: set[str] = set()
    unique: list[dict] = []
    for it in sorted(items, key=lambda x: x["_ts"], reverse=True):
        key = it["title"].lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append({k: v for k, v in it.items() if k != "_ts"})

    logger.info(f"MacroBrief: RSS pulled {len(items)} items → "
                f"{len(unique)} after dedup")
    return unique


# ── LLM curation + summary ─────────────────────────────────────────────────────

def summarize_news_with_llm(
    articles: list[dict], macro: dict, status: dict, api_key: str,
) -> dict:
    """Ask Claude Sonnet 4.6 to (a) pick the 6-8 most macro-relevant headlines
    from the raw RSS pool and (b) produce a bilingual summary + 3 takeaways.

    Returns:
        {"picked_articles": list[dict],   # up to 8 items from `articles`
         "summary_en": str, "summary_zh": str,
         "takeaways_en": list[str], "takeaways_zh": list[str]}

    Falls back to first 8 raw articles + template summary if the key is missing
    or the call fails, so the brief is always deliverable.
    """
    fallback = {
        "picked_articles": articles[:8],
        "summary_en":      _fallback_summary_en(articles, status),
        "summary_zh":      _fallback_summary_zh(articles, status),
        "takeaways_en":    [a["title"] for a in articles[:3]],
        "takeaways_zh":    [a["title"] for a in articles[:3]],
    }
    if not api_key or not articles:
        return fallback

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("MacroBrief: anthropic SDK not installed — falling back")
        return fallback

    macro_lines = []
    for tkr, row in macro.items():
        macro_lines.append(
            f"{row['label_en']}: {row['curr']:.2f} "
            f"(1d {row['chg_1d_pct']:+.2f}%, 1w {row['chg_1w_pct']:+.2f}%, "
            f"1m {row['chg_1m_pct']:+.2f}%)"
        )

    numbered = []
    for i, a in enumerate(articles):
        numbered.append(f"{i}. [{a['source']}] {a['title']}")

    curve = (f"10Y-3M spread {status.get('curve_10y_3m_bps', 0):.0f}bps "
             f"({'inverted' if status.get('curve_inverted') else 'normal'})")
    vix   = status.get("vix_regime", ("?", "?"))[0]

    prompt = (
        "You are a senior market analyst writing a pre-market brief for a "
        "quantitative portfolio manager. From the raw headline pool below, "
        "select the 6-8 items MOST relevant to a global-macro/US-equity view "
        "and produce a bilingual (English + 简体中文) summary.\n\n"

        "Selection criteria — INCLUDE:\n"
        "  • Fed / ECB / BoJ / PBoC policy signals, inflation, employment, GDP\n"
        "  • Geopolitics with market implications (energy, sanctions, war)\n"
        "  • Mega-cap or bellwether earnings (AAPL/MSFT/NVDA/GOOG/META/AMZN/TSLA)\n"
        "  • Major M&A, IPOs, or index-level moves\n"
        "  • US Treasury / USD / VIX-relevant events\n"
        "Selection criteria — EXCLUDE:\n"
        "  • Single-stock analyst upgrade/downgrade of small caps\n"
        "  • Individual insider trades (\"X sold 1000 shares of Y\")\n"
        "  • Pure lifestyle / opinion / crypto-shilling pieces\n"
        "  • Duplicates already covered by a stronger headline\n\n"

        f"MACRO SNAPSHOT:\n" + "\n".join(macro_lines) + "\n"
        f"REGIME: VIX {vix}, {curve}\n\n"
        f"RAW HEADLINES ({len(articles)} items, numbered):\n"
        + "\n".join(numbered) + "\n\n"

        "Return ONLY valid JSON (no markdown fences, no prose), with exactly:\n"
        "{\n"
        '  "picked_indices": [<6-8 integers from the numbered list above>],\n'
        '  "summary_en":     "2-3 sentence market outlook in English",\n'
        '  "summary_zh":     "2-3 句中文市场展望",\n'
        '  "takeaways_en":   ["takeaway 1", "takeaway 2", "takeaway 3"],\n'
        '  "takeaways_zh":   ["要点1", "要点2", "要点3"]\n'
        "}\n"
        "The picked_indices should be ordered by descending importance. "
        "Do NOT recommend specific trades."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        # Strip accidental ```json … ``` fences
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")
        parsed = json.loads(text)

        # Map picked_indices → actual article dicts (guard against out-of-range)
        idxs = parsed.get("picked_indices", []) or []
        picked = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(articles):
                picked.append(articles[i])
        parsed["picked_articles"] = picked[:8] if picked else articles[:8]

        for k in ("summary_en", "summary_zh", "takeaways_en", "takeaways_zh"):
            parsed.setdefault(k, fallback[k])
        logger.info(f"MacroBrief: LLM picked {len(parsed['picked_articles'])} "
                    f"headlines from pool of {len(articles)}")
        return parsed
    except Exception as e:
        logger.warning(f"MacroBrief: LLM summary failed ({e}) — falling back")
        return fallback


def _fallback_summary_en(articles: list[dict], status: dict) -> str:
    vix = status.get("vix_regime", ("normal", "正常"))[0]
    curve = ("inverted" if status.get("curve_inverted") else "normal")
    return (f"Pre-market snapshot: VIX regime {vix}, yield curve {curve}. "
            f"{len(articles)} raw headlines gathered — LLM curation unavailable.")


def _fallback_summary_zh(articles: list[dict], status: dict) -> str:
    vix = status.get("vix_regime", ("normal", "正常"))[1]
    curve = ("倒挂" if status.get("curve_inverted") else "正常")
    return (f"盘前快照：VIX 处于{vix}状态，收益率曲线{curve}。"
            f"抓取 {len(articles)} 条原始新闻，LLM 摘要暂不可用。")


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _fmt_value(row: dict) -> str:
    fmt, v = row["fmt"], row["curr"]
    if fmt == "yield":
        return f"{v:.2f}%"
    if fmt == "index":
        return f"{v:.2f}"
    return f"${v:,.2f}"


def _fmt_chg(pct: float) -> str:
    color = "#27ae60" if pct >= 0 else "#e74c3c"
    sign  = "+" if pct >= 0 else ""
    return (f'<span style="color:{color};font-weight:bold">'
            f'{sign}{pct:.2f}%</span>')


def render_brief_html(
    macro: dict, status: dict, articles: list[dict], summary: dict,
) -> str:
    """Render the bilingual pre-market brief email.

    `articles` here should be the CURATED list (summary["picked_articles"]),
    not the raw RSS pool.
    """
    today = dt.date.today().isoformat()

    macro_rows = ""
    for tkr, row in macro.items():
        macro_rows += f"""
<tr>
  <td style="padding:8px 12px;font-size:13px;border-top:1px solid #eee">
    <span style="color:#2c3e50;font-weight:bold">{row['label_en']}</span><br>
    <span style="color:#aaa;font-size:11px">{row['label_zh']}</span>
  </td>
  <td style="padding:8px 12px;font-size:14px;text-align:right;
             border-top:1px solid #eee;color:#2c3e50;font-weight:bold">
    {_fmt_value(row)}
  </td>
  <td style="padding:8px 12px;font-size:12px;text-align:right;
             border-top:1px solid #eee;white-space:nowrap">
    1d {_fmt_chg(row['chg_1d_pct'])}<br>
    1w {_fmt_chg(row['chg_1w_pct'])}<br>
    1m {_fmt_chg(row['chg_1m_pct'])}
  </td>
</tr>"""

    status_rows = ""
    if "vix_regime" in status:
        vix_en, vix_zh = status["vix_regime"]
        status_rows += (
            f"<li><b>Volatility / 波动性:</b> VIX "
            f"{status.get('vix_level', 0):.1f} — {vix_en} / {vix_zh}</li>"
        )
    if "curve_10y_3m_bps" in status:
        inv_note = (" — INVERTED / 倒挂"
                    if status.get("curve_inverted") else "")
        status_rows += (
            f"<li><b>Yield Curve / 收益率曲线:</b> "
            f"10Y - 3M = {status['curve_10y_3m_bps']:+.0f} bps{inv_note}</li>"
        )
    if "usd_trend" in status:
        usd_en, usd_zh = status["usd_trend"]
        status_rows += (
            f"<li><b>US Dollar / 美元:</b> {usd_en} / {usd_zh} (30d)</li>"
        )
    if "spy_curr" in status:
        smas = []
        for lbl_en, lbl_zh, key in (
            ("20d",  "20日均线",  "spy_above_20"),
            ("50d",  "50日均线",  "spy_above_50"),
            ("200d", "200日均线", "spy_above_200"),
        ):
            arrow = "▲" if status.get(key) else "▼"
            color = "#27ae60" if status.get(key) else "#e74c3c"
            smas.append(f'<span style="color:{color}">{arrow} {lbl_en}</span>')
        status_rows += (
            f"<li><b>SPY Trend / 标普走势:</b> ${status['spy_curr']:.2f} — "
            + " · ".join(smas) + "</li>"
        )

    takeaways_html = ""
    for en, zh in zip(summary.get("takeaways_en", []),
                      summary.get("takeaways_zh", [])):
        takeaways_html += (
            f'<li style="margin-bottom:8px">'
            f'<span style="color:#2c3e50">{en}</span><br>'
            f'<span style="color:#7f8c8d;font-size:12px">{zh}</span></li>'
        )

    news_html = ""
    if articles:
        for a in articles:
            news_html += f"""
<tr><td style="padding:10px 12px;border-top:1px solid #eee">
  <a href="{a.get('url', '#')}"
     style="color:#2c3e50;font-size:13px;font-weight:bold;text-decoration:none">
    {a.get('title', '')}
  </a><br>
  <span style="color:#7f8c8d;font-size:11px">
    {a.get('source', '')}
  </span>
</td></tr>"""
    else:
        news_html = (
            '<tr><td style="padding:14px;text-align:center;'
            'color:#bdc3c7;font-size:12px">'
            'News source unavailable / 新闻源不可用</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Market Brief</title>
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
        Market Brief &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / 市场日报
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; Pre-market snapshot / 盘前快照
      </p>
    </td></tr>

    <!-- LLM summary -->
    <tr><td style="padding:16px 20px;background:#f8f9fa">
      <p style="margin:0 0 6px;font-size:14px;color:#2c3e50">
        {summary.get('summary_en', '')}
      </p>
      <p style="margin:0;font-size:13px;color:#5d6d7e">
        {summary.get('summary_zh', '')}
      </p>
    </td></tr>

    <!-- Key takeaways -->
    <tr><td style="padding:12px 20px 4px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        KEY TAKEAWAYS / 关键要点
      </p>
      <ol style="margin:0 0 10px 18px;padding:0;font-size:13px">
        {takeaways_html}
      </ol>
    </td></tr>

    <!-- Macro table -->
    <tr><td style="padding:0 20px">
      <p style="margin:16px 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        MACRO / 宏观数据
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border:1px solid #ecf0f1;border-radius:4px">
        {macro_rows}
      </table>
    </td></tr>

    <!-- Regime status -->
    <tr><td style="padding:16px 20px">
      <p style="margin:0 0 8px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        MARKET REGIME / 市场状态
      </p>
      <ul style="margin:0;padding-left:18px;font-size:13px;color:#2c3e50;
                 line-height:1.8">
        {status_rows}
      </ul>
    </td></tr>

    <!-- News -->
    <tr><td style="padding:12px 20px 20px">
      <p style="margin:0 0 4px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        CURATED HEADLINES / 精选新闻
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border:1px solid #ecf0f1;border-radius:4px">
        {news_html}
      </table>
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center">
        Sources: yfinance macro · RSS: CNBC / WSJ / MarketWatch / Yahoo / FT ·
        Curation: Claude Sonnet 4.6. Informational only. /
        仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_market_brief_pipeline(config: dict | None = None) -> dict:
    """End-to-end pipeline. Always attempts to send email even if any single
    stage returns empty."""
    logger.info("=== Market Brief pipeline start ===")

    macro    = fetch_macro_data()
    logger.info(f"MacroBrief: fetched {len(macro)} macro tickers")

    status   = compute_market_status(macro)
    logger.info(f"MacroBrief: regime={status.get('vix_regime')}, "
                f"curve_bps={status.get('curve_10y_3m_bps')}")

    raw_news = fetch_news_rss()
    logger.info(f"MacroBrief: {len(raw_news)} RSS items after dedup")

    ant_key  = os.environ.get("ANTHROPIC_API_KEY", "")
    summary  = summarize_news_with_llm(raw_news, macro, status, ant_key)
    picked   = summary.get("picked_articles", [])

    html = render_brief_html(macro, status, picked, summary)

    try:
        from notify.mailer import _smtp_send
        subject = f"[Market Brief] {dt.date.today()} — pre-market"
        _smtp_send(html, subject)
        logger.info("MacroBrief: email sent")
    except Exception as e:
        logger.warning(f"MacroBrief: email send failed — {e}")

    logger.info("=== Market Brief pipeline complete ===")
    return {
        "macro":    macro,
        "status":   status,
        "raw_news": raw_news,
        "picked":   picked,
        "summary":  summary,
        "html":     html,
    }
