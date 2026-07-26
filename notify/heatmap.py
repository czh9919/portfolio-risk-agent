"""
Market Heatmap — post-close ETF heatmap.

Fires at 22:15 UTC weekdays via .github/workflows/heatmap.yml
(RUN_MODE=heatmap). Pipeline:

  1. Fetch top-10 constituents (+ weight) of 13 ETFs via yfinance
     (SPY, QQQ, and the 11 GICS sector SPDRs)
  2. Batch-download 60d OHLCV for all unique constituents (~100 tickers)
  3. Compute per-ticker metrics: 1d/1w/1m return, 52w-high distance, volume ratio
  4. Render finviz-style treemap PNG (squarify, boxes sized by ETF weight
     within its group, colored by 1d return)
  5. Render bilingual HTML: one card per ETF with color-coded return cells
  6. Send single email with treemap CID-embedded above the cards

Fails gracefully:
  - Any ETF whose holdings fetch fails is skipped (others still render)
  - yfinance batch failure → empty brief still sends with a notice
  - matplotlib/squarify unavailable → skip PNG, send HTML-only
"""
from __future__ import annotations

import datetime as dt
import io
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


# ── ETF universe ──────────────────────────────────────────────────────────────

# (ETF ticker, English label, Chinese label)
ETF_UNIVERSE: list[tuple[str, str, str]] = [
    ("SPY",  "S&P 500",              "标普500"),
    ("QQQ",  "Nasdaq 100",           "纳指100"),
    ("XLK",  "Technology",           "科技"),
    ("XLF",  "Financials",           "金融"),
    ("XLE",  "Energy",               "能源"),
    ("XLV",  "Health Care",          "医疗"),
    ("XLY",  "Consumer Discretionary","可选消费"),
    ("XLP",  "Consumer Staples",     "必需消费"),
    ("XLI",  "Industrials",          "工业"),
    ("XLB",  "Materials",            "材料"),
    ("XLRE", "Real Estate",          "房地产"),
    ("XLU",  "Utilities",            "公用事业"),
    ("XLC",  "Communication",        "通信服务"),
]

TOP_N_PER_ETF = 10


# ── ETF holdings ──────────────────────────────────────────────────────────────

def fetch_etf_holdings() -> dict[str, list[dict]]:
    """Fetch top-N constituents (+ weight) per ETF via yfinance funds_data.

    Returns:
        {etf_ticker: [{"symbol": str, "name": str, "weight": float}, ...]}
        Empty list per ETF if fetch fails.
    """
    import yfinance as yf

    out: dict[str, list[dict]] = {}
    for etf, _label_en, _label_zh in ETF_UNIVERSE:
        try:
            fd = yf.Ticker(etf).funds_data
            df = fd.top_holdings
        except Exception as e:
            logger.warning(f"Heatmap: {etf} holdings fetch failed — {e}")
            out[etf] = []
            continue

        if df is None or len(df) == 0:
            out[etf] = []
            continue

        holdings: list[dict] = []
        for sym, row in df.head(TOP_N_PER_ETF).iterrows():
            holdings.append({
                "symbol": str(sym),
                "name":   str(row.get("Name", "")),
                "weight": float(row.get("Holding Percent", 0.0)),
            })
        out[etf] = holdings
    return out


def unique_tickers(holdings: dict[str, list[dict]]) -> list[str]:
    """Flatten ETF holdings into a de-duplicated list of ticker symbols."""
    seen: set[str] = set()
    for etf, hs in holdings.items():
        seen.add(etf)
        for h in hs:
            seen.add(h["symbol"])
    return sorted(seen)


# ── Price metrics ─────────────────────────────────────────────────────────────

def fetch_price_metrics(tickers: list[str]) -> dict[str, dict]:
    """Batch-download 60d daily OHLCV; compute per-ticker return metrics.

    Returns:
        {ticker: {
            "curr":       last close ($),
            "chg_1d_pct": 1-day return %,
            "chg_1w_pct": 5-session return %,
            "chg_1m_pct": 21-session return %,
            "dist_52wh":  % distance from 52-week high (negative when below),
            "vol_ratio":  today's vol / 20-day mean vol,
        }}
        Tickers whose fetch fails silently drop out.
    """
    if not tickers:
        return {}

    import yfinance as yf

    end   = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=380)   # need ~252 sessions for 52wh

    try:
        df = yf.download(
            tickers, start=start, end=end,
            progress=False, auto_adjust=False, group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"Heatmap: yfinance batch failed — {e}")
        return {}

    out: dict[str, dict] = {}
    for tkr in tickers:
        try:
            close = df[tkr]["Close"] if (tkr, "Close") in df.columns else df["Close"]
            volu  = df[tkr]["Volume"] if (tkr, "Volume") in df.columns else df["Volume"]
        except (KeyError, TypeError):
            continue

        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if hasattr(volu, "squeeze"):
            volu = volu.squeeze()
        if hasattr(close, "dropna"):
            close = close.dropna()
        if hasattr(volu, "dropna"):
            volu = volu.dropna()

        if close is None or len(close) < 2:
            continue

        curr = float(close.iloc[-1])
        d1   = float(close.iloc[-2])
        d5   = float(close.iloc[-6])  if len(close) >= 6  else d1
        d21  = float(close.iloc[-22]) if len(close) >= 22 else d5

        # 52-week high uses last 252 sessions (or all available if fewer)
        window52 = close.tail(252)
        high52 = float(window52.max()) if len(window52) else curr
        dist_52wh = ((curr - high52) / high52) * 100 if high52 else 0.0

        # Volume ratio: today / mean of last 20 sessions (exclude today)
        vol_today = float(volu.iloc[-1]) if len(volu) else 0.0
        vol_20    = float(volu.iloc[-21:-1].mean()) if len(volu) >= 21 else vol_today
        vol_ratio = (vol_today / vol_20) if vol_20 > 0 else 1.0

        out[tkr] = {
            "curr":       curr,
            "chg_1d_pct": ((curr - d1)  / d1)  * 100 if d1  else 0.0,
            "chg_1w_pct": ((curr - d5)  / d5)  * 100 if d5  else 0.0,
            "chg_1m_pct": ((curr - d21) / d21) * 100 if d21 else 0.0,
            "dist_52wh":  dist_52wh,
            "vol_ratio":  vol_ratio,
        }
    return out


# ── Color scale ───────────────────────────────────────────────────────────────

def color_for_pct(pct: float) -> str:
    """Return an HTML hex color for a return %.

    Symmetric red↔green scale with 5 bands. ±0.5% is "flat" (light gray).
    """
    if pct >= 3.0:  return "#0a8f39"    # deep green
    if pct >= 1.0:  return "#28a745"    # green
    if pct >= 0.5:  return "#7dc98e"    # light green
    if pct > -0.5:  return "#e5e7ea"    # neutral
    if pct > -1.0:  return "#e6a5a2"    # light red
    if pct > -3.0:  return "#dc3545"    # red
    return "#8b1a1a"                    # deep red


def _text_color_for_bg(hex_bg: str) -> str:
    """Pick white or black label text for readability on `hex_bg`."""
    r = int(hex_bg[1:3], 16)
    g = int(hex_bg[3:5], 16)
    b = int(hex_bg[5:7], 16)
    # Standard sRGB relative luminance
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 140 else "#1a1a1a"


# ── PNG treemap ───────────────────────────────────────────────────────────────

def render_treemap_png(
    holdings: dict[str, list[dict]],
    metrics:  dict[str, dict],
    width_in: float = 12.0,
    height_in: float = 8.0,
    dpi: int = 130,
) -> bytes:
    """Render a finviz-style treemap grouping constituents by ETF.

    Each ETF becomes a sub-rectangle; within it, tickers are sized by
    their weight in the ETF and colored by 1-day % change. Returns
    empty bytes if matplotlib or squarify is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import squarify
    except ImportError as e:
        logger.warning(f"Heatmap: PNG rendering unavailable ({e}); "
                       "email will be HTML-only")
        return b""

    # ETF-level: box size proportional to sum of held weights that we
    # actually have metrics for (so empty ETFs don't get space).
    etf_weight: dict[str, float] = {}
    for etf, hs in holdings.items():
        w = sum(h["weight"] for h in hs
                if h["symbol"] in metrics)
        if w > 0:
            etf_weight[etf] = w
    if not etf_weight:
        return b""

    etf_labels  = list(etf_weight.keys())
    etf_sizes   = [etf_weight[e] for e in etf_labels]

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    ax.set_axis_off()

    # Squarify outer layout for ETFs
    outer_rects = squarify.normalize_sizes(etf_sizes, width_in, height_in)
    outer_rects = squarify.squarify(outer_rects, 0, 0, width_in, height_in)

    for etf, rect in zip(etf_labels, outer_rects):
        x0, y0, w0, h0 = rect["x"], rect["y"], rect["dx"], rect["dy"]

        # Inner: children of this ETF, sized by weight, colored by 1d change
        children = [h for h in holdings[etf] if h["symbol"] in metrics]
        if not children:
            continue

        child_sizes  = [h["weight"] for h in children]
        child_sizes  = squarify.normalize_sizes(child_sizes, w0, h0)
        child_rects  = squarify.squarify(child_sizes, x0, y0, w0, h0)

        # Slight inset so ETF boundaries are visible
        pad = 0.05
        for h, r in zip(children, child_rects):
            sym  = h["symbol"]
            pct  = metrics[sym]["chg_1d_pct"]
            bg   = color_for_pct(pct)
            fg   = _text_color_for_bg(bg)

            rx, ry = r["x"] + pad, r["y"] + pad
            rw, rh = max(r["dx"] - 2 * pad, 0.01), max(r["dy"] - 2 * pad, 0.01)

            ax.add_patch(mpatches.Rectangle(
                (rx, ry), rw, rh, facecolor=bg,
                edgecolor="#ffffff", linewidth=0.6,
            ))

            # Label size scales with box area
            area = rw * rh
            if area > 0.6:
                fs_sym, fs_pct = 11, 9
            elif area > 0.2:
                fs_sym, fs_pct = 8, 6.5
            elif area > 0.06:
                fs_sym, fs_pct = 6, 5
            else:
                continue  # too small to label

            cx, cy = rx + rw / 2, ry + rh / 2
            ax.text(cx, cy + 0.03, sym, ha="center", va="center",
                    fontsize=fs_sym, fontweight="bold", color=fg)
            ax.text(cx, cy - 0.10, f"{pct:+.1f}%", ha="center", va="center",
                    fontsize=fs_pct, color=fg)

        # ETF outer label (top-left of its rectangle)
        ax.text(x0 + 0.08, y0 + h0 - 0.15, etf,
                fontsize=9, fontweight="bold", color="#1a1a1a",
                bbox=dict(facecolor="#ffffff", edgecolor="none",
                          alpha=0.82, pad=1.4))

    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.invert_yaxis()

    # Title with date and legend
    today = dt.date.today().isoformat()
    ax.set_title(f"Market Heatmap — {today}  (color: 1-day return)",
                 fontsize=13, fontweight="bold", loc="left", pad=6)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


# ── HTML rendering ────────────────────────────────────────────────────────────

def _fmt_pct(pct: float) -> str:
    return f"{pct:+.2f}%"


def _render_ticker_cell(sym: str, m: dict) -> str:
    """One row inside an ETF card: ticker + 1d/1w/1m colored + 52wH + vol."""
    p1 = m["chg_1d_pct"]
    bg = color_for_pct(p1)
    fg = _text_color_for_bg(bg)
    return f"""
<tr>
  <td style="padding:6px 8px;font-size:12px;font-weight:bold;
             color:#2c3e50;border-top:1px solid #f0f0f0">{sym}</td>
  <td style="padding:6px 8px;background:{bg};color:{fg};text-align:right;
             font-size:12px;font-weight:bold;border-top:1px solid #f0f0f0">
    {_fmt_pct(p1)}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;
             color:{'#27ae60' if m['chg_1w_pct']>=0 else '#e74c3c'};
             border-top:1px solid #f0f0f0">
    {_fmt_pct(m['chg_1w_pct'])}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;
             color:{'#27ae60' if m['chg_1m_pct']>=0 else '#e74c3c'};
             border-top:1px solid #f0f0f0">
    {_fmt_pct(m['chg_1m_pct'])}
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:10px;color:#7f8c8d;
             border-top:1px solid #f0f0f0">
    {m['dist_52wh']:+.1f}%
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:10px;color:#7f8c8d;
             border-top:1px solid #f0f0f0">
    ×{m['vol_ratio']:.1f}
  </td>
</tr>"""


def _render_etf_card(etf: str, label_en: str, label_zh: str,
                     holdings: list[dict], metrics: dict[str, dict]) -> str:
    """One ETF section: header with ETF's own 1d return, then top-10 rows."""
    etf_pct = metrics.get(etf, {}).get("chg_1d_pct", 0.0)
    etf_bg  = color_for_pct(etf_pct)
    etf_fg  = _text_color_for_bg(etf_bg)

    rows = ""
    for h in holdings:
        sym = h["symbol"]
        if sym not in metrics:
            continue
        rows += _render_ticker_cell(sym, metrics[sym])

    if not rows:
        return ""

    return f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="margin:0 0 14px;border:1px solid #ecf0f1;border-radius:6px;
              overflow:hidden">
  <tr style="background:{etf_bg}">
    <td colspan="6" style="padding:10px 12px;color:{etf_fg}">
      <span style="font-size:14px;font-weight:bold">{etf} · {label_en}</span>
      <span style="font-size:11px;opacity:.85"> / {label_zh}</span>
      <span style="float:right;font-size:14px;font-weight:bold">
        {_fmt_pct(etf_pct)}
      </span>
    </td>
  </tr>
  <tr style="background:#f8f9fa">
    <th style="padding:6px 8px;text-align:left;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">TICKER</th>
    <th style="padding:6px 8px;text-align:right;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">1D</th>
    <th style="padding:6px 8px;text-align:right;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">1W</th>
    <th style="padding:6px 8px;text-align:right;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">1M</th>
    <th style="padding:6px 8px;text-align:right;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">52wH</th>
    <th style="padding:6px 8px;text-align:right;font-size:10px;
               color:#95a5a6;letter-spacing:.4px">VOL</th>
  </tr>
  {rows}
</table>"""


def render_heatmap_html(
    holdings: dict[str, list[dict]],
    metrics:  dict[str, dict],
    treemap_cid: str = "",
) -> str:
    today = dt.date.today().isoformat()

    cards = ""
    for etf, label_en, label_zh in ETF_UNIVERSE:
        cards += _render_etf_card(etf, label_en, label_zh,
                                  holdings.get(etf, []), metrics)

    treemap_html = ""
    if treemap_cid:
        treemap_html = f"""
    <tr><td style="padding:12px 12px 4px;background:#fff;text-align:center">
      <img src="cid:{treemap_cid}" alt="Market Heatmap Treemap"
           style="max-width:100%;height:auto;border:1px solid #ecf0f1;
                  border-radius:4px">
    </td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Market Heatmap</title>
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
        Market Heatmap &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / 市场热力图
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; Post-close snapshot / 收盘后快照
      </p>
    </td></tr>

    {treemap_html}

    <!-- ETF cards -->
    <tr><td style="padding:16px 20px 8px">
      <p style="margin:0 0 12px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        BY ETF · TOP-10 CONSTITUENTS / 按 ETF · 前十成分
      </p>
      {cards}
    </td></tr>

    <!-- Legend -->
    <tr><td style="padding:0 20px 12px">
      <p style="margin:0 0 6px;font-size:10px;color:#95a5a6;
                letter-spacing:.4px">
        LEGEND / 图例
      </p>
      <table cellpadding="0" cellspacing="0" style="font-size:10px">
        <tr>
          <td style="background:#8b1a1a;color:#fff;padding:2px 8px">&lt;−3%</td>
          <td style="background:#dc3545;color:#fff;padding:2px 8px">−1 to −3%</td>
          <td style="background:#e6a5a2;color:#1a1a1a;padding:2px 8px">−0.5 to −1%</td>
          <td style="background:#e5e7ea;color:#1a1a1a;padding:2px 8px">±0.5%</td>
          <td style="background:#7dc98e;color:#1a1a1a;padding:2px 8px">0.5 to 1%</td>
          <td style="background:#28a745;color:#fff;padding:2px 8px">1 to 3%</td>
          <td style="background:#0a8f39;color:#fff;padding:2px 8px">&gt;+3%</td>
        </tr>
      </table>
      <p style="margin:8px 0 0;font-size:10px;color:#95a5a6;line-height:1.4">
        <b>1D/1W/1M:</b> 1-day / 5-session / 21-session return · &nbsp;
        <b>52wH:</b> distance from 52-week high (negative if below) · &nbsp;
        <b>VOL:</b> today's volume / 20-day mean
      </p>
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center">
        Sources: yfinance (ETF holdings + prices). Treemap: squarify.
        Informational only. / 仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_heatmap_pipeline(config: dict | None = None) -> dict:
    """End-to-end. Always attempts to send email even if PNG rendering fails."""
    logger.info("=== Market Heatmap pipeline start ===")

    holdings = fetch_etf_holdings()
    non_empty = {k: v for k, v in holdings.items() if v}
    logger.info(f"Heatmap: fetched holdings for {len(non_empty)}/"
                f"{len(ETF_UNIVERSE)} ETFs")

    tickers = unique_tickers(holdings)
    metrics = fetch_price_metrics(tickers)
    logger.info(f"Heatmap: metrics for {len(metrics)}/{len(tickers)} tickers")

    png = render_treemap_png(holdings, metrics)
    logger.info(f"Heatmap: treemap PNG {len(png)} bytes")

    cid = "heatmapTreemap" if png else ""
    html = render_heatmap_html(holdings, metrics, treemap_cid=cid)

    try:
        from notify.mailer import _smtp_send
        subject = f"[Heatmap] {dt.date.today()} — sector rotation"
        _smtp_send(html, subject, chart_bytes=png, cid=cid)
        logger.info("Heatmap: email sent")
    except Exception as e:
        logger.warning(f"Heatmap: email send failed — {e}")

    logger.info("=== Market Heatmap pipeline complete ===")
    return {
        "holdings": holdings,
        "metrics":  metrics,
        "png_len":  len(png),
        "html":     html,
    }
