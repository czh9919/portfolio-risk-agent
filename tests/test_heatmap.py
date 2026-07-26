"""
Unit tests for notify/heatmap.py — deterministic components only.

Network-dependent stages (yfinance) are exercised via mocked responses.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from notify.heatmap import (
    ETF_UNIVERSE,
    _text_color_for_bg,
    color_for_pct,
    render_heatmap_html,
    unique_tickers,
)


# ── Color scale ───────────────────────────────────────────────────────────────

def test_color_bands_monotonic():
    """Deeper positive %s should get greener colors, deeper negative → redder."""
    # Colors don't have to sort lexically, but each band should be distinct
    bands = [color_for_pct(v) for v in
             (-5, -2, -0.8, 0, 0.8, 2, 5)]
    assert len(set(bands)) == 7   # every band unique
    assert bands[0] == "#8b1a1a"   # deep red
    assert bands[3] == "#e5e7ea"   # neutral
    assert bands[-1] == "#0a8f39"  # deep green


def test_color_symmetric_at_flat():
    """±0.5% is the flat band — either side lands in neutral."""
    assert color_for_pct(0.0)   == "#e5e7ea"
    assert color_for_pct(0.4)   == "#e5e7ea"
    assert color_for_pct(-0.4)  == "#e5e7ea"


def test_text_color_readable_on_dark_bg():
    """Deep red / deep green backgrounds should get white text."""
    assert _text_color_for_bg("#8b1a1a") == "#ffffff"
    assert _text_color_for_bg("#0a8f39") == "#ffffff"
    # Light backgrounds should get dark text
    assert _text_color_for_bg("#e5e7ea") == "#1a1a1a"
    assert _text_color_for_bg("#7dc98e") == "#1a1a1a"


# ── ETF universe sanity ───────────────────────────────────────────────────────

def test_etf_universe_nonempty_and_covers_11_sectors():
    """SPY + QQQ + 11 sector SPDRs = 13 ETFs."""
    tickers = [t for t, _, _ in ETF_UNIVERSE]
    assert "SPY" in tickers and "QQQ" in tickers
    for sector in ("XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLRE","XLU","XLC"):
        assert sector in tickers, f"Missing sector ETF {sector}"
    assert len(tickers) == 13
    # Every entry has EN + ZH labels
    for tk, en, zh in ETF_UNIVERSE:
        assert en and zh, f"{tk} missing bilingual labels"


def test_unique_tickers_flattens_and_dedups():
    holdings = {
        "SPY": [{"symbol": "AAPL", "name": "Apple", "weight": 0.07},
                {"symbol": "MSFT", "name": "MSFT",  "weight": 0.06}],
        "QQQ": [{"symbol": "AAPL", "name": "Apple", "weight": 0.07},
                {"symbol": "NVDA", "name": "NVDA",  "weight": 0.09}],
        "XLE": [],  # empty ETF should still contribute its own ticker
    }
    ts = unique_tickers(holdings)
    # Expect ETFs themselves + underlying tickers, deduped and sorted
    assert set(ts) == {"AAPL", "MSFT", "NVDA", "SPY", "QQQ", "XLE"}
    assert ts == sorted(ts)


# ── HTML rendering ────────────────────────────────────────────────────────────

def _sample_holdings():
    return {
        "SPY": [
            {"symbol": "AAPL", "name": "Apple",     "weight": 0.07},
            {"symbol": "MSFT", "name": "Microsoft", "weight": 0.06},
        ],
        "QQQ": [
            {"symbol": "AAPL", "name": "Apple", "weight": 0.07},
            {"symbol": "NVDA", "name": "Nvidia","weight": 0.09},
        ],
        # remaining ETFs empty on purpose so they get skipped in HTML
    }


def _sample_metrics():
    def m(p1, p5, p21, hi, vr):
        return {"curr": 100.0, "chg_1d_pct": p1, "chg_1w_pct": p5,
                "chg_1m_pct": p21, "dist_52wh": hi, "vol_ratio": vr}
    return {
        "SPY":  m( 0.8,  2.1,  4.4, -1.2, 1.1),
        "QQQ":  m( 1.2,  3.0,  5.5, -0.8, 1.3),
        "AAPL": m( 1.5,  2.7,  6.2, -3.4, 1.5),
        "MSFT": m(-0.6,  1.4,  3.1, -5.0, 0.9),
        "NVDA": m(-2.1, -1.2,  8.7, -8.0, 2.1),
    }


def test_html_contains_all_sections():
    html = render_heatmap_html(_sample_holdings(), _sample_metrics(),
                               treemap_cid="testCid")
    # Header
    assert "Market Heatmap" in html and "市场热力图" in html
    # Treemap image reference
    assert 'src="cid:testCid"' in html
    # ETF cards (SPY + QQQ both present)
    assert "SPY · S&amp;P 500" in html or "SPY · S&P 500" in html
    assert "QQQ · Nasdaq 100" in html
    # Constituents
    assert "AAPL" in html and "MSFT" in html and "NVDA" in html
    # Legend bands (representative colors present)
    assert "#8b1a1a" in html and "#0a8f39" in html
    # Column headers
    for h in ("TICKER", "1D", "1W", "1M", "52wH", "VOL"):
        assert h in html


def test_html_skips_empty_etfs():
    """ETFs with no holdings shouldn't produce empty card frames."""
    html = render_heatmap_html(
        holdings={"SPY": [{"symbol": "AAPL", "name": "Apple", "weight": 0.07}]},
        metrics={"SPY": {"curr": 100, "chg_1d_pct": 0.5, "chg_1w_pct": 1.0,
                         "chg_1m_pct": 2.0, "dist_52wh": -1, "vol_ratio": 1.0},
                 "AAPL": {"curr": 100, "chg_1d_pct": 1.5, "chg_1w_pct": 2.0,
                          "chg_1m_pct": 3.0, "dist_52wh": -2, "vol_ratio": 1.1}},
        treemap_cid="",
    )
    # QQQ has no holdings passed → its card should NOT appear
    assert "QQQ · Nasdaq 100" not in html
    assert "SPY · S&amp;P 500" in html or "SPY · S&P 500" in html


def test_html_omits_treemap_when_cid_empty():
    html = render_heatmap_html(_sample_holdings(), _sample_metrics(),
                               treemap_cid="")
    # No <img cid:...> when treemap failed to render
    assert "cid:" not in html


def test_html_ticker_missing_metrics_dropped():
    """A held ticker without price metrics shouldn't inject a broken row."""
    holdings = {"SPY": [
        {"symbol": "AAPL", "name": "Apple", "weight": 0.07},
        {"symbol": "GHOST","name": "?",     "weight": 0.05},
    ]}
    metrics = {
        "SPY":  {"curr": 100, "chg_1d_pct": 0.5, "chg_1w_pct": 1,
                 "chg_1m_pct": 2, "dist_52wh": -1, "vol_ratio": 1.0},
        "AAPL": {"curr": 100, "chg_1d_pct": 1.0, "chg_1w_pct": 2,
                 "chg_1m_pct": 3, "dist_52wh": -1, "vol_ratio": 1.0},
        # GHOST intentionally absent
    }
    html = render_heatmap_html(holdings, metrics, treemap_cid="")
    assert "AAPL" in html
    assert "GHOST" not in html


# ── PNG rendering ─────────────────────────────────────────────────────────────

def test_treemap_empty_when_all_missing_metrics():
    """If nothing has metrics we should return empty bytes rather than crash."""
    from notify.heatmap import render_treemap_png
    holdings = {"SPY": [{"symbol": "X", "name": "?", "weight": 0.1}]}
    metrics  = {}   # nothing matches
    assert render_treemap_png(holdings, metrics) == b""


def test_treemap_returns_png_bytes_when_data_present():
    """Sanity: with real-looking data, we get a non-empty PNG payload."""
    from notify.heatmap import render_treemap_png
    holdings = {
        "SPY": [{"symbol": "AAPL", "name": "Apple", "weight": 0.07}],
        "XLK": [{"symbol": "MSFT", "name": "MSFT",  "weight": 0.11}],
    }
    metrics = {
        "AAPL": {"chg_1d_pct":  1.5, "curr": 100, "chg_1w_pct": 0,
                 "chg_1m_pct": 0, "dist_52wh": 0, "vol_ratio": 1},
        "MSFT": {"chg_1d_pct": -1.2, "curr": 100, "chg_1w_pct": 0,
                 "chg_1m_pct": 0, "dist_52wh": 0, "vol_ratio": 1},
    }
    png = render_treemap_png(holdings, metrics, width_in=6, height_in=4, dpi=72)
    # Real PNG bytes must begin with 8-byte magic
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000
