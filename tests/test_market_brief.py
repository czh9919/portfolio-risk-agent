"""
Unit tests for notify/market_brief.py — deterministic components only.

Network-dependent stages (yfinance, RSS, Anthropic) are exercised via mocked
responses so tests stay fast and offline-safe.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from notify.market_brief import (
    RSS_FEEDS,
    compute_market_status,
    fetch_news_rss,
    render_brief_html,
    summarize_news_with_llm,
)


# ── Regime derivation ─────────────────────────────────────────────────────────

def _macro(**over):
    """Minimal macro dict; caller overrides values as needed."""
    base = {
        "^VIX":     {"label_en": "VIX",       "label_zh": "波动率", "fmt": "index",
                     "curr": 14.0,  "chg_1d_pct": 0, "chg_1w_pct": 0, "chg_1m_pct": 0},
        "^TNX":     {"label_en": "10Y",       "label_zh": "10年",   "fmt": "yield",
                     "curr": 4.25,  "chg_1d_pct": 0, "chg_1w_pct": 0, "chg_1m_pct": 0},
        "^IRX":     {"label_en": "3M",        "label_zh": "3月",    "fmt": "yield",
                     "curr": 5.40,  "chg_1d_pct": 0, "chg_1w_pct": 0, "chg_1m_pct": 0},
        "DX-Y.NYB": {"label_en": "DXY",       "label_zh": "美元",   "fmt": "index",
                     "curr": 104.0, "chg_1d_pct": 0, "chg_1w_pct": 0, "chg_1m_pct": 0.5},
    }
    for tkr, patch in over.items():
        base.setdefault(tkr, {}).update(patch)
    return base


def test_vix_buckets():
    for level, expected in [(10, "calm"), (17, "normal"),
                            (22, "elevated"), (30, "stressed")]:
        with patch("notify.market_brief._spy_trend_status", return_value={}):
            m = _macro(**{"^VIX": {"curr": level}})
            s = compute_market_status(m)
        assert s["vix_regime"][0] == expected
        assert s["vix_level"] == float(level)


def test_curve_inversion_detected():
    # 3M > 10Y ⇒ inverted (10Y - 3M negative)
    m = _macro(**{"^TNX": {"curr": 4.0}, "^IRX": {"curr": 5.0}})
    with patch("notify.market_brief._spy_trend_status", return_value={}):
        s = compute_market_status(m)
    assert s["curve_inverted"] is True
    assert s["curve_10y_3m_bps"] < 0

    m = _macro(**{"^TNX": {"curr": 5.0}, "^IRX": {"curr": 4.0}})
    with patch("notify.market_brief._spy_trend_status", return_value={}):
        s = compute_market_status(m)
    assert s["curve_inverted"] is False
    assert s["curve_10y_3m_bps"] > 0


def test_usd_trend_buckets():
    for pct, expected in [(2.0, "strengthening"), (-2.0, "weakening"),
                          (0.5, "range-bound")]:
        m = _macro(**{"DX-Y.NYB": {"chg_1m_pct": pct}})
        with patch("notify.market_brief._spy_trend_status", return_value={}):
            s = compute_market_status(m)
        assert s["usd_trend"][0] == expected


# ── RSS fetch ─────────────────────────────────────────────────────────────────

def _rss_entry(title: str, hours_ago: float = 1.0, url: str = "https://x/y",
               summary: str = ""):
    """Build a fake feedparser entry dict."""
    ts = (dt.datetime.utcnow() - dt.timedelta(hours=hours_ago)).utctimetuple()
    return {
        "title":            title,
        "link":             url,
        "summary":          summary,
        "published_parsed": ts,
    }


def _fake_feed(entries: list[dict]):
    return SimpleNamespace(entries=entries)


def test_rss_lookback_drops_old_items():
    fresh = _rss_entry("Fresh headline", hours_ago=1)
    stale = _rss_entry("Stale headline", hours_ago=48)
    fake_feedparser = MagicMock()
    fake_feedparser.parse = MagicMock(return_value=_fake_feed([fresh, stale]))

    with patch.dict("sys.modules", {"feedparser": fake_feedparser}):
        picked = fetch_news_rss(feeds={"only": "http://x"},
                                lookback_hours=24)
    assert [a["title"] for a in picked] == ["Fresh headline"]


def test_rss_dedup_across_feeds():
    """Same title from two feeds appears only once."""
    entry = _rss_entry("Fed pauses rate hikes", hours_ago=1)
    fake_feedparser = MagicMock()
    fake_feedparser.parse = MagicMock(return_value=_fake_feed([entry]))

    with patch.dict("sys.modules", {"feedparser": fake_feedparser}):
        picked = fetch_news_rss(feeds={"A": "http://a", "B": "http://b"},
                                lookback_hours=24)
    assert len(picked) == 1
    assert picked[0]["title"] == "Fed pauses rate hikes"


def test_rss_feed_error_skipped_gracefully():
    """A dead feed logs a warning and the others still return items."""
    fake_feedparser = MagicMock()
    ok_entry = _rss_entry("Live headline", hours_ago=1)
    call_count = {"n": 0}

    def parse(url, agent=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("dead feed")
        return _fake_feed([ok_entry])
    fake_feedparser.parse = parse

    with patch.dict("sys.modules", {"feedparser": fake_feedparser}):
        picked = fetch_news_rss(feeds={"dead": "http://d", "ok": "http://o"},
                                lookback_hours=24)
    assert [a["title"] for a in picked] == ["Live headline"]


def test_rss_max_per_source_cap():
    entries = [_rss_entry(f"Headline {i}", hours_ago=1) for i in range(50)]
    fake_feedparser = MagicMock()
    fake_feedparser.parse = MagicMock(return_value=_fake_feed(entries))

    with patch.dict("sys.modules", {"feedparser": fake_feedparser}):
        picked = fetch_news_rss(feeds={"only": "http://x"},
                                lookback_hours=24, max_per_source=5)
    assert len(picked) == 5


def test_rss_missing_pub_date_skipped():
    """RSS items without published_parsed can't be windowed — drop them."""
    good = _rss_entry("Dated headline", hours_ago=1)
    bad  = {"title": "Undated headline", "link": "x", "summary": ""}
    fake_feedparser = MagicMock()
    fake_feedparser.parse = MagicMock(return_value=_fake_feed([good, bad]))

    with patch.dict("sys.modules", {"feedparser": fake_feedparser}):
        picked = fetch_news_rss(feeds={"only": "http://x"}, lookback_hours=24)
    assert [a["title"] for a in picked] == ["Dated headline"]


def test_rss_feed_list_nonempty():
    """Sanity: don't ship a config that would silently deliver no news."""
    assert len(RSS_FEEDS) >= 5
    assert any("cnbc.com" in url for url in RSS_FEEDS.values())


# ── LLM curation + summary ────────────────────────────────────────────────────

def _fake_anthropic_module(response_text: str):
    fake_block = MagicMock();  fake_block.text = response_text
    fake_msg   = MagicMock();  fake_msg.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_module = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=fake_client)
    return fake_module, fake_client


def _mk_articles(n: int) -> list[dict]:
    return [{"title": f"Headline {i}", "source": "Test",
             "url": f"https://x/{i}", "summary": "", "published": ""}
            for i in range(n)]


def test_llm_fallback_when_no_key():
    articles = _mk_articles(3)
    out = summarize_news_with_llm(articles, {}, {}, api_key="")
    assert out["picked_articles"] == articles[:8]
    assert "summary_en" in out and "summary_zh" in out


def test_llm_returns_picked_articles_from_indices(monkeypatch):
    import sys
    articles = _mk_articles(20)

    payload = {
        "picked_indices": [3, 7, 15, 1, 9],
        "summary_en":     "Calm session.",
        "summary_zh":     "平静的一天。",
        "takeaways_en":   ["A", "B", "C"],
        "takeaways_zh":   ["甲", "乙", "丙"],
    }
    fake_module, _ = _fake_anthropic_module(json.dumps(payload))
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = summarize_news_with_llm(articles, {}, {}, api_key="dummy")
    picked_titles = [a["title"] for a in out["picked_articles"]]
    assert picked_titles == ["Headline 3", "Headline 7", "Headline 15",
                             "Headline 1", "Headline 9"]
    assert out["summary_en"] == "Calm session."
    assert out["takeaways_zh"] == ["甲", "乙", "丙"]


def test_llm_out_of_range_indices_ignored(monkeypatch):
    import sys
    articles = _mk_articles(5)

    payload = {
        "picked_indices": [2, 99, -1, 0],   # 99 and -1 should be dropped
        "summary_en":     "x", "summary_zh": "y",
        "takeaways_en":   ["a"], "takeaways_zh": ["甲"],
    }
    fake_module, _ = _fake_anthropic_module(json.dumps(payload))
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = summarize_news_with_llm(articles, {}, {}, api_key="dummy")
    assert [a["title"] for a in out["picked_articles"]] == [
        "Headline 2", "Headline 0"
    ]


def test_llm_strips_code_fences(monkeypatch):
    import sys
    articles = _mk_articles(3)
    payload = {
        "picked_indices": [0], "summary_en": "x", "summary_zh": "y",
        "takeaways_en": ["a"], "takeaways_zh": ["甲"],
    }
    fenced = f"```json\n{json.dumps(payload)}\n```"
    fake_module, _ = _fake_anthropic_module(fenced)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = summarize_news_with_llm(articles, {}, {}, api_key="dummy")
    assert out["summary_en"] == "x"
    assert out["picked_articles"] == [articles[0]]


def test_llm_falls_back_on_api_error(monkeypatch):
    import sys
    articles = _mk_articles(3)

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("API down")
    fake_module = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = summarize_news_with_llm(articles, {}, {}, api_key="dummy")
    # Fallback still delivers first 8 raw articles
    assert out["picked_articles"] == articles[:8]


# ── HTML rendering ────────────────────────────────────────────────────────────

def test_render_html_contains_all_sections():
    macro = _macro()
    status = {
        "vix_regime": ("calm", "低波动"), "vix_level": 14.0,
        "curve_10y_3m_bps": -115.0, "curve_inverted": True,
        "usd_trend": ("strengthening", "走强"),
    }
    articles = [{
        "title": "Fed holds rates steady",
        "url":   "https://reuters.com/x",
        "source":"CNBC", "summary": "", "published": "",
    }]
    summary = {
        "summary_en":   "Calm pre-market.",
        "summary_zh":   "盘前平静。",
        "takeaways_en": ["Fed on hold", "USD firm", "Watch CPI"],
        "takeaways_zh": ["联储按兵不动", "美元坚挺", "关注 CPI"],
    }
    html = render_brief_html(macro, status, articles, summary)

    assert "Market Brief" in html and "市场日报" in html
    assert "Calm pre-market." in html and "盘前平静。" in html
    assert "Fed on hold" in html and "联储按兵不动" in html
    assert "VIX" in html
    assert "INVERTED" in html and "倒挂" in html
    assert 'href="https://reuters.com/x"' in html
    assert "Fed holds rates steady" in html
    # No sentiment display any more
    assert "sentiment" not in html.lower()


def test_render_html_handles_empty_news():
    """When RSS returns nothing, HTML still renders with a placeholder."""
    html = render_brief_html(
        macro=_macro(), status={}, articles=[],
        summary={"summary_en": "", "summary_zh": "",
                 "takeaways_en": [], "takeaways_zh": []},
    )
    assert "News source unavailable" in html
    assert "新闻源不可用" in html


def test_render_html_handles_negative_pct_color():
    m = _macro(**{"^VIX": {"chg_1d_pct": -3.2}})
    html = render_brief_html(
        macro=m, status={}, articles=[],
        summary={"summary_en": "", "summary_zh": "",
                 "takeaways_en": [], "takeaways_zh": []},
    )
    assert "#e74c3c" in html
