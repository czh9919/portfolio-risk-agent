"""
Unit tests for notify/ai_panel.py — deterministic components only.
"""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from notify.ai_panel import (
    _classify_action,
    _fmt_mcap,
    _fmt_pct,
    _parse_price,
    _pct_color,
    _safe_json,
    all_tickers,
    analyze_with_llm,
    compute_thermometer,
    render_html,
)


# ── Config helpers ────────────────────────────────────────────────────────────

def test_all_tickers_flattens_and_sorts():
    cfg = {
        "groups": {
            "chips":  {"tickers": ["NVDA", "AMD"]},
            "cloud":  {"tickers": ["MSFT", "GOOGL", "NVDA"]},  # NVDA dup
            "empty":  {"tickers": []},
        }
    }
    assert all_tickers(cfg) == ["AMD", "GOOGL", "MSFT", "NVDA"]


def test_all_tickers_missing_groups_key():
    assert all_tickers({}) == []


# ── Insider Text parsing ──────────────────────────────────────────────────────

def test_parse_price_extracts_from_text():
    assert _parse_price("Sale at price 402.84 per share.") == pytest.approx(402.84)
    assert _parse_price("Purchase at price 12.5 per share.") == pytest.approx(12.5)
    assert _parse_price("Stock Award(Grant) at price 0.00 per share.") == 0.0


def test_parse_price_handles_commas():
    assert _parse_price("Sale at price 1,234.56 per share.") == pytest.approx(1234.56)


def test_parse_price_returns_none_when_no_match():
    assert _parse_price("Some unrelated text") is None
    assert _parse_price("") is None
    assert _parse_price(None) is None


def test_classify_action_covers_common_verbs():
    assert _classify_action("Sale at price 100.00 per share.", 100, 10000) == "sell"
    assert _classify_action("Purchase at price 50.00 per share.", 100, 5000) == "buy"
    assert _classify_action("Stock Award(Grant) at price 0.00 per share.", 100, 0) == "grant"
    assert _classify_action("Stock Gift at price 0.00 per share.", 100, 0) == "gift"
    assert _classify_action("Option Exercise at price 10.00 per share.", 100, 1000) == "option_exercise"
    assert _classify_action("Some other transaction", 100, 0) == "other"


# ── Thermometer math ──────────────────────────────────────────────────────────

def _mk_metrics(n_up: int, n_down: int, avg_5d: float = 0.0,
                vol_ratio: float = 1.0) -> dict:
    """Fabricate `metrics` with n_up green + n_down red tickers."""
    out = {}
    for i in range(n_up):
        out[f"U{i}"] = {"chg_1d_pct":  2.0, "chg_5d_pct": avg_5d,
                        "chg_1m_pct": 0, "vol_ratio": vol_ratio}
    for i in range(n_down):
        out[f"D{i}"] = {"chg_1d_pct": -2.0, "chg_5d_pct": avg_5d,
                        "chg_1m_pct": 0, "vol_ratio": vol_ratio}
    return out


def test_thermometer_hot_when_all_up_beating_spy():
    metrics = _mk_metrics(n_up=10, n_down=0, avg_5d=8.0, vol_ratio=1.5)
    spy     = {"chg_1d_pct": -0.5}
    ins     = {f"U{i}": {"n_trades_30d": 3, "net_shares": 100}
               for i in range(10)}
    t = compute_thermometer(metrics, spy, ins)
    assert t["composite"] >= 75
    assert t["label_en"] in ("Hot", "Overheated / Hot")


def test_thermometer_cold_when_all_down():
    metrics = _mk_metrics(n_up=0, n_down=10, avg_5d=-8.0, vol_ratio=0.6)
    spy     = {"chg_1d_pct": 0.5}
    ins     = {f"D{i}": {"n_trades_30d": 3, "net_shares": -100}
               for i in range(10)}
    t = compute_thermometer(metrics, spy, ins)
    assert t["composite"] <= 30
    assert t["label_en"] in ("Cold", "Cool")


def test_thermometer_empty_metrics_returns_neutral():
    t = compute_thermometer({}, {}, {})
    assert t["composite"] == 50.0
    assert t["label_en"] == "N/A"


def test_thermometer_components_in_unit_interval():
    metrics = _mk_metrics(n_up=6, n_down=4, avg_5d=2.0)
    spy     = {"chg_1d_pct": 0.0}
    ins     = {}
    t = compute_thermometer(metrics, spy, ins)
    for k, v in t["components"].items():
        assert 0.0 <= v <= 1.0, f"{k} out of [0,1]: {v}"


# ── Formatting helpers ────────────────────────────────────────────────────────

def test_pct_color_bands():
    assert _pct_color(  5.0) == "#0a8f39"
    assert _pct_color(  1.5) == "#28a745"
    assert _pct_color(  0.5) == "#7f8c8d"
    assert _pct_color( -2.0) == "#dc3545"
    assert _pct_color( -5.0) == "#8b1a1a"
    assert _pct_color(None ) == "#7f8c8d"


def test_fmt_pct_none_shows_dash():
    assert _fmt_pct(None)  == "—"
    assert _fmt_pct(2.5)   == "+2.50%"
    assert _fmt_pct(-1.25) == "-1.25%"


def test_fmt_mcap_thresholds():
    assert _fmt_mcap(None)         == "—"
    assert _fmt_mcap(3.4e12)       == "$3.40T"
    assert _fmt_mcap(500e9)        == "$500.0B"
    assert _fmt_mcap(2.5e9)        == "$2.5B"
    assert _fmt_mcap(500e6)        == "$500M"


# ── LLM narrative ─────────────────────────────────────────────────────────────

def _fake_thermo(composite=50):
    return {"composite": composite, "label_en": "Neutral", "label_zh": "中性",
            "components": {}, "raw": {"pct_green": 50, "avg_1d_pct": 0.1,
                                       "avg_5d_pct": 0.3, "spy_1d_pct": 0.1,
                                       "rel_strength": 0.0, "avg_vol_ratio": 1.0,
                                       "pct_net_buy": 30, "n_leaders": 25}}


def test_llm_fallback_when_no_key():
    out = analyze_with_llm(_fake_thermo(60), [], [], {}, api_key="")
    assert "60" in out["narrative_en"]
    assert "60" in out["narrative_zh"]


def test_llm_parses_json_output(monkeypatch):
    import sys
    payload = {"narrative_en": "Sector cooling.", "narrative_zh": "板块降温。"}
    fake_block = MagicMock(); fake_block.text = json.dumps(payload)
    fake_msg   = MagicMock(); fake_msg.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_module = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = analyze_with_llm(_fake_thermo(), [], [], {}, api_key="dummy")
    assert out["narrative_en"] == "Sector cooling."
    assert out["narrative_zh"] == "板块降温。"


def test_safe_json_regex_fallback_when_unescaped_quote():
    """Sonnet occasionally emits an unescaped quote — verify the regex path."""
    bad = ('{"narrative_en": "Fed pauses "hawkishly" tonight.", '
           '"narrative_zh": "联储今晚鹰派停顿。"}')
    out = _safe_json(bad)
    assert out is not None
    assert "narrative_zh" in out
    assert "联储" in out["narrative_zh"]


def test_safe_json_strips_code_fences():
    fenced = ('```json\n{"narrative_en": "x", "narrative_zh": "y"}\n```')
    out = _safe_json(fenced)
    assert out == {"narrative_en": "x", "narrative_zh": "y"}


# ── HTML rendering ────────────────────────────────────────────────────────────

def test_render_html_includes_all_sections():
    thermo = _fake_thermo(65)
    thermo["label_en"] = "Hot"
    thermo["label_zh"] = "偏热"

    metrics = {
        "NVDA": {"chg_1d_pct": 2.5, "chg_5d_pct": 6.0, "chg_1m_pct": 12.0,
                 "curr": 1300.0, "vol_ratio": 1.2,
                 "dist_52wh": -1.5, "dist_52wl": 50.0},
        "MSFT": {"chg_1d_pct": -0.3, "chg_5d_pct": 1.1, "chg_1m_pct": 3.0,
                 "curr": 410.0, "vol_ratio": 0.9,
                 "dist_52wh": -2.0, "dist_52wl": 15.0},
    }
    valuations = {
        "NVDA": {"market_cap": 3.4e12, "fwd_pe": 32.0, "peg": 0.58,
                 "trailing_pe": 60, "ps": 20, "ev_ebitda": 28,
                 "profit_margin": 0.6, "revenue_growth": 0.5,
                 "earnings_growth": 1.0, "beta": 2.1, "pb": 25},
        "MSFT": {"market_cap": 2.8e12, "fwd_pe": 30.0, "peg": 1.2,
                 "trailing_pe": 35, "ps": 12, "ev_ebitda": 22,
                 "profit_margin": 0.35, "revenue_growth": 0.15,
                 "earnings_growth": 0.20, "beta": 1.1, "pb": 12},
    }
    groups = {
        "chips_hardware": {"label_en": "Chips", "label_zh": "芯片",
                           "tickers": ["NVDA"]},
        "cloud_infra":    {"label_en": "Cloud", "label_zh": "云",
                           "tickers": ["MSFT"]},
    }
    trades = [{
        "ticker": "MSFT", "insider": "NUMOTO TAKESHI", "position": "Officer",
        "action": "sell", "shares": 4500, "price": 402.84,
        "value_usd": 1_812_780, "date": "2026-06-10",
    }]
    insider_summary = {
        "NVDA": {"n_trades_30d": 5, "buys_shares": 1000,
                 "sells_shares": 500, "net_shares": 500},
        "MSFT": {"n_trades_30d": 3, "buys_shares": 0,
                 "sells_shares": 7000, "net_shares": -7000},
    }
    narr = {"narrative_en": "AI hot.", "narrative_zh": "AI 偏热。"}
    cids = {"thermometer": "ai_thermometer", "valuation": "ai_valuation",
            "insider": "ai_insider"}

    html = render_html(thermo, metrics, valuations, groups, trades,
                       insider_summary, narr, cids)

    # Header
    assert "AI Sector Panel" in html and "AI 板块检测" in html
    # Narrative
    assert "AI hot." in html and "AI 偏热。" in html
    # Charts embedded
    assert 'src="cid:ai_thermometer"' in html
    assert 'src="cid:ai_valuation"' in html
    assert 'src="cid:ai_insider"' in html
    # Sector groups
    assert "Chips" in html and "芯片" in html
    assert "Cloud" in html and "云" in html
    # Tickers + returns
    assert "NVDA" in html and "MSFT" in html
    assert "+2.50%" in html
    # Valuation columns
    assert "32.0" in html  # NVDA fwd PE
    assert "0.58" in html  # NVDA PEG
    assert "$3.40T" in html  # NVDA market cap
    # Insider table
    assert "NUMOTO TAKESHI" in html
    assert "sell" in html.lower() or "SELL" in html
    assert "2026-06-10" in html


def test_render_html_no_trades_shows_placeholder():
    html = render_html(
        _fake_thermo(), {}, {}, {}, [], {},
        {"narrative_en": "", "narrative_zh": ""}, {},
    )
    assert ("No material insider" in html
            or "过去 30 天无重要内部交易" in html)


# ── Insider fetch (with mocked yfinance) ──────────────────────────────────────

def test_fetch_insider_filters_lookback_and_skips_grants(monkeypatch):
    """Older transactions and zero-price grants drop out of `trades`."""
    from notify import ai_panel as mod

    today = dt.date.today()
    recent = today - dt.timedelta(days=5)
    old    = today - dt.timedelta(days=90)

    df = pd.DataFrame({
        "Shares":   [1000, 500, 2000],
        "Value":    [50000, 20000, 0],
        "URL":      ["", "", ""],
        "Text":     ["Sale at price 50.00 per share.",
                     "Purchase at price 40.00 per share.",
                     "Stock Award(Grant) at price 0.00 per share."],
        "Insider":  ["EXEC ONE", "EXEC TWO", "EXEC THREE"],
        "Position": ["CFO", "CEO", "Director"],
        "Transaction": ["Sale", "Purchase", "Grant"],
        "Start Date":  [pd.Timestamp(recent), pd.Timestamp(recent), pd.Timestamp(old)],
        "Ownership":   ["D", "D", "D"],
    })

    class FakeTicker:
        def __init__(self, sym): self.sym = sym
        @property
        def insider_transactions(self): return df

    with patch("yfinance.Ticker", FakeTicker):
        trades, summary = mod.fetch_insider_transactions(["FOO"], lookback_days=30)

    # Grant excluded (0-price), old excluded (>30d)
    assert len(trades) == 2
    actions = sorted(t["action"] for t in trades)
    assert actions == ["buy", "sell"]

    assert summary["FOO"]["n_trades_30d"] == 2
    assert summary["FOO"]["buys_shares"]  == 500
    assert summary["FOO"]["sells_shares"] == 1000
    assert summary["FOO"]["net_shares"]   == -500


def test_fetch_insider_no_tickers_returns_empty():
    from notify import ai_panel as mod
    trades, summary = mod.fetch_insider_transactions([])
    assert trades == [] and summary == {}
