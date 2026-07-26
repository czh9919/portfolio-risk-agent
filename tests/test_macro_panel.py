"""
Unit tests for notify/macro_panel.py — deterministic components only.

Network-dependent stages (FRED, CME, yfinance, Anthropic) are exercised
via mocked responses so tests stay fast and offline-safe.
"""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from notify.macro_panel import (
    FOMC_MEETINGS_2026_2027,
    FUTURES_MONTH_CODES,
    _fmt_value,
    _zq_contract,
    latest_value,
    narrate_with_llm,
    render_html,
    summarize_latest,
    transform_qoq_saar,
    transform_yoy,
)


# ── Transforms ────────────────────────────────────────────────────────────────

def test_yoy_from_price_index():
    # Index doubles year-over-year
    idx = pd.date_range("2024-01-01", periods=24, freq="MS")
    vals = list(range(100, 112)) + list(range(200, 212))  # first year → second year
    s = pd.Series(vals, index=idx)
    yoy = transform_yoy(s)
    assert yoy is not None
    # Last value = 211/111 - 1 ≈ 90.09%
    assert yoy.iloc[-1] == pytest.approx(211 / 111 * 100 - 100, rel=1e-3)


def test_yoy_returns_none_when_insufficient_data():
    idx = pd.date_range("2026-01-01", periods=6, freq="MS")
    s = pd.Series([100, 101, 102, 103, 104, 105], index=idx)
    assert transform_yoy(s) is None


def test_qoq_saar():
    # 1% QoQ growth compounded to annual = (1.01)^4 - 1 ≈ 4.06%
    idx = pd.date_range("2025-01-01", periods=8, freq="QS")
    vals = [100 * (1.01 ** i) for i in range(8)]
    s = pd.Series(vals, index=idx)
    saar = transform_qoq_saar(s)
    assert saar is not None
    assert saar.iloc[-1] == pytest.approx((1.01 ** 4 - 1) * 100, rel=1e-3)


def test_latest_value_skips_nans():
    idx = pd.date_range("2026-06-01", periods=4, freq="D")
    s = pd.Series([1.0, 2.0, np.nan, np.nan], index=idx)
    v, d = latest_value(s)
    assert v == 2.0
    assert d == idx[1].date()


def test_latest_value_empty():
    assert latest_value(pd.Series(dtype=float)) == (None, None)
    assert latest_value(None) == (None, None)


# ── ZQ futures ────────────────────────────────────────────────────────────────

def test_zq_contract_symbol():
    # 2026-09 → ZQU26.CBT (U = September)
    assert _zq_contract(2026, 9)  == "ZQU26.CBT"
    assert _zq_contract(2027, 1)  == "ZQF27.CBT"
    assert _zq_contract(2026, 12) == "ZQZ26.CBT"


def test_futures_month_codes_all_12():
    assert len(FUTURES_MONTH_CODES) == 12
    assert set(FUTURES_MONTH_CODES.keys()) == set(range(1, 13))


def test_fomc_calendar_populated():
    """Sanity: at least one upcoming meeting on the calendar; codes are dates."""
    upcoming = [m for m in FOMC_MEETINGS_2026_2027 if m >= dt.date(2026, 1, 1)]
    assert len(upcoming) >= 4


# ── ZQ implied rate fetch (with mocked yfinance) ──────────────────────────────

def test_zq_implied_rates_computes_delta_and_direction(monkeypatch):
    """Mock a yfinance Ticker returning a known price → verify implied
    rate and delta/direction calculation."""
    from notify import macro_panel as mp

    class FakeHistory:
        def __init__(self, closes):
            self.data = closes

        def __len__(self):
            return len(self.data)

        @property
        def loc(self):
            return None

        def __getitem__(self, k):
            return pd.Series(self.data)

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, period=None):
            # Price 96.50 → implied 3.50%
            return pd.DataFrame({"Close": [96.50, 96.51, 96.50]})

    monkeypatch.setattr(mp, "FOMC_MEETINGS_2026_2027",
                        [dt.date.today() + dt.timedelta(days=30)])

    with patch("yfinance.Ticker", FakeTicker):
        # current EFFR = 3.63%, next meeting implied 3.50% → -13 bps → cut
        rates = mp.fetch_zq_implied_rates(current_effr=3.63, n_meetings=1)

    assert len(rates) == 1
    r = rates[0]
    assert r["implied_rate_pct"] == pytest.approx(3.50)
    assert r["delta_bps"] == pytest.approx(-13.0, abs=0.1)
    assert r["direction_en"] == "cut"
    assert r["direction_zh"] == "降息"
    # Binary 25bps model: |13|/25 * 100 = 52%
    assert r["probability_pct"] == pytest.approx(52.0, abs=1)


# ── CME fallback ──────────────────────────────────────────────────────────────

def test_fetch_cme_returns_empty_on_403(monkeypatch):
    """CME's real endpoint 403s us; the helper must degrade to []."""
    from notify import macro_panel as mp
    import urllib.request

    def raise_403(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://cme...", 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_403)
    assert mp.fetch_cme_fedwatch() == []


# ── Snapshot summarizer ───────────────────────────────────────────────────────

def _sample_fred_df() -> pd.DataFrame:
    """Fake FRED-shaped dataframe with just enough to run summarize_latest."""
    idx = pd.date_range("2024-01-01", periods=30, freq="MS")
    data = {
        "PAYEMS":   np.linspace(155_000, 160_000, 30),
        "UNRATE":   np.linspace(4.5, 4.2, 30),
        "PCEPILFE": np.linspace(120, 130, 30),   # index → +8.3% YoY-ish
        "DFEDTARU": [3.75] * 30,
        "DFEDTARL": [3.50] * 30,
        "FEDFUNDS": [3.63] * 30,
        "GDPC1":    np.linspace(22_000, 24_000, 30),
        "NFCI":     np.linspace(-0.3, -0.5, 30),
        "T10Y2Y":   np.linspace(0.1, 0.4, 30),
        "T10Y3M":   np.linspace(0.3, 0.7, 30),
    }
    return pd.DataFrame(data, index=idx)


def test_summarize_latest_covers_all_available_series():
    df = _sample_fred_df()
    snap = summarize_latest(df)

    # Should include what we passed in
    assert "PAYEMS"   in snap
    assert "UNRATE"   in snap
    assert "PCEPILFE" in snap  # YoY-transformed
    assert "FEDFUNDS" in snap

    # YoY-transformed value should be positive (we drew a rising index)
    assert snap["PCEPILFE"]["value"] > 0
    assert snap["PCEPILFE"]["transform"] == "yoy"

    # Levels preserved as-is
    assert snap["UNRATE"]["value"] == pytest.approx(4.2, abs=0.01)
    assert snap["FEDFUNDS"]["value"] == pytest.approx(3.63)


def test_summarize_latest_skips_missing_columns():
    df = _sample_fred_df().drop(columns=["PAYEMS", "UNRATE"])
    snap = summarize_latest(df)
    assert "PAYEMS" not in snap
    assert "UNRATE" not in snap
    assert "FEDFUNDS" in snap  # others still there


# ── Formatting helpers ────────────────────────────────────────────────────────

def test_fmt_value_yoy_shows_sign_and_pct():
    assert _fmt_value("PCEPILFE",  2.5, "yoy")      == "+2.50%"
    assert _fmt_value("PCEPILFE", -1.2, "yoy")      == "-1.20%"


def test_fmt_value_level_no_pct_for_labour_counts():
    assert _fmt_value("PAYEMS", 158_984.0, "level") == "158,984"
    assert _fmt_value("ICSA",   187_000.0, "level") == "187,000"


def test_fmt_value_percent_for_rates():
    assert _fmt_value("UNRATE",   4.2, "level") == "4.20%"
    assert _fmt_value("FEDFUNDS", 3.63, "level") == "3.63%"


def test_fmt_value_signed_spread():
    assert _fmt_value("T10Y2Y",  0.36, "level") == "+0.36"
    assert _fmt_value("T10Y3M", -0.10, "level") == "-0.10"


# ── LLM narrative ─────────────────────────────────────────────────────────────

def test_llm_fallback_when_no_key():
    snapshot = {
        "FEDFUNDS": {"value": 3.63, "date": "2026-06-01",
                     "transform": "level", "label_en": "EFFR", "label_zh": "有效利率"},
        "DFEDTARL": {"value": 3.50, "date": "2026-07-25",
                     "transform": "level", "label_en": "Lower", "label_zh": "下限"},
        "DFEDTARU": {"value": 3.75, "date": "2026-07-25",
                     "transform": "level", "label_en": "Upper", "label_zh": "上限"},
    }
    out = narrate_with_llm(snapshot, [], api_key="")
    assert "3.63" in out["narrative_en"]
    assert "3.63" in out["narrative_zh"]
    assert "3.50" in out["narrative_en"]


def test_llm_parses_json_output(monkeypatch):
    import sys
    payload = {
        "narrative_en": "Fed pauses in Q3 with dovish tilt.",
        "narrative_zh": "联储三季度维持，倾向鸽派。",
    }
    fake_block = MagicMock(); fake_block.text = json.dumps(payload)
    fake_msg   = MagicMock(); fake_msg.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_module = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = narrate_with_llm({"FEDFUNDS": {"value": 3.63, "date": "2026-06-01",
                                          "transform": "level", "label_en": "x",
                                          "label_zh": "y"}},
                           [], api_key="dummy")
    assert out["narrative_en"] == "Fed pauses in Q3 with dovish tilt."
    assert out["narrative_zh"] == "联储三季度维持，倾向鸽派。"


# ── HTML rendering ────────────────────────────────────────────────────────────

def test_html_contains_all_sections():
    snapshot = {
        "FEDFUNDS": {"value": 3.63, "date": "2026-06-01",
                     "transform": "level", "label_en": "EFFR",
                     "label_zh": "有效联邦基金利率"},
        "PCEPILFE": {"value": 2.68, "date": "2026-05-01",
                     "transform": "yoy",   "label_en": "Core PCE",
                     "label_zh": "核心PCE"},
        "UNRATE":   {"value": 4.2,  "date": "2026-06-01",
                     "transform": "level", "label_en": "Unemployment",
                     "label_zh": "失业率"},
    }
    fedwatch = [{"date": "2026-09-16", "days_away": 52,
                 "contract": "ZQU26.CBT", "implied_rate_pct": 3.50,
                 "delta_bps": -13, "direction_en": "cut", "direction_zh": "降息",
                 "probability_pct": 52.0}]
    narr = {"narrative_en": "Dovish stance.", "narrative_zh": "偏鸽派。"}
    chart_cids = {"employment": "macro_employment", "inflation": "macro_inflation"}

    html = render_html(snapshot, fedwatch, narr, chart_cids)

    # Header
    assert "Macro Panel" in html and "宏观数据面板" in html
    # Narrative
    assert "Dovish stance." in html and "偏鸽派。" in html
    # Snapshot values
    assert "3.63%" in html    # FEDFUNDS
    assert "+2.68%" in html   # Core PCE YoY
    assert "4.20%" in html    # UNRATE
    # FedWatch table
    assert "2026-09-16" in html
    assert "-13bps" in html
    assert "cut" in html and "降息" in html
    # Charts embedded via CID
    assert 'src="cid:macro_employment"' in html
    assert 'src="cid:macro_inflation"' in html


def test_html_omits_fedwatch_when_empty():
    html = render_html(
        snapshot={},
        fedwatch=[],
        narrative={"narrative_en": "", "narrative_zh": ""},
        chart_cids={},
    )
    assert "MARKET-IMPLIED FOMC PATH" not in html
