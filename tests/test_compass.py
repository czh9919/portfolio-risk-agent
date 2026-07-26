"""
Unit tests for notify/compass.py — deterministic components only.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from notify.compass import (
    DEFAULT_WEIGHTS,
    _band_color,
    append_today,
    compute_compass,
    load_history,
    render_html,
    render_trend_chart_png,
    save_history,
    score_breadth,
    score_credit,
    score_curve,
    score_momentum,
    score_trend,
    score_volatility,
)


def _series(values: list[float], start: str = "2025-08-01") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates)


# ── Trend ─────────────────────────────────────────────────────────────────────

def test_trend_bullish_when_far_above_200sma():
    # 300 samples rising from 100 → 130
    vals = np.linspace(100, 130, 300).tolist()
    score, raw = score_trend(_series(vals))
    assert score > 0.5
    assert raw["golden_cross"] is True
    assert raw["pct_above_200sma"] > 0


def test_trend_bearish_when_below_200sma_and_death_cross():
    vals = np.linspace(130, 90, 300).tolist()
    score, raw = score_trend(_series(vals))
    assert score < -0.5
    assert raw["golden_cross"] is False


def test_trend_zero_when_insufficient_history():
    score, raw = score_trend(_series([100, 101, 102]))
    assert score == 0.0


# ── Breadth ───────────────────────────────────────────────────────────────────

def test_breadth_mostly_above_sma200():
    # 200 tickers, all rising → 100% above 200SMA
    tickers = {}
    for i in range(200):
        tickers[f"T{i}"] = _series(np.linspace(100, 120, 220).tolist())
    score, raw = score_breadth(tickers)
    assert score == 1.0
    assert raw["pct_above_200sma"] == 100.0


def test_breadth_mostly_below_sma200():
    tickers = {}
    for i in range(200):
        tickers[f"T{i}"] = _series(np.linspace(120, 100, 220).tolist())
    score, raw = score_breadth(tickers)
    assert score == -1.0


def test_breadth_empty_returns_zero():
    score, raw = score_breadth({})
    assert score == 0.0


# ── Volatility ────────────────────────────────────────────────────────────────

def test_volatility_low_vix_bullish():
    score, raw = score_volatility(_series([11.5]), None)
    assert score > 0.5


def test_volatility_high_vix_bearish():
    score, raw = score_volatility(_series([30.0]), None)
    assert score < -0.5


def test_volatility_backwardation_penalty():
    # VIX 20, VIX9D 25 (backwardated) — penalise the level score
    baseline, _ = score_volatility(_series([20.0]), None)
    penalised, raw = score_volatility(_series([20.0]), _series([25.0]))
    assert penalised < baseline
    assert raw["backwardation"] is True


# ── Credit ────────────────────────────────────────────────────────────────────

def test_credit_hyg_outperforming_bullish():
    hyg = _series([100.0] * 20 + [104.0])   # +4% over 20d
    ief = _series([100.0] * 20 + [100.5])   # +0.5% over 20d
    score, raw = score_credit(hyg, ief)
    assert score > 0.5


def test_credit_ief_outperforming_bearish():
    hyg = _series([100.0] * 20 + [96.0])
    ief = _series([100.0] * 20 + [102.0])
    score, raw = score_credit(hyg, ief)
    assert score < -0.5


# ── Curve ─────────────────────────────────────────────────────────────────────

def test_curve_inverted_negative_score():
    # 10Y 4.0%, 3M 5.0% → -100bps
    score, raw = score_curve(_series([4.0]), _series([5.0]))
    assert score < 0
    assert raw["inverted"] is True
    assert raw["slope_bps"] == pytest.approx(-100.0)


def test_curve_normal_positive_score():
    score, raw = score_curve(_series([5.0]), _series([3.0]))
    assert score > 0
    assert raw["inverted"] is False
    assert raw["slope_bps"] == pytest.approx(200.0)


# ── Momentum ──────────────────────────────────────────────────────────────────

def test_momentum_strong_uptrend():
    # 6m of gains (~5% every month)
    vals = np.linspace(100, 130, 200).tolist()
    score, raw = score_momentum(_series(vals), None, None)
    assert score > 0.5


def test_momentum_gold_copper_rising_pulls_score_down():
    spy_flat = _series([100.0] * 200)  # zero return
    gold_flat  = _series([2000.0] * 30)
    copper_flat = _series([4.0] * 30)
    baseline, _ = score_momentum(spy_flat, gold_flat, copper_flat)

    # Now gold rising 20% vs copper → safe-haven bid = bearish
    gold_up = _series([2000.0] * 21 + [2400.0])
    copper_flat2 = _series([4.0] * 22)
    with_gc, _ = score_momentum(spy_flat, gold_up, copper_flat2)
    assert with_gc < baseline


# ── Composite ─────────────────────────────────────────────────────────────────

def test_composite_all_bull_gives_positive_100():
    scores = dict.fromkeys(DEFAULT_WEIGHTS.keys(), 1.0)
    out = compute_compass(scores, DEFAULT_WEIGHTS)
    assert out["composite"] == pytest.approx(100.0)
    assert out["label_en"] == "Bull"
    assert out["label_zh"] == "强牛"


def test_composite_all_bear_gives_negative_100():
    scores = dict.fromkeys(DEFAULT_WEIGHTS.keys(), -1.0)
    out = compute_compass(scores, DEFAULT_WEIGHTS)
    assert out["composite"] == pytest.approx(-100.0)
    assert out["label_en"] == "Bear"
    assert out["label_zh"] == "强熊"


def test_composite_neutral_zero():
    scores = dict.fromkeys(DEFAULT_WEIGHTS.keys(), 0.0)
    out = compute_compass(scores, DEFAULT_WEIGHTS)
    assert out["composite"] == 0.0
    assert out["label_en"] == "Neutral"


def test_composite_weighting_respected():
    # Only trend positive, weight=0.20 → composite = 100*0.20 = 20
    scores = {"trend": 1.0, "breadth": 0.0, "vol": 0.0,
              "credit": 0.0, "curve": 0.0, "momentum": 0.0}
    out = compute_compass(scores, DEFAULT_WEIGHTS)
    assert out["composite"] == pytest.approx(20.0)


# ── History persistence ───────────────────────────────────────────────────────

def test_history_dedups_by_date_keeping_last(tmp_path, monkeypatch):
    import notify.compass as mod
    monkeypatch.setattr(mod, "HISTORY_DIR",  tmp_path)
    monkeypatch.setattr(mod, "HISTORY_FILE", tmp_path / "history.json")

    e1 = {"date": "2026-07-25", "composite": 10.0}
    e2 = {"date": "2026-07-26", "composite": 20.0}
    e3 = {"date": "2026-07-26", "composite": 25.0}  # same date as e2 → replace
    save_history([e1, e2, e3], keep_days=180)

    loaded = load_history()
    assert len(loaded) == 2
    assert loaded[-1]["composite"] == 25.0


def test_history_rotates_to_keep_days(tmp_path, monkeypatch):
    import notify.compass as mod
    monkeypatch.setattr(mod, "HISTORY_DIR",  tmp_path)
    monkeypatch.setattr(mod, "HISTORY_FILE", tmp_path / "history.json")

    entries = []
    base = dt.date(2026, 1, 1)
    for i in range(50):
        entries.append({"date": (base + dt.timedelta(days=i)).isoformat(),
                        "composite": float(i)})
    save_history(entries, keep_days=10)

    loaded = load_history()
    assert len(loaded) == 10
    # Should be the newest 10 (i=40..49)
    assert loaded[0]["composite"] == 40.0
    assert loaded[-1]["composite"] == 49.0


def test_append_today_replaces_same_date():
    prev = [{"date": "2026-07-25", "composite": 10},
            {"date": "2026-07-26", "composite": 20}]
    new  = {"date": "2026-07-26", "composite": 30}
    out = append_today(prev, new, keep_days=30)
    dates = [e["date"] for e in out]
    assert dates == ["2026-07-25", "2026-07-26"]
    assert out[-1]["composite"] == 30


# ── Chart PNG ─────────────────────────────────────────────────────────────────

def test_chart_returns_png_bytes():
    history = []
    for i in range(30):
        d = dt.date(2026, 6, 1) + dt.timedelta(days=i)
        history.append({"date": d.isoformat(),
                        "composite": float(np.sin(i / 4) * 40)})
    png = render_trend_chart_png(history, width_in=6, height_in=3, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 500


def test_chart_empty_when_no_history():
    assert render_trend_chart_png([]) == b""
    assert render_trend_chart_png([{"date": "2026-07-26", "composite": 0}]) == b""


# ── HTML rendering ────────────────────────────────────────────────────────────

def test_band_color_bull_bear_extremes():
    bg_bull, _  = _band_color( 0.8)
    bg_bear, _  = _band_color(-0.8)
    bg_neutral,_= _band_color( 0.0)
    assert bg_bull  == "#0a8f39"
    assert bg_bear  == "#8b1a1a"
    assert bg_neutral == "#bfbfbf"


def test_html_renders_all_sections():
    compass = {"composite": 34.5, "label_en": "Mildly Bullish", "label_zh": "偏牛"}
    scores  = {"trend": 0.8, "breadth": 0.3, "vol": 0.1,
               "credit": 0.2, "curve": -0.1, "momentum": 0.5}
    raw     = {"spy": 555.0, "vix": 14.5, "slope_bps": -35.0,
               "pct_above_200sma": 68.0}
    narr    = {"narrative_en": "Balanced regime.",
               "narrative_zh": "均衡状态。"}
    html = render_html(compass, scores, raw, narr, chart_cid="trendcid")

    assert "Bull-Bear Compass" in html and "牛熊罗盘" in html
    assert "+34.5" in html
    assert "Mildly Bullish" in html and "偏牛" in html
    assert "Balanced regime." in html and "均衡状态。" in html
    assert 'src="cid:trendcid"' in html
    # Every factor label present
    for fk in ("Trend", "Breadth", "Volatility", "Credit",
               "Yield Curve", "Momentum"):
        assert fk in html
    for zh in ("趋势", "广度", "波动", "信用", "利率曲线", "动量"):
        assert zh in html


def test_html_omits_chart_when_cid_empty():
    compass = {"composite": 0, "label_en": "Neutral", "label_zh": "中性"}
    html = render_html(compass, dict.fromkeys(DEFAULT_WEIGHTS, 0), {},
                       {"narrative_en": "", "narrative_zh": ""}, chart_cid="")
    assert "cid:" not in html
