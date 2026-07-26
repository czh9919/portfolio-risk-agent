"""
Unit tests for notify/flow_panel.py — deterministic components only.

Network stages (FINRA POST, House Clerk ZIP, PDF download, Anthropic) are
exercised via mocked responses so tests stay fast and offline-safe.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from notify.flow_panel import (
    _fmt_dollar,
    _safe_json,
    analyze_with_llm,
    chart_congress_bar,
    chart_dark_pool_bar,
    combine_universe,
    fetch_dark_pool,
    list_recent_ptr_filings,
    parse_ptr_pdf,
    render_html,
)


# ── Universe combiner ─────────────────────────────────────────────────────────

def test_combine_universe_pulls_from_ai_config(tmp_path, monkeypatch):
    """AI leaders should feed into the universe."""
    # AI_CONFIG_PATH is Path("config/ai_leaders.yaml") — a *relative* path,
    # so switching cwd + writing that file wires it up cleanly.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ai_leaders.yaml").write_text(
        "groups:\n"
        "  chips:\n"
        "    tickers: [NVDA, AMD]\n"
        "  cloud:\n"
        "    tickers: [MSFT]\n",
        encoding="utf-8",
    )

    tickers = combine_universe()
    assert set(tickers) == {"AMD", "MSFT", "NVDA"}
    assert tickers == sorted(tickers)


def test_combine_universe_merges_watchlist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "watchlist.csv").write_text(
        "ticker,weight,name,asset_class,currency\n"
        "AAPL,1.0,Apple,equity,USD\n"
        "TSLA,1.0,Tesla,equity,USD\n",
        encoding="utf-8",
    )

    # Empty AI config so we test the watchlist path in isolation
    from notify import ai_panel as ap
    (tmp_path / "config" / "ai_leaders.yaml").write_text(
        "groups: {}", encoding="utf-8")
    monkeypatch.setattr(ap, "AI_CONFIG_PATH", tmp_path / "config" / "ai_leaders.yaml")

    tickers = combine_universe()
    assert "AAPL" in tickers and "TSLA" in tickers


# ── FINRA dark pool ───────────────────────────────────────────────────────────

def _finra_row(ticker: str, week: str = "2026-07-14",
               shares: int = 100000, notional: int = 50_000_000,
               trades: int = 1500, mpid: str = "UBSA",
               name: str = "UBS ATS"):
    """A single row as returned by FINRA's CSV DictReader (all strings)."""
    return {
        "issueSymbolIdentifier":    ticker,
        "weekStartDate":            week,
        "totalWeeklyShareQuantity": str(shares),
        "totalNotionalSum":         str(notional),
        "totalWeeklyTradeCount":    str(trades),
        "MPID":                     mpid,
        "marketParticipantName":    name,
    }


def test_fetch_dark_pool_aggregates_multi_venue(monkeypatch):
    """Two MPID rows for the same ticker should aggregate."""
    from notify import flow_panel as fp

    monkeypatch.setattr(fp, "_fetch_latest_finra_week", lambda: "2026-07-14")

    tier_rows = {
        "T1": [
            _finra_row("NVDA", shares=100000, notional=50_000_000,
                       trades=1500, mpid="UBSA", name="UBS ATS"),
            _finra_row("NVDA", shares=200000, notional=100_000_000,
                       trades=2000, mpid="MSCO", name="Morgan Stanley"),
            _finra_row("OTHER", shares=999, notional=999,
                       trades=1, mpid="ZZ", name="not tracked"),
        ],
        "T2": [],
    }
    monkeypatch.setattr(fp, "_fetch_finra_week_csv",
                        lambda week, tier: tier_rows.get(tier, []))

    result = fetch_dark_pool(["NVDA"])
    d = result["NVDA"]
    assert d["week_start"] == "2026-07-14"
    assert d["total_shares"]   == 300_000
    assert d["total_notional"] == 150_000_000
    assert d["total_trades"]   == 3_500
    assert d["n_venues"] == 2
    # Top venue by shares should be Morgan Stanley (200k > 100k)
    assert d["top_venues"][0]["mpid"] == "MSCO"


def test_fetch_dark_pool_empty_when_no_week(monkeypatch):
    from notify import flow_panel as fp
    monkeypatch.setattr(fp, "_fetch_latest_finra_week", lambda: None)
    r = fetch_dark_pool(["NVDA"])
    assert r == {"NVDA": {}}


def test_fetch_dark_pool_ticker_missing_from_bulk(monkeypatch):
    """A ticker with no FINRA row that week gets an empty {} entry."""
    from notify import flow_panel as fp
    monkeypatch.setattr(fp, "_fetch_latest_finra_week", lambda: "2026-07-14")
    monkeypatch.setattr(fp, "_fetch_finra_week_csv",
                        lambda week, tier: [_finra_row("NVDA")])
    r = fetch_dark_pool(["NVDA", "GHOST"])
    assert r["NVDA"]["total_shares"] > 0
    assert r["GHOST"] == {}


# ── House Clerk XML → recent PTR filter ───────────────────────────────────────

def _make_zip(xml_content: str, year: int) -> bytes:
    """Wrap the XML in a ZIP that matches House Clerk's format."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{year}FD.xml", xml_content)
        zf.writestr(f"{year}FD.txt", "placeholder")
    return buf.getvalue()


def test_list_recent_ptr_filters_by_type_and_date(monkeypatch):
    """Only FilingType='P' within the lookback window should be returned."""
    today = dt.date.today()
    recent = today - dt.timedelta(days=5)
    old    = today - dt.timedelta(days=90)

    def _fmt(d): return f"{d.month}/{d.day}/{d.year}"

    xml_this_year = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<FinancialDisclosure>'
        f'<Member><Prefix>Hon.</Prefix><Last>Alford</Last><First>Mark</First>'
        f'<Suffix/><FilingType>P</FilingType><StateDst>MO04</StateDst>'
        f'<Year>{today.year}</Year><FilingDate>{_fmt(recent)}</FilingDate>'
        f'<DocID>20034201</DocID></Member>'
        f'<Member><Prefix/><Last>Skipped</Last><First>C</First>'
        f'<Suffix/><FilingType>C</FilingType><StateDst>XX01</StateDst>'
        f'<Year>{today.year}</Year><FilingDate>{_fmt(recent)}</FilingDate>'
        f'<DocID>99999999</DocID></Member>'  # Candidate filing, not PTR
        f'<Member><Prefix>Hon.</Prefix><Last>Old</Last><First>Peter</First>'
        f'<Suffix/><FilingType>P</FilingType><StateDst>NY01</StateDst>'
        f'<Year>{today.year}</Year><FilingDate>{_fmt(old)}</FilingDate>'
        f'<DocID>20030000</DocID></Member>'  # PTR but too old
        '</FinancialDisclosure>'
    )
    zip_this_year = _make_zip(xml_this_year, today.year)

    from notify import flow_panel as fp

    def _fake_urlopen(*args, **kwargs):
        class FakeResp:
            def read(self): return zip_this_year
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return FakeResp()

    monkeypatch.setattr(fp.urllib.request, "urlopen", _fake_urlopen)

    filings = list_recent_ptr_filings(days_back=30)
    # Candidate + old-PTR should be dropped; only recent PTR remains.
    # (list_recent_ptr calls both current + prior year — we're returning the
    # same fake response, so both add the fresh row. Filter for uniques.)
    docids = {f["docid"] for f in filings}
    assert "20034201" in docids
    assert "99999999" not in docids   # wrong FilingType
    assert "20030000" not in docids   # outside 30-day window


# ── PDF parse regex ───────────────────────────────────────────────────────────

def test_parse_ptr_pdf_extracts_trades(monkeypatch):
    """Real Alford-style text should yield transactions with ticker + type + date + amount."""
    fake_text = (
        "Filing ID #20034201\n"
        "Name: Hon. Mark Alford\n"
        "Amazon.com, Inc. - Common Stock (AMZN) S (partial) "
        "03/16/2026 03/16/2026 $1,001 - $15,000\n"
        "Apple Inc. - Common Stock (AAPL) S (partial) "
        "03/16/2026 03/16/2026 $1,001 - $15,000\n"
        "Berkshire Hathaway Inc. New Common Stock (BRK.B) P "
        "03/15/2026 03/15/2026 $15,001 - $50,000\n"
    )

    class FakePage:
        def extract_text(self): return fake_text
    class FakePdf:
        pages = [FakePage()]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_open(_buf):
        return FakePdf()

    fake_pdfplumber = MagicMock()
    fake_pdfplumber.open = _fake_open
    with patch.dict("sys.modules", {"pdfplumber": fake_pdfplumber}):
        txns = parse_ptr_pdf(b"unused")

    tickers = [t["ticker"] for t in txns]
    types   = [t["type"]   for t in txns]
    assert "AMZN"  in tickers
    assert "AAPL"  in tickers
    assert "BRK.B" in tickers

    # Both AMZN + AAPL are Sale (partial)
    for t in txns:
        if t["ticker"] == "AMZN":
            assert t["type"] == "sell"
            assert t["transaction_date"] == "03/16/2026"
            assert t["amount_low"]  == 1_001
            assert t["amount_high"] == 15_000
        if t["ticker"] == "BRK.B":
            # 'P' code from regex → buy
            assert t["type"] == "buy"
            assert t["amount_low"]  == 15_001
            assert t["amount_high"] == 50_000


def test_parse_ptr_pdf_returns_empty_when_no_text(monkeypatch):
    class FakePage:
        def extract_text(self): return ""
    class FakePdf:
        pages = [FakePage()]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake_pdfplumber = MagicMock()
    fake_pdfplumber.open = lambda buf: FakePdf()
    with patch.dict("sys.modules", {"pdfplumber": fake_pdfplumber}):
        txns = parse_ptr_pdf(b"unused")
    assert txns == []


# ── LLM narrative ─────────────────────────────────────────────────────────────

def test_llm_fallback_when_no_key():
    dark = {"NVDA": {"total_notional": 5_000_000_000, "total_shares": 20_000_000,
                     "week_start": "2026-07-14", "n_venues": 15}}
    trades = [{"ticker": "PLTR", "type": "buy", "transaction_date": "2026-07-20",
               "amount_low": 100_000, "amount_high": 250_000,
               "filer": "Some Filer", "state_dist": "XX01",
               "filing_date": "2026-07-22"}]
    out = analyze_with_llm(dark, trades, api_key="")
    assert "NVDA" in out["narrative_en"] or "$5" in out["narrative_en"]
    assert "30" in out["narrative_en"] or "1" in out["narrative_en"]


def test_llm_json_parses(monkeypatch):
    import sys
    payload = {"narrative_en": "Flow steady.", "narrative_zh": "资金流稳定。"}
    fake_block = MagicMock(); fake_block.text = json.dumps(payload)
    fake_msg   = MagicMock(); fake_msg.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_module = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    out = analyze_with_llm({}, [], api_key="dummy")
    assert out["narrative_en"] == "Flow steady."
    assert out["narrative_zh"] == "资金流稳定。"


def test_safe_json_extracts_via_regex_on_broken_quotes():
    bad = ('{"narrative_en": "Congress bought "aggressively" this week.", '
           '"narrative_zh": "国会本周积极买入。"}')
    out = _safe_json(bad)
    assert out is not None and "国会" in out["narrative_zh"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_fmt_dollar_scales():
    assert _fmt_dollar(None)      == "—"
    assert _fmt_dollar(500)       == "$500"
    assert _fmt_dollar(1500)      == "$2K"
    assert _fmt_dollar(2_500_000) == "$2.5M"
    assert _fmt_dollar(5.4e9)     == "$5.40B"


# ── Charts ────────────────────────────────────────────────────────────────────

def test_chart_dark_pool_empty_returns_empty_bytes():
    assert chart_dark_pool_bar({}) == b""
    assert chart_dark_pool_bar({"NVDA": {}}) == b""


def test_chart_dark_pool_returns_png_bytes():
    dark = {
        "NVDA": {"total_notional": 5_000_000_000, "total_shares": 20_000_000,
                 "week_start": "2026-07-14", "top_venues": [], "n_venues": 3,
                 "total_trades": 100_000},
        "MSFT": {"total_notional": 2_000_000_000, "total_shares": 10_000_000,
                 "week_start": "2026-07-14", "top_venues": [], "n_venues": 2,
                 "total_trades": 50_000},
    }
    png = chart_dark_pool_bar(dark)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000


def test_chart_congress_returns_png_bytes_when_trades():
    trades = [
        {"ticker": "NVDA", "type": "buy",  "amount_low": 100_000,
         "amount_high": 250_000, "transaction_date": "2026-07-20",
         "filer": "F1", "state_dist": "XX01"},
        {"ticker": "MSFT", "type": "sell", "amount_low": 15_001,
         "amount_high": 50_000, "transaction_date": "2026-07-18",
         "filer": "F2", "state_dist": "YY01"},
    ]
    png = chart_congress_bar(trades)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_congress_empty_when_no_trades():
    assert chart_congress_bar([]) == b""


# ── HTML rendering ────────────────────────────────────────────────────────────

def test_html_contains_all_sections():
    universe = ["NVDA", "MSFT", "AMZN"]
    dark = {
        "NVDA": {"total_notional": 5_000_000_000, "total_shares": 20_000_000,
                 "week_start": "2026-07-14", "total_trades": 100_000,
                 "top_venues": [{"mpid": "UBSA", "name": "UBS ATS", "shares": 10_000_000}],
                 "n_venues": 15}
    }
    trades = [{
        "ticker": "MSFT", "asset_name": "Microsoft Corp",
        "type": "sell", "transaction_date": "2026-07-20",
        "notification_date": "2026-07-22",
        "amount_low": 15_001, "amount_high": 50_000,
        "filer": "Hon. Mark Alford", "state_dist": "MO04",
        "filing_date": "2026-07-22", "docid": "20034201", "year": 2026,
    }]
    narr = {"narrative_en": "Flow test.", "narrative_zh": "资金流测试。"}
    cids = {"dark_pool": "flow_dark_pool", "congress": "flow_congress"}

    html = render_html(universe, dark, trades, narr, cids)

    # Header + bilingual title
    assert "Flow Panel" in html and "资金流面板" in html
    # Narrative
    assert "Flow test." in html and "资金流测试。" in html
    # Chart imgs
    assert 'src="cid:flow_dark_pool"' in html
    assert 'src="cid:flow_congress"' in html
    # Dark pool table has NVDA + weekly notional
    assert "NVDA" in html
    assert "$5.00B" in html
    assert "UBS ATS" in html
    # Congress table
    assert "Hon. Mark Alford" in html
    assert "MO04" in html
    assert "20034201" in html
    assert "sell" in html.lower() or "SELL" in html
    # Link to raw PDF
    assert "20034201.pdf" in html


def test_html_placeholders_when_empty():
    html = render_html([], {}, [], {"narrative_en": "", "narrative_zh": ""}, {})
    assert ("No dark-pool data" in html
            or "跟踪范围内暂无暗池数据" in html)
    assert ("No congressional trades" in html
            or "过去 30 天跟踪范围内无国会交易" in html)
