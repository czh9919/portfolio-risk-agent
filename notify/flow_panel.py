"""
Flow Panel — Congressional trades + dark pool activity.

Fires at 23:00 UTC weekdays via .github/workflows/flow_panel.yml
(RUN_MODE=flow_panel). Pipeline:

  1. Build tracked universe = union(AI leaders, watchlist, holdings tickers)
  2. Dark Pool: for each tracked ticker, POST to FINRA ATS weeklySummary
     API (official, free, 2-week delay) — aggregate weekly share volume,
     notional value, trade count, per-venue MPID breakdown
  3. Congressional: download {current_year}FD.xml + {prev_year}FD.xml
     from House Clerk (~100-800KB), filter to FilingType='P' (Periodic
     Transaction Report), keep last 30 days. For each PTR, download the
     PDF (small ~50-100KB), extract asset+ticker+date+amount via
     pdfplumber + regex, filter to tracked universe. Caches parsed
     results to cache/ptr_parsed/{docid}.json so subsequent runs skip
     already-processed filings.
  4. Sonnet 4.6 analyses both flows for signal (unusual dark-pool
     ratios, notable congressional buys/sells)
  5. Bilingual HTML email with two tables + two PNG charts

Data sources — all free and official:
  - FINRA: https://api.finra.org/data/group/otcMarket/name/weeklySummary
  - House Clerk: https://disclosures-clerk.house.gov/public_disc/*

Fails gracefully — a dead FINRA endpoint or an unreadable PDF each
degrade one section, never blocks delivery.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

LLM_MODEL       = "claude-sonnet-4-6"
USER_AGENT      = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36")

# House Clerk endpoints
_HOUSE_ZIP_URL  = ("https://disclosures-clerk.house.gov/public_disc/"
                   "financial-pdfs/{year}FD.ZIP")
_HOUSE_PDF_URL  = ("https://disclosures-clerk.house.gov/public_disc/"
                   "ptr-pdfs/{year}/{docid}.pdf")

# FINRA ATS — two endpoints:
#   - weeklyDownloadDetails: index of published weeks (metadata only)
#   - weeklySummary: bulk CSV of per-ticker-per-venue ATS activity, filtered
#     by weekStartDate + tierIdentifier (the partition fields); returns CSV
#     text rather than JSON. Returns all NMS Tier 1/2 tickers in that week.
_FINRA_INDEX = ("https://api.finra.org/data/group/otcMarket/name/"
                "weeklyDownloadDetails")
_FINRA_DATA  = ("https://api.finra.org/data/group/otcMarket/name/"
                "weeklySummary")

# ATS Tiers we care about (Tier 1 = large-cap NMS, Tier 2 = smaller NMS)
_FINRA_TIERS = ["T1", "T2"]

# Cache dirs
CACHE_DIR       = Path("cache")
PTR_CACHE_DIR   = CACHE_DIR / "ptr_parsed"


# ── Universe combiner ─────────────────────────────────────────────────────────

def combine_universe() -> list[str]:
    """AI leaders + watchlist + committed holdings tickers, uniq + sorted."""
    tickers: set[str] = set()

    # AI leaders
    try:
        from notify.ai_panel import load_ai_config, all_tickers as ai_all
        tickers.update(ai_all(load_ai_config()))
    except Exception as e:
        logger.warning(f"FlowPanel: AI config load failed — {e}")

    # Watchlist (config/watchlist.csv)
    try:
        import csv
        with open("config/watchlist.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tk = (row.get("ticker") or "").strip().upper()
                if tk:
                    tickers.add(tk)
    except Exception as e:
        logger.debug(f"FlowPanel: watchlist read failed — {e}")

    # Committed holdings (from paper_state / latest snapshot if available)
    try:
        import json as _json
        snap = Path("data/paper_state/latest_snapshot.json")
        if snap.exists():
            data = _json.loads(snap.read_text(encoding="utf-8"))
            for pos in data.get("positions", []):
                tk = (pos.get("ticker") or "").strip().upper()
                if tk:
                    tickers.add(tk)
    except Exception as e:
        logger.debug(f"FlowPanel: paper_state read failed — {e}")

    return sorted(tickers)


# ── FINRA dark pool ───────────────────────────────────────────────────────────

def _fetch_latest_finra_week() -> str | None:
    """Query the weeklyDownloadDetails INDEX for the latest published week
    (there's a ~2 week lag from real-time)."""
    body = {"limit": 10}
    try:
        req = urllib.request.Request(
            _FINRA_INDEX, data=json.dumps(body).encode("utf-8"),
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            rows = json.loads(r.read())
        weeks = [row.get("weekStartDate") for row in rows if row.get("weekStartDate")]
        return max(weeks) if weeks else None
    except Exception as e:
        logger.warning(f"FlowPanel: FINRA index failed — {e}")
        return None


def _fetch_finra_week_csv(week: str, tier: str) -> list[dict]:
    """Bulk pull one (week, tier) partition of ATS data as CSV, return rows.

    Response is CSV, not JSON. Rows contain per-ticker-per-MPID share
    quantity + notional + trade count.
    """
    body = {
        "limit": 20000,
        "compareFilters": [
            {"compareType": "EQUAL", "fieldName": "weekStartDate",
             "fieldValue": week},
            {"compareType": "EQUAL", "fieldName": "tierIdentifier",
             "fieldValue": tier},
        ],
    }
    try:
        req = urllib.request.Request(
            _FINRA_DATA, data=json.dumps(body).encode("utf-8"),
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
    except Exception as e:
        logger.warning(f"FlowPanel: FINRA {tier}/{week} bulk failed — {e}")
        return []

    import csv
    from io import StringIO
    rows: list[dict] = []
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        rows.append(row)
    return rows


def fetch_dark_pool(tickers: list[str]) -> dict[str, dict]:
    """Aggregate per-ticker dark pool metrics from the latest FINRA week.

    Downloads one bulk CSV per (week × tier) partition (~1 MB each) rather
    than per-ticker queries, which is both faster and gives us data the
    per-ticker filter can't produce.

    Returns {ticker: {week_start, total_shares, total_notional,
                      total_trades, top_venues, n_venues}}.
    """
    week = _fetch_latest_finra_week()
    if not week:
        return {tk: {} for tk in tickers}

    ticker_set = set(tickers)
    # {ticker: [row, row, ...]}
    bucket: dict[str, list[dict]] = {tk: [] for tk in tickers}

    for tier in _FINRA_TIERS:
        rows = _fetch_finra_week_csv(week, tier)
        for row in rows:
            tk = row.get("issueSymbolIdentifier")
            if tk in ticker_set:
                bucket[tk].append(row)

    out: dict[str, dict] = {}
    for tk, rows in bucket.items():
        if not rows:
            out[tk] = {}
            continue
        total_shares = sum(int(float(r.get("totalWeeklyShareQuantity") or 0))
                           for r in rows)
        total_notional = sum(int(float(r.get("totalNotionalSum") or 0))
                             for r in rows)
        total_trades = sum(int(float(r.get("totalWeeklyTradeCount") or 0))
                           for r in rows)
        venues = [{
            "mpid":   r.get("MPID"),
            "name":   r.get("marketParticipantName", ""),
            "shares": int(float(r.get("totalWeeklyShareQuantity") or 0)),
        } for r in rows if r.get("MPID")]
        venues.sort(key=lambda v: v["shares"], reverse=True)
        out[tk] = {
            "week_start":     week,
            "total_shares":   total_shares,
            "total_notional": total_notional,
            "total_trades":   total_trades,
            "top_venues":     venues[:3],
            "n_venues":       len(venues),
        }
    return out


# ── House Clerk filings ───────────────────────────────────────────────────────

def _fetch_house_xml(year: int) -> ET.Element | None:
    """Download the annual FD.ZIP and return the parsed root element."""
    url = _HOUSE_ZIP_URL.format(year=year)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
    except Exception as e:
        logger.warning(f"FlowPanel: {year}FD.ZIP fetch failed — {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            xml_name = next(n for n in zf.namelist() if n.endswith(".xml"))
            return ET.fromstring(zf.read(xml_name))
    except Exception as e:
        logger.warning(f"FlowPanel: {year}FD.ZIP parse failed — {e}")
        return None


def list_recent_ptr_filings(days_back: int = 30) -> list[dict]:
    """Return PTR filings from the last `days_back` days across current and
    prior year XML dumps. Each entry: {docid, year, filer, state_dist, date}.
    """
    today = dt.date.today()
    cutoff = today - dt.timedelta(days=days_back)
    filings: list[dict] = []

    for year in (today.year, today.year - 1):
        root = _fetch_house_xml(year)
        if root is None:
            continue

        for member in root:
            filing_type = (member.findtext("FilingType") or "").strip()
            if filing_type != "P":
                continue
            fd_str = (member.findtext("FilingDate") or "").strip()
            try:
                # Format is M/D/YYYY or MM/DD/YYYY
                m, d, y = fd_str.split("/")
                filing_date = dt.date(int(y), int(m), int(d))
            except (ValueError, AttributeError):
                continue
            if filing_date < cutoff:
                continue

            filings.append({
                "docid":       (member.findtext("DocID") or "").strip(),
                "year":        int((member.findtext("Year") or year)),
                "filer":       " ".join(
                    (member.findtext(f) or "").strip()
                    for f in ("Prefix", "First", "Last", "Suffix")
                ).strip(),
                "state_dist":  (member.findtext("StateDst") or "").strip(),
                "filing_date": filing_date.isoformat(),
            })
    filings.sort(key=lambda f: f["filing_date"], reverse=True)
    return filings


# ── PDF parsing ───────────────────────────────────────────────────────────────

# Regex: `Asset Name (TICKER) [Category] Type Date Notification Amount`
# Example row we've seen:
#   "Amazon.com, Inc. - Common Stock S (partial) 03/16/2026 03/16/2026 $1,001 - $15,000"
#   "(AMZN) [ST]"
_PDF_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,5})\)")
_PDF_DATE_RE   = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")
_PDF_AMOUNT_RE = re.compile(r"\$([\d,]+)(?:\s*-\s*\$([\d,]+))?")
_PDF_TYPE_RE   = re.compile(
    r"\b(P|S|E)\s*\(partial\)|\b(P|S|E|Purchase|Sale|Exchange)\b(?!\S)",
    re.IGNORECASE,
)


def _fetch_pdf(year: int, docid: str) -> bytes | None:
    url = _HOUSE_PDF_URL.format(year=year, docid=docid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.read()
    except Exception as e:
        logger.debug(f"FlowPanel: PDF {docid} fetch failed — {e}")
        return None


def parse_ptr_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract transactions from a PTR PDF via pdfplumber.

    Returns list of {ticker, asset_name, type, transaction_date,
                     notification_date, amount_low, amount_high, raw_line}
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("FlowPanel: pdfplumber not installed — skipping PDF parse")
        return []

    txns: list[dict] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        logger.debug(f"FlowPanel: pdfplumber open failed — {e}")
        return []

    if not all_text:
        return []

    # PTR PDFs list one transaction per row (line) inside a page-spanning
    # table. Each ticker match anchors us; walk out from there.
    lines = all_text.splitlines()
    for i, line in enumerate(lines):
        m = _PDF_TICKER_RE.search(line)
        if not m:
            continue
        ticker = m.group(1)

        # Combine current line + next 2 lines to cover wrapped rows
        context = " ".join(lines[i:i + 3])

        # Dates: first two are transaction / notification
        date_matches = _PDF_DATE_RE.findall(context)
        if len(date_matches) < 1:
            continue

        # Amount bracket
        amt = _PDF_AMOUNT_RE.search(context)
        low = high = None
        if amt:
            low = int(amt.group(1).replace(",", ""))
            if amt.group(2):
                high = int(amt.group(2).replace(",", ""))
            else:
                high = low

        # Type: 'P'/'S'/'E' with optional (partial). Look at the fragment
        # between the ticker and the first date.
        tx_type = "?"
        head_frag = line[m.end():]
        m2 = _PDF_TYPE_RE.search(head_frag)
        if m2:
            raw = (m2.group(0) or "").strip()
            if "s" in raw.lower():
                tx_type = "sell"
            elif "p" in raw.lower() or "purchase" in raw.lower():
                tx_type = "buy"
            elif "e" in raw.lower() or "exchange" in raw.lower():
                tx_type = "exchange"

        asset_name = line[:m.start()].strip().rstrip("-").strip()
        txns.append({
            "ticker":            ticker,
            "asset_name":        asset_name,
            "type":              tx_type,
            "transaction_date":  date_matches[0] if date_matches else "",
            "notification_date": date_matches[1] if len(date_matches) > 1 else "",
            "amount_low":        low,
            "amount_high":       high,
        })
    return txns


def fetch_congressional_trades(
    universe: list[str], days_back: int = 30,
) -> list[dict]:
    """For each recent PTR filing, download + parse the PDF, filter
    transactions to the universe. Each result carries the filer info.
    Uses cache/ptr_parsed/{docid}.json to skip already-processed filings.
    """
    filings = list_recent_ptr_filings(days_back=days_back)
    logger.info(f"FlowPanel: {len(filings)} recent PTRs found")

    PTR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    universe_set = set(universe)

    out: list[dict] = []
    for f in filings:
        docid = f["docid"]
        cache_path = PTR_CACHE_DIR / f"{docid}.json"

        if cache_path.exists():
            try:
                txns = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                txns = None
        else:
            txns = None

        if txns is None:
            pdf = _fetch_pdf(f["year"], docid)
            if pdf is None:
                continue
            txns = parse_ptr_pdf(pdf)
            try:
                cache_path.write_text(
                    json.dumps(txns, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        for t in txns:
            if t["ticker"] not in universe_set:
                continue
            out.append({**t, "filer":       f["filer"],
                             "state_dist":  f["state_dist"],
                             "filing_date": f["filing_date"],
                             "docid":       docid,
                             "year":        f["year"]})

    # Newest transaction first
    def _sort_key(t):
        try:
            m, d, y = t["transaction_date"].split("/")
            return (int(y), int(m), int(d))
        except (ValueError, AttributeError):
            return (0, 0, 0)
    out.sort(key=_sort_key, reverse=True)
    return out


# ── Charts ────────────────────────────────────────────────────────────────────

def _mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _to_png(fig, dpi: int = 130) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def chart_dark_pool_bar(dark: dict, top_n: int = 15) -> bytes:
    """Horizontal bar of dark-pool notional value by ticker, top N."""
    plt = _mpl_setup()
    import matplotlib.pyplot as plt

    items = [(tk, d.get("total_notional", 0)) for tk, d in dark.items()
             if d.get("total_notional", 0) > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_n]
    if not items:
        return b""

    items.reverse()   # largest at top
    tickers, values = zip(*items)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(items))), dpi=130)
    ax.barh(range(len(items)), values, color="#2c3e50", edgecolor="#1a1a1a",
            linewidth=0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(tickers, fontsize=9)
    ax.set_xlabel("Weekly Dark-Pool Notional ($)", fontsize=10)
    ax.set_title(
        "Dark-Pool Notional  (latest FINRA week, tracked universe)",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.grid(True, alpha=0.15, axis="x")

    def _fmt_x(x, _pos):
        if x >= 1e9:  return f"${x/1e9:.1f}B"
        if x >= 1e6:  return f"${x/1e6:.0f}M"
        return f"${x:,.0f}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_x))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_png(fig)


def chart_congress_bar(trades: list[dict], top_n: int = 15) -> bytes:
    """Bars per ticker of net congressional buy − sell mid-range $ over 30d."""
    plt = _mpl_setup()
    import matplotlib.pyplot as plt

    per_ticker: dict[str, float] = {}
    for t in trades:
        if t["amount_low"] is None:
            continue
        mid = (t["amount_low"] + (t["amount_high"] or t["amount_low"])) / 2
        signed = mid if t["type"] == "buy" else -mid if t["type"] == "sell" else 0
        per_ticker[t["ticker"]] = per_ticker.get(t["ticker"], 0) + signed

    items = sorted(per_ticker.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not items:
        return b""

    items.reverse()
    tickers, values = zip(*items)
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(items))), dpi=130)
    ax.barh(range(len(items)), values, color=colors, edgecolor="#2c3e50",
            linewidth=0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(tickers, fontsize=9)
    ax.axvline(0, color="#7f8c8d", linewidth=0.5)
    ax.set_xlabel("Net Congressional Flow (buys − sells, mid $)", fontsize=10)
    ax.set_title("Congressional Net Flow  (last 30 days, tracked universe)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_png(fig)


# ── LLM analysis ──────────────────────────────────────────────────────────────

def analyze_with_llm(
    dark: dict, trades: list[dict], api_key: str,
) -> dict:
    """Sonnet 4.6 bilingual analytical read on both flows."""
    fallback = {
        "narrative_en": _fallback_en(dark, trades),
        "narrative_zh": _fallback_zh(dark, trades),
    }
    if not api_key:
        return fallback

    try:
        from anthropic import Anthropic
    except ImportError:
        return fallback

    dp_lines = []
    for tk, d in sorted(dark.items(), key=lambda x: x[1].get("total_notional", 0),
                        reverse=True)[:8]:
        if not d.get("total_notional"):
            continue
        dp_lines.append(
            f"  {tk:6s} ${d['total_notional']:>16,.0f} notional, "
            f"{d['total_shares']:>13,.0f} shares, "
            f"{d.get('n_venues', 0)} venues (week of {d['week_start']})"
        )

    tr_lines = []
    for t in trades[:8]:
        low  = t.get("amount_low") or 0
        high = t.get("amount_high") or low
        tr_lines.append(
            f"  {t['transaction_date']}  {t['ticker']:6s} {t['type']:5s} "
            f"${low:,}-${high:,} — {t['filer']} ({t['state_dist']})"
        )

    prompt = (
        "You are a market-microstructure analyst writing a daily bilingual "
        "(English + 简体中文) brief on institutional and political flow for a "
        "quant portfolio manager. In 3-4 sentences per language, cover:\n"
        "  (a) any ticker with unusual dark-pool concentration or venue count\n"
        "  (b) notable congressional buys or sells (mid $ amount matters)\n"
        "  (c) any convergence between the two signals (Congress buying + "
        "     dark-pool accumulation, or vice versa)\n"
        "Do NOT recommend specific trades.\n\n"

        "DARK POOL (latest FINRA week):\n" + "\n".join(dp_lines) + "\n\n"
        "CONGRESSIONAL TRADES (last 30d, tracked universe):\n"
        + ("\n".join(tr_lines) if tr_lines else "  (none in tracked universe)")
        + "\n\n"

        "Return ONLY valid JSON, no fences, no prose outside:\n"
        "{\n"
        '  "narrative_en": "3-4 sentence English read",\n'
        '  "narrative_zh": "3-4 句中文分析"\n'
        "}\n"
        "IMPORTANT: do NOT use the double-quote character inside your string "
        "values — use single quotes or 「」."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        parsed = _safe_json(text) or fallback
        for k in ("narrative_en", "narrative_zh"):
            parsed.setdefault(k, fallback[k])
        return parsed
    except Exception as e:
        logger.warning(f"FlowPanel: LLM analysis failed ({e}) — fallback")
        return fallback


def _safe_json(text: str) -> dict | None:
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
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    def _grab(k):
        pat = rf'"{k}"\s*:\s*"(.*?)(?<!\\)"\s*(?:,|\}})'
        mm = re.search(pat, stripped, re.DOTALL)
        return mm.group(1).replace("\\n", " ").strip() if mm else None
    en = _grab("narrative_en"); zh = _grab("narrative_zh")
    if en or zh:
        return {"narrative_en": en or "", "narrative_zh": zh or ""}
    return None


def _fallback_en(dark, trades):
    parts = []
    if dark:
        top = max(dark.items(), key=lambda x: x[1].get("total_notional", 0),
                  default=None)
        if top and top[1].get("total_notional"):
            parts.append(f"Top dark-pool notional: {top[0]} "
                         f"${top[1]['total_notional']/1e6:.0f}M weekly.")
    parts.append(f"{len(trades)} congressional trades in tracked universe "
                 f"in the last 30 days.")
    return " ".join(parts) if parts else "Flow snapshot unavailable."


def _fallback_zh(dark, trades):
    parts = []
    if dark:
        top = max(dark.items(), key=lambda x: x[1].get("total_notional", 0),
                  default=None)
        if top and top[1].get("total_notional"):
            parts.append(f"暗池最大：{top[0]} 周成交额 "
                         f"${top[1]['total_notional']/1e6:.0f}M。")
    parts.append(f"过去 30 天跟踪范围内共 {len(trades)} 笔国会交易。")
    return "".join(parts) if parts else "流数据暂不可用。"


# ── HTML rendering ────────────────────────────────────────────────────────────

def _fmt_dollar(v: int | float | None) -> str:
    if v is None: return "—"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def render_html(
    universe: list[str], dark: dict, trades: list[dict],
    narrative: dict, chart_cids: dict[str, str],
) -> str:
    today = dt.date.today().isoformat()

    # Dark pool table
    dp_items = sorted(
        ((tk, d) for tk, d in dark.items() if d.get("total_notional", 0) > 0),
        key=lambda x: x[1]["total_notional"], reverse=True,
    )
    dp_rows = ""
    for tk, d in dp_items:
        venues_str = ", ".join(v["name"][:15] for v in d.get("top_venues", []))
        dp_rows += f"""
<tr>
  <td style="padding:6px 10px;font-size:12px;font-weight:bold;color:#2c3e50;
             border-top:1px solid #f0f0f0">{tk}</td>
  <td style="padding:6px 10px;text-align:right;font-size:12px;color:#2c3e50;
             border-top:1px solid #f0f0f0">{d['total_shares']:,.0f}</td>
  <td style="padding:6px 10px;text-align:right;font-size:12px;font-weight:bold;
             color:#2c3e50;border-top:1px solid #f0f0f0">
    {_fmt_dollar(d['total_notional'])}
  </td>
  <td style="padding:6px 10px;text-align:right;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f0f0f0">{d['total_trades']:,}</td>
  <td style="padding:6px 10px;text-align:right;font-size:11px;color:#5d6d7e;
             border-top:1px solid #f0f0f0">{d.get('n_venues', 0)}</td>
  <td style="padding:6px 10px;font-size:10px;color:#95a5a6;
             border-top:1px solid #f0f0f0">{venues_str}</td>
</tr>"""

    dp_html = ""
    if dp_rows:
        latest_week = dp_items[0][1].get("week_start", "?") if dp_items else "?"
        dp_html = f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="margin:0 0 14px;border:1px solid #ecf0f1;border-radius:6px;
              overflow:hidden">
  <tr style="background:#2c3e50">
    <td colspan="6" style="padding:10px 12px;color:#fff">
      <span style="font-size:13px;font-weight:bold">Dark Pool / 暗池</span>
      <span style="font-size:11px;opacity:.7"> · week of {latest_week}
        · FINRA ATS Transparency</span>
    </td>
  </tr>
  <tr style="background:#f8f9fa">
    <th style="padding:6px 10px;text-align:left;font-size:10px;color:#95a5a6">TKR</th>
    <th style="padding:6px 10px;text-align:right;font-size:10px;color:#95a5a6">SHARES</th>
    <th style="padding:6px 10px;text-align:right;font-size:10px;color:#95a5a6">NOTIONAL</th>
    <th style="padding:6px 10px;text-align:right;font-size:10px;color:#95a5a6">TRADES</th>
    <th style="padding:6px 10px;text-align:right;font-size:10px;color:#95a5a6">VENUES</th>
    <th style="padding:6px 10px;text-align:left;font-size:10px;color:#95a5a6">TOP VENUES</th>
  </tr>
  {dp_rows}
</table>"""
    else:
        dp_html = ('<p style="margin:0;font-size:12px;color:#95a5a6;'
                   'text-align:center;padding:16px">No dark-pool data '
                   'returned for tracked universe. / 跟踪范围内暂无暗池数据。</p>')

    # Congressional table
    tr_rows = ""
    for t in trades[:60]:  # cap at 60 rows for email size
        side_c = ("#27ae60" if t["type"] == "buy"
                  else "#e74c3c" if t["type"] == "sell"
                  else "#7f8c8d")
        side_zh = ("买入" if t["type"] == "buy"
                   else "卖出" if t["type"] == "sell"
                   else t["type"])
        amt_str = "—"
        if t["amount_low"] is not None:
            amt_str = (f"${t['amount_low']:,}"
                       if t["amount_high"] == t["amount_low"]
                       else f"${t['amount_low']:,}–${t['amount_high']:,}")
        pdf_url = _HOUSE_PDF_URL.format(year=t["year"], docid=t["docid"])
        tr_rows += f"""
<tr>
  <td style="padding:6px 8px;font-size:11px;color:#5d6d7e;
             border-top:1px solid #ecf0f1">{t['transaction_date']}</td>
  <td style="padding:6px 8px;font-size:12px;font-weight:bold;color:#2c3e50;
             border-top:1px solid #ecf0f1">{t['ticker']}</td>
  <td style="padding:6px 8px;font-size:11px;color:#2c3e50;
             border-top:1px solid #ecf0f1">
    {t['filer']}
    <div style="color:#95a5a6;font-size:10px">{t['state_dist']}</div>
  </td>
  <td style="padding:6px 8px;text-align:center;
             border-top:1px solid #ecf0f1">
    <span style="background:{side_c};color:#fff;font-size:10px;
                 font-weight:bold;padding:2px 6px;border-radius:3px">
      {t['type'].upper()} / {side_zh}
    </span>
  </td>
  <td style="padding:6px 8px;text-align:right;font-size:11px;color:#2c3e50;
             border-top:1px solid #ecf0f1">{amt_str}</td>
  <td style="padding:6px 8px;text-align:center;border-top:1px solid #ecf0f1">
    <a href="{pdf_url}" style="color:#3498db;font-size:10px">PDF</a>
  </td>
</tr>"""

    tr_html = ""
    if tr_rows:
        tr_html = f"""
<table width="100%" cellpadding="0" cellspacing="0"
       style="margin:0 0 14px;border:1px solid #ecf0f1;border-radius:6px;
              overflow:hidden">
  <tr style="background:#2c3e50">
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">DATE</th>
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">TKR</th>
    <th style="padding:8px;text-align:left;font-size:11px;color:#fff">FILER</th>
    <th style="padding:8px;text-align:center;font-size:11px;color:#fff">SIDE</th>
    <th style="padding:8px;text-align:right;font-size:11px;color:#fff">AMOUNT</th>
    <th style="padding:8px;text-align:center;font-size:11px;color:#fff">SRC</th>
  </tr>
  {tr_rows}
</table>"""
    else:
        tr_html = ('<p style="margin:0;font-size:12px;color:#95a5a6;'
                   'text-align:center;padding:16px">No congressional trades '
                   'in tracked universe in the last 30 days. /'
                   ' 过去 30 天跟踪范围内无国会交易。</p>')

    def _img(cid):
        return (f'<img src="cid:{cid}" style="max-width:100%;height:auto;'
                f'border:1px solid #ecf0f1;border-radius:4px" alt="chart">'
                if cid else "")

    dp_img = _img(chart_cids.get("dark_pool", ""))
    cg_img = _img(chart_cids.get("congress", ""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Flow Panel</title>
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
        Flow Panel &nbsp;
        <span style="font-size:14px;font-weight:normal;color:rgba(255,255,255,.7)">
          / 资金流面板
        </span>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(255,255,255,.7)">
        {today} &nbsp;·&nbsp; Dark Pool + Congressional Trades / 暗池+国会交易
        &nbsp;·&nbsp; universe: {len(universe)} tickers
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

    <!-- Dark pool chart -->
    <tr><td style="padding:12px 20px 4px;text-align:center">
      {dp_img}
    </td></tr>

    <!-- Dark pool table -->
    <tr><td style="padding:12px 20px 4px">
      <p style="margin:0 0 10px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        DARK POOL / 暗池
      </p>
      {dp_html}
    </td></tr>

    <!-- Congress chart -->
    <tr><td style="padding:12px 20px 4px;text-align:center">
      {cg_img}
    </td></tr>

    <!-- Congress table -->
    <tr><td style="padding:12px 20px">
      <p style="margin:0 0 10px;font-size:11px;font-weight:bold;
                color:#95a5a6;letter-spacing:.5px">
        CONGRESSIONAL TRADES · LAST 30 DAYS / 国会交易明细（近 30 天）
      </p>
      {tr_html}
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1;background:#fafbfc">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center;line-height:1.5">
        Sources: FINRA ATS Transparency (dark pool), House Clerk Financial
        Disclosures (STOCK Act PTRs) — both official &amp; free. PDF parsing
        via pdfplumber. Analysis: Claude Sonnet 4.6. Informational only. /
        仅供参考，不构成投资建议。
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>"""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_flow_panel_pipeline(config: dict | None = None) -> dict:
    logger.info("=== Flow Panel pipeline start ===")

    universe = combine_universe()
    logger.info(f"FlowPanel: tracking {len(universe)} tickers")

    dark = fetch_dark_pool(universe)
    n_dark = sum(1 for d in dark.values() if d.get("total_notional", 0) > 0)
    logger.info(f"FlowPanel: dark-pool data for {n_dark}/{len(universe)} tickers")

    trades = fetch_congressional_trades(universe, days_back=30)
    logger.info(f"FlowPanel: {len(trades)} congressional trades in tracked "
                "universe (last 30d)")

    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    narrative = analyze_with_llm(dark, trades, ant_key)

    charts = {
        "dark_pool": chart_dark_pool_bar(dark),
        "congress":  chart_congress_bar(trades),
    }
    chart_cids = {k: f"flow_{k}" for k, v in charts.items() if v}
    logger.info(f"FlowPanel: {len(chart_cids)}/2 charts rendered")

    html = render_html(universe, dark, trades, narrative, chart_cids)

    try:
        from notify.mailer import _smtp_send
        images = [(cid, charts[k]) for k, cid in chart_cids.items()]
        subject = (f"[Flow Panel] {dt.date.today()} — "
                   f"{n_dark} dark-pool · {len(trades)} congress")
        _smtp_send(html, subject, images=images)
        logger.info("FlowPanel: email sent")
    except Exception as e:
        logger.warning(f"FlowPanel: email send failed — {e}")

    logger.info("=== Flow Panel pipeline complete ===")
    return {
        "universe":  universe,
        "dark_pool": dark,
        "trades":    trades,
        "narrative": narrative,
    }
