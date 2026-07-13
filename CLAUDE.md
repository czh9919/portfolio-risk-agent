# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stock AI Agent** is an automated portfolio risk and factor analysis system for quantitative portfolio management. It runs a unified pipeline that:

1. **Fetches live holdings** from multiple brokers (IBKR Flex Query XML, Trading 212 REST API, eToro)
2. **Loads price data** via yfinance with comprehensive data-quality validation
3. **Runs Fama-French 5-factor regression** per stock in the watchlist (RUN_MODE=full)
4. **Computes portfolio risk metrics** using EWMA VaR, Cornish-Fisher (primary alert), and EVT/GPD VaR
5. **Performs Monte Carlo simulation** (DCC-GARCH + Hawkes) over 50,000 paths × 21-day horizon
6. **Generates Markowitz efficient frontier** with FF5-implied expected returns
7. **Delivers bilingual (EN/ZH) HTML email reports** with embedded risk-return charts
8. **Runs an automated paper-trading loop** (RUN_MODE=paper_trade) against an Alpaca paper account: FF5 screening of watchlist + random S&P 500 sample, robust-signal-gated orders, risk gates, stop-losses, and self-maintaining watchlist promotion/pruning
9. **Serves a FastAPI + Vue dashboard** (`api/` + `web/`) with JWT auth, WebSocket log streaming, and manual run triggers

All **real-money** trading decisions remain manual. Paper trading is fully automated but uses simulated funds only.

## Architecture Highlights

### Pipeline and Run Modes (main.py)

The system operates in **five distinct modes**, selected via RUN_MODE environment variable:

| Mode | Trigger | Pipeline |
|---|---|---|
| full | 21:30 UTC (after US close) | Holdings → prices → FF5 factor ranking → portfolio risk → Monte Carlo → optimizer → report |
| portfolio | Manual / local | Holdings → prices → portfolio risk only (skip watchlist ranking) |
| alert_check | 14:00 UTC (pre-market) | Holdings → prices → threshold check only (fire RED alerts) |
| backtest | Manual | S&P 500 universe → walk-forward validation (FF5 μ vs historical μ) |
| paper_trade | 22:00 UTC Mon–Fri | Universe expansion → FF5 screen → watchlist promote/prune → risk gates → Alpaca paper orders → state commit + email summary |

Entry point: `python main.py` reads RUN_MODE, config files, and delegates to `run_portfolio_pipeline()`, `run_backtest_pipeline()`, or `strategy.paper_engine.run_paper_trade_pipeline()`.

### Key Data Flow

```
FX Rates (load_fx_rates)
    ↓
Live Holdings (fetch_all_holdings)
    ├→ IBKR Flex Query XML
    ├→ T212 REST API
    ├→ eToro (mock/CSV)
    └→ Cache fallback
    ↓
Equity Tickers → Price Loader (load_prices)
    ├→ yfinance OHLCV (auto_adjust=True for splits/divs)
    ├→ Data quality checks (staleness, gaps, spikes)
    └→ PriceData objects (ok/reduced/excluded/unavailable)
    ↓
Risk Engine (risk.risk_engine.compute_all)
    ├→ EWMA volatility (λ=0.94)
    ├→ Cornish-Fisher VaR 95/99% (primary alert)
    ├→ EVT/GPD VaR 99% (extreme quantiles)
    └→ Portfolio metrics (Sharpe, Beta vs SPY, HHI, max DD)
    ↓
[If RUN_MODE == full]
FF5 Factor Model (strategy.factor_model)
    ├→ Fetch FF5 daily factors (Ken French library)
    ├→ Per-stock: run_factor_regression (252-day OLS window, min 63 days)
    ├→ Output: α, t(α), IR, R², 5 betas, BUY/SELL/HOLD signal
    ├→ Robust signal: PSR + deflated IR + multi-window consistency + OOS
    ├→ Cross-sectional IC analysis (temporal validation)
    ├→ Factor collinearity diagnostics (VIF, condition #, high-corr pairs)
    └→ Signal-weighted allocation (capped 20% per position)
    ↓
Optimizer (risk.optimizer)
    ├→ Markowitz frontier: 1,200 MC Dirichlet + max-Sharpe via SLSQP
    ├→ Expected returns from FF5-implied μ (no look-ahead)
    ├→ Rebalancing suggestions (3-tier: Act/Watch/Hold)
    └→ Diversification candidates (bonds/gold marginal impact)
    ↓
Monte Carlo (risk.mc_portfolio)
    ├→ DCC-GARCH + Hawkes: 50,000 paths × 21 days
    ├→ Apply Irish CGT 33%, trading costs, margin breach
    └→ Output: VaR/CVaR/drawdown/Sharpe/Sortino percentiles
    ↓
Report + Email (notify.report_gen, notify.mailer)
    ├→ Bilingual HTML (EN/ZH)
    ├→ Embedded efficient frontier chart (PNG, CID)
    └→ SMTP (primary) or SendGrid (fallback)
```

### Module Responsibilities

**`data/`** – Holdings & price management:
- `fetch_holdings.py`: Aggregates IBKR + T212 + eToro; converts all to EUR
- `price_loader.py`: yfinance batch download (primary) with Massive REST per-ticker fallback (`massive.py`, paced ≤5 req/min); PriceData status flags
- `cleaner.py`: Validates staleness, gaps, price spikes
- `cache.py`: Atomic JSON snapshots (holdings + metrics); 7-day retention
- `spy_universe.py`: S&P 500 scrape (Wikipedia) for backtest & paper-trade universe
- `paper_state/`: **Committed** paper-trade state (NAV history, trade log, trailing-stop watermarks) — lives here, NOT in .gitignored `cache/`, so it survives between GitHub Actions runs

**`strategy/`** – Quantitative models:
- `factor_model.py`: FF5 OLS regression; `run_factor_regression()` → α/β/IR/signal; portfolio exposure; 21-day attribution; robust signal filters; decision trace
- `ic_analysis.py`: Cross-sectional Information Coefficient; 8 rolling evaluation periods
- `factor_ortho.py`: Collinearity diagnostics (VIF, condition number, high-corr pairs)
- `weight_allocator.py`: Signal-weighted allocation; capped 20% per position; portfolio factor exposure warnings
- `paper_engine.py`: Paper-trading pipeline — universe expansion (watchlist + seeded random S&P 500 fill), FF5 screening, watchlist promotion/pruning, risk gates (max positions, VaR, beta, drawdown halt, VIX), stop-loss + trailing stop, CGT-aware rebalancing, IR-proportional sizing, core ETF parking (idle cash swept into a weighted mix e.g. SPY/QQQ, sold proportionally to fund signal BUYs; sweeps fire even when market is closed — Alpaca queues day-tif orders for the next open), short overlay (whole-share shorts on robust SELL signals, per-name + gross exposure caps, covered on signal fade or stop-loss; short proceeds treated as collateral, not sweepable cash), NAV/trade-log persistence, bilingual email summary

**`brokers/`** – Trading APIs:
- `alpaca.py`: Alpaca paper-trading REST client (account, positions, market/fractional orders, cancel-all, market clock). `is_market_open()` fails OPEN on clock-API errors

**`api/` + `web/`** – Dashboard:
- `api/app.py`: FastAPI app — JWT auth (`api/auth.py`; requires `WEB_PASS_HASH` or `WEB_PASS`, denies all logins if neither set), run triggers, price/watchlist/config routes, WebSocket log streaming, background scheduler, Massive WebSocket relay
- `web/`: Vue 3 + Tailwind frontend (dev server on :5173)

**`risk/`** – Risk quantification:
- `risk_engine.py`: EWMA VaR, Cornish-Fisher VaR (primary alert), EVT/GPD, CVaR, Sharpe, Beta, HHI, max DD → RAG status
- `mc_engine.py`: DCC-GARCH + Hawkes vectorized simulator (50K paths/s)
- `mc_portfolio.py`: Buy-and-hold PnL simulation; CGT 33%, trading costs, margin monitoring
- `stress.py`: Historical scenarios (COVID −34%, rate shock −25%, GFC −57%)
- `optimizer.py`: Markowitz frontier via FF5 implied μ; 3-tier rebalancing suggestions; marginal impact analysis

**`backtest/`** – Validation:
- `walk_forward.py`: Rolling 3-year train / 1-quarter test; FF5 premiums restricted to training window (no look-ahead); survivorship bias warning
- `historical.py`: Optional SMA-20/50 crossover backtest

**`notify/`** – Reporting & delivery:
- `report_gen.py`: Bilingual mobile-first HTML; inline CSS; all sections configurable
- `chart.py`: Headless matplotlib efficient frontier PNG (Agg backend, CJK fallback)
- `mailer.py`: SMTP + SendGrid fallback; CID-embedded images; backup on failure

**`alert.py`** – Alert system:
- Fires bilingual alert emails for RED metrics
- 24-hour deduplication via `cache/alert_dedup.json`

**`scheduler.py`** – Local APScheduler:
- Full pipeline: 21:30 UTC Mon–Fri
- Alert checks: every 4h UTC Mon–Fri

## Configuration

### Environment Variables (Required)

| Variable | Purpose |
|---|---|
| `RUN_MODE` | `full` \| `portfolio` \| `alert_check` \| `backtest` |
| `SMTP_HOST` | E.g., `smtp.gmail.com` |
| `SMTP_PORT` | Usually `587` |
| `SMTP_USER` | Sender email |
| `SMTP_PASS` | App password |
| `REPORT_TO_EMAIL` | Report recipient |

### Optional Broker APIs (≥1 needed for live portfolio)

| Variable | Platform |
|---|---|
| `IBKR_FLEX_TOKEN`, `IBKR_QUERY_ID` | Interactive Brokers |
| `T212_API_KEY`, `T212_API_SECRET` | Trading 212 |
| `ETORO_API_KEY` | eToro |

**Fallback:** If all broker APIs fail, pipeline uses most recent `cache/` snapshot.

### Paper Trading / Data / Dashboard (Optional)

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca paper account (RUN_MODE=paper_trade) |
| `PAPER_DRY_RUN` | `true` → screen + log orders, skip execution |
| `MASSIVE_API_KEY` | Massive REST price fallback when yfinance fails |
| `WEB_SECRET_KEY` | JWT signing key for dashboard (random per-process if unset → tokens don't survive restart) |
| `WEB_USER`, `WEB_PASS_HASH` (or `WEB_PASS`) | Dashboard login; **all logins denied if neither password var is set** |

### YAML Config Files

- **`config/settings.yaml`**: History window (730d), FF5 window (252d), MC paths (50K), model selection, `paper_trade:` block (universe size, risk-gate limits, stop-loss/trailing-stop, rebalance threshold, CGT rate, VIX halt, earnings-avoid window, core parking: idle cash swept into a weighted mix via `core_parking_tickers` (e.g. `{SPY: 0.6, QQQ: 0.4}`; legacy scalar `core_parking_ticker` also accepted), sold proportionally to fund signal BUYs, exempt from caps/stops/signal trading; short overlay: `short_pos_pct` per name / `short_max_total_pct` gross cap, 0 disables)
- **`config/thresholds.yaml`**: Per-metric alert thresholds, severity, bilingual labels; `amber_ratio` (default 0.8) — metric within 80–100% of its threshold shows AMBER in reports; only RED (actual breach) fires alert emails
- **`config/volatility.yaml`**: EWMA λ (0.94), GARCH params, ECB rate (3.5%), window sizes
- **`config/watchlist.csv`**: Tickers (equity/bond/gold), asset_class, currency — edit and commit; takes effect on next run

### .env

Copy `.env.example` → `.env`. Never commit real keys.

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Fill in SMTP_*, REPORT_TO_EMAIL, broker APIs (optional)

# 3. Full pipeline (portfolio + FF5 watchlist ranking)
RUN_MODE=full python main.py

# 4. Portfolio risk only (skip watchlist ranking)
RUN_MODE=portfolio python main.py

# 5. Alert check (no report, fire RED metrics only)
RUN_MODE=alert_check python main.py

# 6. Walk-forward backtest
RUN_MODE=backtest python main.py

# 7. Paper trading (set PAPER_DRY_RUN=true to screen without placing orders)
RUN_MODE=paper_trade python main.py

# 8. Local scheduler (full at 21:30 UTC, alerts every 4h)
python scheduler.py

# 9. Dashboard (FastAPI backend + Vue frontend)
uvicorn api.app:app --reload        # backend on :8000
cd web && npm run dev               # frontend on :5173

# 10. Tests (same as CI)
pytest tests/             # unit tests: risk math + paper engine
python test_pipeline.py   # import + smoke test
```

## GitHub Actions Workflows

**`.github/workflows/daily_run.yml`**
- Cron: 14:00 UTC (alert_check) & 21:30 UTC (full), Mon–Fri
- Manual dispatch: choose run mode
- Restores `cache/` (alert dedup, FF5 pickle, SPY universe)
- Installs optional deps (vectorbt, duckdb) and CJK fonts for matplotlib

**`.github/workflows/paper_trade.yml`**
- Cron: 22:00 UTC Mon–Fri (30 min after daily_run full mode)
- Manual dispatch with optional dry-run flag
- Commits paper-trade state back to the repo (watchlist promotions/prunes + `data/paper_state/`) with `[skip ci]`

**`.github/workflows/ci.yml`**
- Runs on push/PR to main or master
- Syntax check: all .py files (py_compile)
- Unit tests: `pytest tests/` (risk math + paper engine)
- Smoke test: imports + factor model + attribution + report sections + decision trace

## Key Design Patterns & Constraints

### Fama-French 5-Factor Model

- **Window:** Last 252 aligned trading days (falls back if <252 but ≥63; skips if <63)
- **Signal:** t(α) > 1.5 → BUY, t(α) < −1.5 → SELL
- **Robust filters:** PSR (≥85%), deflated IR (>0), multi-window consistency (≥2/3), OOS hit rate (>55%)
- **Collinearity:** VIF warnings (>5), condition # (>10), pairwise |r| (>0.4)
- **Portfolio exposure:** Market-value-weighted aggregate betas; warnings if >1.5×
- **Attribution:** 21-day decomposition into factor contributions + residual α

### VaR Methodology (Primary: Cornish-Fisher)

Plain normal EWMA underestimates tail loss by 15–30% in stress. Three estimates computed in parallel:

| Metric | Method | Use |
|---|---|---|
| `var_95_ewma` | Normal EWMA | Reference only |
| `var_95_cf` | Cornish-Fisher (EWMA σ adjusted for skewness + excess kurtosis) | **Primary alert trigger** |
| `var_99_evt` | EVT/GPD peaks-over-threshold | Most reliable at 99%+; falls back to historical if <20 exceedances |

### Monte Carlo Simulation

- **Model:** DCC-GARCH + Hawkes self-exciting jumps (50K paths, 21-day horizon)
- **Calibration:** Per-asset GARCH(1,1); time-varying correlations via DCC recursion; Hawkes jumps from tail frequency
- **Overlays:** Irish CGT 33% on realized gains (>€1,270 exemption); trading costs (commission + half-spread + √market-impact); IBKR RegT 25% margin breach detection
- **Outputs:** VaR/CVaR percentiles, max drawdown, P(loss >10%/>20%), Sharpe/Sortino/Calmar medians

### Markowitz Optimizer

- **Expected returns:** FF5-implied μ (factor premiums + α from same window, no look-ahead)
- **Frontier:** 1,200 MC Dirichlet portfolios sampled; max-Sharpe via SLSQP
- **Rebalancing tiers:**
  - Tier 1 (Act): |Δweight| ≥5%, no CGT obstacle → red
  - Tier 2 (Watch): 2–5% delta, or reducing winner → amber
  - Tier 3 (Hold): |Δweight| <2% → grey
- **Decision trace:** Each suggestion includes α-based "why" (e.g., α=+8.2% t=+2.3 → BUY signal)

### Data Quality Flags

Holdings & prices classified as `ok` / `reduced` / `excluded` / `unavailable`:
- `ok`: ≥252 days → full window
- `reduced`: 21–251 days → use actual, flag window length
- `excluded`: <21 days → include in NAV/P&L, skip from VaR calcs
- `unavailable`: fetch failed

**Staleness warnings:** Last close >4 calendar days old; FX rate >4 days old.

### Backtest Survivorship Bias Warning

Walk-forward uses current S&P 500 constituent list (Wikipedia). Companies removed via bankruptcy, acquisition, or ejection are never in historical universe, overstating Sharpe and understating drawdowns. Every result flagged `[!] SURVIVORSHIP BIAS`.

## Code Style & Conventions

- **Logging:** `logger = logging.getLogger(__name__)` in each module; INFO for pipelines, WARNING for anomalies
- **Config loading:** `main.load_yaml()` for YAML; `os.environ.get()` for env vars (with defaults)
- **EUR baseline:** All currency-denominated fields end in `_eur` (e.g., `nav_eur`, `market_value_eur`)
- **Bilingual strings:** `label_en` & `label_zh` pairs in config; HTML templates return `(html_en, html_zh)` tuple
- **NaN handling:** Use `x != x` (IEEE 754) or `np.isnan()` to test for NaN (not `x is nan`)
- **Pandas frequency:** Assume business days in all rolling windows; explicit `freq="B"` in `date_range()`
- **Factor names:** `["Mkt-RF", "SMB", "HML", "RMW", "CMA"]` per Ken French

## Common Development Tasks

### Adding a New Risk Metric

1. Compute scalar in `risk/risk_engine.py::compute_all()`
2. Add RAG threshold in `config/thresholds.yaml` (label_en, label_zh, severity)
3. Map short key → threshold key in `alert.py::_ALERT_TO_THRESH_KEY`
4. Add format function in `alert.py::_get_value()`
5. Include in `notify/report_gen.py` HTML section

### Adding a New Watchlist Asset

Edit `config/watchlist.csv`:
```
TICKER,1.0,Description,equity|bond|gold,USD|EUR|GBP
```
Changes take effect on next run (no code redeploy).

### Modifying FF5 Window or Risk Metrics

Edit `config/volatility.yaml` or `config/settings.yaml`; no code changes needed.

### Running Tests Locally

```bash
python test_pipeline.py
```
Smoke test: imports all modules, runs factor model + attribution + report generation with synthetic data.

### Adding a New Broker Integration

1. Implement fetcher in `data/fetch_holdings.py` (add platform)
2. Return list of dicts: `[{ticker, platform, quantity, cost_basis_eur, market_value_eur, currency, ...}]`
3. Add environment variable to `.env.example`
4. Add to GitHub Actions secrets

## Common Pitfalls

- **Look-ahead bias:** FF5 factor premiums **must be restricted to training window** in backtest. Check `backtest/walk_forward.py::run_walk_forward()`.
- **NaN propagation:** Use `df.fillna(method='ffill').fillna(0)` carefully; explicit intent prevents silent data loss.
- **Staleness:** Holdings & prices cached in `cache/`. If broker APIs fail silently, fallback snapshot may be days old → check logs.
- **FX rates:** Always USD/EUR base (inverted from yfinance). `EURUSD=X` → `1 / rate` to get EUR per USD.
- **Time zones:** All cron schedules in **UTC**. GitHub Actions runner is UTC; Dublin is UTC+0/+1, ET is UTC−4/−5.
- **matplotlib fonts:** Must install `fonts-wqy-microhei` on CI for CJK axis labels. Windows may need manual font install.
- **Margin calculations:** IBKR RegT 25% breach flag only; system does **not auto-liquidate**. Manual action required.
- **CGT calculation:** Irish 33% applies to realized gains >€1,270. Monte Carlo applies this on synthetic paths; actual trades require manual calculation.
- **Paper-trade state:** `data/paper_state/` and `config/watchlist.csv` are committed by the paper_trade workflow after each run — pull before editing locally to avoid push conflicts. On a rebase conflict the workflow currently drops that day's state commit silently.
- **Dashboard auth:** logins are denied unless `WEB_PASS_HASH` (preferred) or `WEB_PASS` is set; if `WEB_SECRET_KEY` is unset the JWT key is random per-process, so tokens invalidate on every restart.

## Performance & Limits

- **Price fetch:** ~2–3 sec per 50 tickers (yfinance batch)
- **FF5 factors:** Cached 20h in `cache/ff5_daily.pkl`
- **Monte Carlo:** 50K paths × 21 days ≈ 1 sec (vectorized numpy)
- **Report generation:** ~10–15 sec (chart + HTML rendering)
- **Email:** ~5–10 sec (SMTP + fallback retry)
- **Total pipeline:** ~3–5 min (full mode) end-to-end

**Limits:**
- Max watchlist: ~100 equities (factor model O(n) in OLS window)
- Max positions: no hard limit; concentration warnings at HHI >0.25 and single position >30%
- History: default 730d; configurable in `config/settings.yaml`