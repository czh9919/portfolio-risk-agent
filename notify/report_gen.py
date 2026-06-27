"""
M5 — Bilingual mobile-first HTML email report.
All CSS is inline (Gmail strips <style> blocks).
Max output target: < 100 KB.
"""
from datetime import date
from typing import Optional


_RAG_COLOR  = {"RED": "#c0392b", "AMBER": "#e67e22", "GREEN": "#27ae60", "GREY": "#95a5a6"}
_RAG_BG     = {"RED": "#fdf2f2", "AMBER": "#fef9f0", "GREEN": "#f0fdf4", "GREY": "#f5f5f5"}
_RAG_ZH     = {"RED": "红", "AMBER": "黄", "GREEN": "绿", "GREY": "灰"}

_S = "font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;"
_CARD = "display:inline-block;width:130px;padding:10px 12px;margin:4px;border-radius:8px;background:#f8f9ff;vertical-align:top;"

_TIER_COLOR = {1: "#c0392b", 2: "#e67e22", 3: "#95a5a6"}


def _pct(v, decimals=1):
    if v is None or v != v:
        return "N/A"
    return f"{v*100:.{decimals}f}%"

def _eur(v):
    if v is None or v != v:
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}€{abs(v):,.0f}"

def _f(v, decimals=2):
    if v is None or v != v:
        return "N/A"
    return f"{v:.{decimals}f}"

def _delta(now, prev):
    if now is None or prev is None or now != now or prev != prev:
        return ""
    d = now - prev
    sign = "▲" if d >= 0 else "▼"
    color = "#27ae60" if d >= 0 else "#c0392b"
    return f' <span style="color:{color};font-size:11px">{sign}{abs(d)*100:.1f}pp</span>'


def _sig_stars(t_val) -> str:
    if t_val is None or t_val != t_val:
        return "—"
    a = abs(t_val)
    if a >= 3.0: return "★★★"
    if a >= 2.0: return "★★"
    if a >= 1.5: return "★"
    return "—"


def _t_color(t_val) -> str:
    if t_val is None or t_val != t_val:
        return "#bbb"
    a = abs(t_val)
    if a >= 3.0: return "#27ae60"
    if a >= 2.0: return "#2980b9"
    if a >= 1.5: return "#e67e22"
    return "#bbb"


_EXEC_VERDICT = {
    "REDUCE":     ("exec_v_reduce",     "#c0392b"),
    "OBSERVE":    ("exec_v_observe",    "#95a5a6"),
    "ACCUMULATE": ("exec_v_accumulate", "#27ae60"),
    "WAIT":       ("exec_v_wait",       "#95a5a6"),
}


def _ladder_chips(rungs: list, t: dict, chip_bg: str, chip_border: str,
                  txt_col: str) -> str:
    """Render one ladder's rungs as inline price/size chips."""
    if not rungs:
        return "<span style='font-size:11px;color:#bbb'>—</span>"
    out = ""
    for r in rungs:
        hvn_tag = (f"<sup style='color:#2980b9;font-size:8px'>{t['exec_hvn_tag']}</sup>"
                   if r.get("at_hvn") else "")
        sh = r.get("shares")
        sh_str = (f" <span style='color:#555'>×{sh:,}</span>"
                  if sh else "")
        cap_flag = ("<sup style='color:#c0392b;font-size:8px'>!</sup>"
                    if r.get("over_cap") else "")
        out += (
            f"<span style='display:inline-block;background:{chip_bg};border:1px solid {chip_border};"
            f"border-radius:6px;padding:2px 7px;margin:2px 3px 2px 0;font-size:11px;color:{txt_col}'>"
            f"{r['price']:.2f}{hvn_tag}{sh_str}{cap_flag} "
            f"<span style='color:#aaa'>({r['dist_pct']:+.1f}%)</span></span>"
        )
    return out


def _exec_card(e: dict, t: dict) -> str:
    """One execution-plan card: ascending (减仓) + descending (建仓) ladders."""
    v_key, v_col = _EXEC_VERDICT.get(e.get("verdict", "WAIT"), ("exec_v_wait", "#95a5a6"))
    badge = (f"<span style='background:{v_col};color:#fff;padding:2px 8px;"
             f"border-radius:10px;font-size:11px;font-weight:700'>{t[v_key]}</span>")

    prem = e.get("premium_pct")
    prem_str = ""
    if prem is not None:
        pcol = "#c0392b" if prem >= 0 else "#27ae60"
        prem_str = (f"<span style='color:{pcol};font-size:11px'>"
                    f"{t['exec_anchor']} {prem:+.1f}%</span>")

    primary = e.get("primary", "up")
    star = "<span style='color:#e67e22'>&#9733;</span> "  # ★ marks the primary intent
    up_label = (star if primary == "up" else "") + t['exec_ladder_up']
    down_label = (star if primary == "down" else "") + t['exec_ladder_down']

    up_html = _ladder_chips(e.get("ladder_up", []), t, "#fdf3f2", "#f3d6d2", "#c0392b")
    down_html = _ladder_chips(e.get("ladder_down", []), t, "#eef9f1", "#cdeed6", "#1e8449")

    # ADV cap + days-to-exit
    cap_bits = []
    if e.get("max_shares_day"):
        cap_bits.append(f"{t['exec_adv_cap']} {e['max_shares_day']:,}/d "
                        f"({e.get('participation_pct', 0)*100:.0f}% ADV)")
    if e.get("days_to_exit") is not None:
        cap_bits.append(t['exec_days'].format(d=e['days_to_exit']))
    cap_str = " · ".join(cap_bits)

    # OPEX option levels
    opts = e.get("options") or {}
    opex_bits = []
    if opts.get("max_pain") is not None:
        opex_bits.append(f"{t['exec_maxpain']} {opts['max_pain']:.0f}")
    if opts.get("peak_gamma") is not None:
        opex_bits.append(f"{t['exec_gamma']} {opts['peak_gamma']:.0f}")
    opex_str = ""
    if opex_bits:
        opex_str = (f"<span style='color:#888'>{t['exec_opex']} {opts.get('expiry','')}: "
                    f"{' · '.join(opex_bits)}</span>")

    footer = " &nbsp;·&nbsp; ".join(x for x in (cap_str, opex_str) if x)
    footer_html = (f"<div style='font-size:11px;color:#888;margin-top:4px'>{footer}</div>"
                   if footer else "")

    return f"""
<div style="border:1px solid #eee;border-radius:8px;padding:8px 10px;margin:0 0 8px;background:#fff">
  <div style="font-size:13px;color:#2c3e50">
    <b>{e['ticker']}</b> &nbsp; {badge} &nbsp;
    <span style="color:#888">@ {e.get('price', '—')}</span> &nbsp; {prem_str}
  </div>
  <div style="margin-top:5px"><span style="font-size:11px;color:#c0392b;font-weight:700">{up_label}</span>: {up_html}</div>
  <div style="margin-top:3px"><span style="font-size:11px;color:#1e8449;font-weight:700">{down_label}</span>: {down_html}</div>
  {footer_html}
</div>"""


def build_report(metrics: dict, stress: dict, holdings: list[dict],
                 price_data: dict, last_week: Optional[dict] = None,
                 lang: str = "both",
                 frontier: Optional[dict] = None,
                 suggestions: Optional[list] = None,
                 has_chart_en: bool = False,
                 has_chart_zh: bool = False,
                 div_candidates: Optional[list] = None,
                 fx_rates: Optional[dict] = None,
                 stock_factors: Optional[list] = None,
                 port_exposure: Optional[dict] = None,
                 attribution: Optional[dict] = None,
                 collinearity: Optional[dict] = None,
                 ic_result: Optional[dict] = None,
                 signal_weights: Optional[list] = None,
                 execution: Optional[list] = None) -> tuple[str, str]:
    """Returns (html_en, html_zh)."""
    en = _build(metrics, stress, holdings, price_data, last_week, "en",
                frontier, suggestions, has_chart=has_chart_en,
                div_candidates=div_candidates, fx_rates=fx_rates,
                stock_factors=stock_factors, port_exposure=port_exposure,
                attribution=attribution, collinearity=collinearity,
                ic_result=ic_result, signal_weights=signal_weights,
                execution=execution)
    zh = _build(metrics, stress, holdings, price_data, last_week, "zh",
                frontier, suggestions, has_chart=has_chart_zh,
                div_candidates=div_candidates, fx_rates=fx_rates,
                stock_factors=stock_factors, port_exposure=port_exposure,
                attribution=attribution, collinearity=collinearity,
                ic_result=ic_result, signal_weights=signal_weights,
                execution=execution)
    return en, zh


def _build(metrics, stress, holdings, price_data, last_week, lang,
           frontier=None, suggestions=None, has_chart=False,
           div_candidates=None, fx_rates=None,
           stock_factors=None, port_exposure=None, attribution=None,
           collinearity=None, ic_result=None, signal_weights=None,
           execution=None):
    t        = _T[lang]
    rag      = metrics.get("overall_rag", "GREY")
    rag_col  = _RAG_COLOR[rag]
    rag_bg   = _RAG_BG[rag]
    today    = date.today().isoformat()
    rag_label = _RAG_ZH[rag] if lang == "zh" else rag

    prev_metrics = (last_week or {}).get("metrics", {})

    sections = []

    # ── Header ─────────────────────────────────────────────────────────────────
    sections.append(f"""
<div style="{_S}background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px;border-radius:12px 12px 0 0">
  <h1 style="margin:0 0 4px;font-size:20px;color:#fff">{t['title']}</h1>
  <p style="margin:0;font-size:12px;color:rgba(255,255,255,.65)">{t['generated']} {today}</p>
  <div style="margin-top:10px;display:inline-block;padding:4px 14px;border-radius:20px;
       background:{rag_col};color:#fff;font-weight:700;font-size:13px">
    {t['risk_level']}: {rag_label}
  </div>
</div>""")

    # ── Alert banner ───────────────────────────────────────────────────────────
    alert_map = metrics.get("alerts", {})
    active_alerts = [k for k, v in alert_map.items() if v == "RED"]
    if active_alerts:
        al_text = " &nbsp;|&nbsp; ".join(
            t['alert_labels'].get(k, k) for k in active_alerts
        )
        sections.append(f"""
<div style="background:#fdf2f2;border-left:4px solid #c0392b;padding:10px 16px;margin:0">
  <strong style="color:#c0392b">{t['alert_banner']}</strong>
  <br><span style="font-size:13px;color:#555">{al_text}</span>
</div>""")

    # ── Snapshot cards ─────────────────────────────────────────────────────────
    nav      = metrics.get("nav_eur", 0)
    pnl      = metrics.get("total_pnl_eur", 0)
    var95    = metrics.get("var_95_cf", metrics.get("var_95_ewma"))  # CF-adjusted, fallback to EWMA
    sharpe   = metrics.get("sharpe")
    pnl_col  = "#27ae60" if (pnl or 0) >= 0 else "#c0392b"

    bond_holdings  = [h for h in holdings if h.get("asset_class") == "bond"]
    bond_nav       = sum(h["market_value_eur"] for h in bond_holdings)
    bond_pct       = bond_nav / nav if nav else 0.0
    bond_card      = ""
    if bond_holdings:
        bond_card = f"""
    <div style="{_CARD}">
      <div style="font-size:11px;color:#888">{t['bond_exposure']}</div>
      <div style="font-size:18px;font-weight:700;color:#2c3e50">{_pct(bond_pct)}</div>
      <div style="font-size:11px;color:#aaa">{_eur(bond_nav)}</div>
    </div>"""

    sections.append(f"""
<div style="padding:16px 20px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['snapshot']}
  </h2>
  <div>
    <div style="{_CARD}">
      <div style="font-size:11px;color:#888">{t['nav']}</div>
      <div style="font-size:18px;font-weight:700;color:#1a1a2e">{_eur(nav)}</div>
    </div>
    <div style="{_CARD}">
      <div style="font-size:11px;color:#888">{t['pnl']}</div>
      <div style="font-size:18px;font-weight:700;color:{pnl_col}">{_eur(pnl)}</div>
    </div>
    <div style="{_CARD}">
      <div style="font-size:11px;color:#888">{t['var95']}</div>
      <div style="font-size:18px;font-weight:700;color:{_RAG_COLOR[alert_map.get('var_95','GREY')]}">{_pct(var95)}</div>
    </div>
    <div style="{_CARD}">
      <div style="font-size:11px;color:#888">{t['sharpe']}</div>
      <div style="font-size:18px;font-weight:700;color:{_RAG_COLOR[alert_map.get('sharpe','GREY')]}">{_f(sharpe)}</div>
    </div>{bond_card}
  </div>
</div>""")

    # ── Risk-return chart (CID inline image) ───────────────────────────────────
    if has_chart and frontier:
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['rr_chart']}
  </h2>
  <img src="cid:rrChart" alt="{t['rr_chart']}"
       style="max-width:100%;display:block;border-radius:8px;border:1px solid #eee">
  <p style="font-size:11px;color:#aaa;margin:6px 0 0">{t['rr_chart_note']}</p>
</div>""")

    # ── Top 3 worst stress scenarios ───────────────────────────────────────────
    top3 = (stress or {}).get("top3_worst", [])
    if top3:
        rows = ""
        for s in top3:
            rows += (
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0'>"
                f"<b>{s['name']}</b><br><span style='color:#888;font-size:11px'>{s['name_zh']}</span></td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#c0392b'>{_eur(s['eur_loss'])}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#c0392b'>{_pct(s['pct_loss'])}</td>"
                f"</tr>"
            )
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['stress_top3']}
  </h2>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr style="background:#f8f9ff">
      <th style="padding:6px 8px;text-align:left;font-weight:600;color:#666">{t['scenario']}</th>
      <th style="padding:6px 8px;text-align:left;font-weight:600;color:#666">{t['eur_loss']}</th>
      <th style="padding:6px 8px;text-align:left;font-weight:600;color:#666">{t['pct_loss']}</th>
    </tr>{rows}
  </table>
</div>""")

    # ── Risk indicators table with descriptions ────────────────────────────────
    prev_m = prev_metrics
    # Each row: (key, en_label, zh_label, val, prev_val, (en_desc, zh_desc))
    risk_rows = [
        ("var_95",   "VaR 95% (adj.)",    "VaR 95%（Cornish-Fisher）",
         _pct(metrics.get("var_95_cf", metrics.get("var_95_ewma"))),
         _pct(prev_m.get("var_95_cf",  prev_m.get("var_95_ewma"))),
         ("1-in-20 daily loss — EWMA volatility corrected for skewness & kurtosis",
          "1/20概率最差日损失，已对偏态与厚尾进行Cornish-Fisher修正")),
        ("var_99",   "VaR 99% (EVT/adj.)", "VaR 99%（EVT/CF）",
         _pct(metrics.get("var_99_evt",  metrics.get("var_99_cf", metrics.get("var_99_ewma")))),
         _pct(prev_m.get("var_99_evt",   prev_m.get("var_99_cf",  prev_m.get("var_99_ewma")))),
         ("1-in-100 daily loss — EVT/GPD estimate; falls back to Cornish-Fisher then normal",
          "1/100概率最差日损失，优先使用EVT广义Pareto估计，次之CF修正")),
        ("cvar",     "CVaR (95%)",   "CVaR（95%）",
         _pct(metrics.get("cvar_95")),        _pct(prev_m.get("cvar_95")),
         ("Average loss in the worst 5% of outcomes (tail risk)",
          "最差5%情景下的平均损失（尾部风险）")),
        ("max_dd",   "Max Drawdown", "最大回撤",
         _pct(metrics.get("max_drawdown")),  _pct(prev_m.get("max_drawdown")),
         ("Largest peak-to-trough decline in the period",
          "区间内最大峰谷回撤幅度")),
        ("sharpe",   "Sharpe Ratio", "夏普比率",
         _f(metrics.get("sharpe")),           _f(prev_m.get("sharpe")),
         ("Excess return per unit of risk; >1 = good, <0.5 = poor",
          "每单位风险的超额收益；>1为良好，<0.5为较差")),
        ("beta",     "Beta",         "贝塔系数",
         _f(metrics.get("beta")),             _f(prev_m.get("beta")),
         ("Portfolio sensitivity to the market (S&P 500); 1 = tracks index",
          "对市场（标普500）的敏感度；1=与指数同步波动")),
        ("hhi",      "HHI",          "HHI集中度",
         _f(metrics.get("hhi")),              _f(prev_m.get("hhi")),
         ("Concentration index: 0 = fully diversified, 1 = single position",
          "集中度指数：0=完全分散，1=单一持仓")),
        ("max_pos",  "Max Position", "最大单仓",
         _pct(metrics.get("max_position_wt")),_pct(prev_m.get("max_position_wt")),
         ("Largest single holding as % of total NAV",
          "最大单一持仓占总市值的比例")),
        ("port_vol", "Portfolio σ",  "组合波动率",
         _pct(metrics.get("port_sigma_annual")),_pct(prev_m.get("port_sigma_annual")),
         ("Annualised portfolio volatility (standard deviation of returns)",
          "年化组合波动率（收益率标准差）")),
    ]
    risk_html = ""
    for key, en_label, zh_label, val, prev_val, descs in risk_rows:
        rag_key = {"var_95":"var_95","var_99":"var_95","cvar":"cvar_ratio",
                   "max_dd":"max_dd","sharpe":"sharpe","beta":"beta",
                   "hhi":"hhi","max_pos":"max_pos","port_vol":"var_95"}.get(key, key)
        r_color  = _RAG_COLOR.get(alert_map.get(rag_key, "GREY"), "#888")
        label    = zh_label if lang == "zh" else f"{en_label}（{zh_label}）"
        desc     = descs[1] if lang == "zh" else descs[0]
        prev_tag = ""
        if prev_val not in ("N/A", None, ""):
            prev_tag = f" <span style='color:#bbb;font-size:10px'>prev {prev_val}</span>"
        risk_html += (
            f"<tr><td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:13px'>"
            f"<span style='font-weight:600'>{label}</span>"
            f"<br><span style='color:#aaa;font-size:10px'>{desc}</span></td>"
            f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:13px;"
            f"color:{r_color};font-weight:600;white-space:nowrap'>{val}{prev_tag}</td></tr>"
        )

    sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['risk_indicators']}
  </h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:6px 8px;text-align:left;font-weight:600;color:#666;font-size:13px">{t['metric']}</th>
      <th style="padding:6px 8px;text-align:left;font-weight:600;color:#666;font-size:13px">{t['value']}</th>
    </tr>{risk_html}
  </table>
</div>""")

    # ── Portfolio factor exposure ──────────────────────────────────────────────
    if port_exposure:
        pe = port_exposure
        _PE_META = [
            ("beta_mkt", "se_mkt", "t_mkt", "Market",       "市场"),
            ("beta_smb", "se_smb", "t_smb", "Size (SMB)",   "规模(SMB)"),
            ("beta_hml", "se_hml", "t_hml", "Value (HML)",  "价值(HML)"),
            ("beta_rmw", "se_rmw", "t_rmw", "Profit (RMW)", "盈利(RMW)"),
            ("beta_cma", "se_cma", "t_cma", "Invest (CMA)", "投资(CMA)"),
        ]
        alpha_val = pe.get("alpha_ann", 0.0) or 0.0
        a_col     = "#27ae60" if alpha_val >= 0 else "#c0392b"
        pe_rows   = (
            f"<tr>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
            f"{'α (ann.)' if lang == 'en' else 'α（年化）'}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
            f"font-weight:700;color:{a_col}'>{_pct(alpha_val)}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#bbb'>—</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#bbb'>—</td>"
            f"</tr>"
        )
        for bkey, sekey, tkey, en_lbl, zh_lbl in _PE_META:
            lbl  = zh_lbl if lang == "zh" else en_lbl
            bval = pe.get(bkey)
            sval = pe.get(sekey)
            tval = pe.get(tkey)
            se_str   = f"±{sval:.3f}" if (sval is not None and sval == sval) else ""
            tval_str = _f(tval, 1) if (tval is not None and tval == tval) else "—"
            stars    = _sig_stars(tval)
            tc       = _t_color(tval)
            se_html  = (f"<br><span style='font-size:10px;color:#bbb'>{se_str}</span>"
                        if se_str else "")
            pe_rows += (
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{lbl}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
                f"{_f(bval)}{se_html}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{tc};font-weight:600'>{tval_str}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{tc}'>{stars}</td>"
                f"</tr>"
            )
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['port_exposure']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['port_exposure_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:40%">{t['pe_factor']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:24%">β (±SE)</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:20%">{t['pe_t']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:16%">{t['pe_sig']}</th>
    </tr>{pe_rows}
  </table>
  <p style="font-size:10px;color:#ccc;margin:6px 0 0">{t['pe_se_note']}</p>
</div>""")

    # ── Performance attribution ───────────────────────────────────────────────
    if attribution and attribution.get("contributions"):
        atr         = attribution
        alpha_gross = atr.get("alpha_gross", atr.get("alpha", 0.0))
        alpha_net   = atr.get("alpha_net", alpha_gross)
        cost        = atr.get("trading_cost_pct", 0.0) or 0.0
        total       = atr["total_return"]
        factor_tot  = atr["factor_total"]
        attr_error  = atr.get("attribution_error")
        p_ols       = atr.get("portfolio_ols")

        ag_col    = "#27ae60" if alpha_gross >= 0 else "#c0392b"
        an_col    = "#27ae60" if alpha_net   >= 0 else "#c0392b"
        total_col = "#27ae60" if total       >= 0 else "#c0392b"

        _AFL = {
            "Mkt-RF": ("Market",  "市场"),
            "SMB":    ("Size",    "规模"),
            "HML":    ("Value",   "价值"),
            "RMW":    ("Profit",  "盈利"),
            "CMA":    ("Invest",  "投资"),
        }
        # RF row
        rf_val    = atr.get("rf", 0.0)
        attr_rows = (
            f"<tr style='background:#f8f9ff'>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#888'>"
            f"{'Risk-free (RF)' if lang == 'en' else '无风险利率 (RF)'}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#bbb'>—</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#888'>"
            f"{_pct(rf_val)}</td>"
            f"</tr>"
        )
        # Factor rows — 3 columns: Factor+factor_ret sub | β+t/sig sub | contrib+pct sub
        for fname, info in atr["contributions"].items():
            lbl_pair = _AFL.get(fname, (fname, fname))
            lbl      = lbl_pair[1] if lang == "zh" else lbl_pair[0]
            beta     = info["beta"]
            fr       = info["factor_ret"]
            con      = info["contrib"]
            tval     = info.get("beta_t")
            c_se     = info.get("contrib_se")
            pct_tot  = info.get("pct_of_total")

            fr_col  = "#27ae60" if (fr or 0) >= 0 else "#c0392b"
            fr_sub  = f"<br><span style='font-size:10px;color:{fr_col}'>{_pct(fr, 2)}</span>"

            stars   = _sig_stars(tval)
            tc      = _t_color(tval)
            tv_sub  = ""
            if tval is not None and tval == tval:
                tv_sub = (f"<br><span style='font-size:10px;color:{tc}'>"
                          f"t={_f(tval, 1)} {stars}</span>")

            c_col   = "#27ae60" if (con or 0) >= 0 else "#c0392b"
            pct_sub = ""
            if pct_tot is not None and pct_tot == pct_tot:
                pct_sub = f"<br><span style='font-size:10px;color:#bbb'>({_pct(pct_tot, 1)})</span>"
            se_sub  = ""
            if c_se is not None and c_se == c_se and c_se > 1e-10:
                se_sub = f"<span style='font-size:10px;color:#ddd'> ±{_pct(c_se, 2)}</span>"

            attr_rows += (
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
                f"<b>{lbl}</b>{fr_sub}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
                f"{_f(beta)}{tv_sub}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{c_col};font-weight:600'>{_pct(con)}{pct_sub}{se_sub}</td>"
                f"</tr>"
            )

        # Factor total
        ft_pct = atr.get("factor_total_pct")
        ft_sub = ""
        if ft_pct is not None and ft_pct == ft_pct:
            ft_sub = f" <span style='font-size:10px;color:#bbb'>({_pct(ft_pct, 1)})</span>"
        attr_rows += (
            f"<tr style='border-top:2px solid #e0e0e0;background:#f8f8f8'>"
            f"<td colspan='2' style='padding:6px 8px;font-size:12px;font-weight:600;color:#555'>"
            f"{'Factor total' if lang == 'en' else '因子合计'}</td>"
            f"<td style='padding:6px 8px;font-size:12px;font-weight:600'>"
            f"{_pct(factor_tot)}{ft_sub}</td>"
            f"</tr>"
            f"<tr style='background:#fff8e1'>"
            f"<td colspan='2' style='padding:6px 8px;font-size:12px;font-weight:600'>"
            f"{t['attr_alpha_gross']}</td>"
            f"<td style='padding:6px 8px;font-size:13px;font-weight:700;color:{ag_col}'>"
            f"{_pct(alpha_gross)}</td>"
            f"</tr>"
        )
        if cost != 0.0:
            an_pct = atr.get("alpha_net_pct")
            an_sub = ""
            if an_pct is not None and an_pct == an_pct:
                an_sub = f" <span style='font-size:10px;color:#bbb'>({_pct(an_pct, 1)})</span>"
            attr_rows += (
                f"<tr>"
                f"<td colspan='2' style='padding:5px 8px;font-size:12px;color:#888'>"
                f"{t['attr_trading_cost']}</td>"
                f"<td style='padding:5px 8px;font-size:12px;color:#c0392b'>−{_pct(abs(cost))}</td>"
                f"</tr>"
                f"<tr style='background:#f0fdf4'>"
                f"<td colspan='2' style='padding:6px 8px;font-size:12px;font-weight:700'>"
                f"{t['attr_alpha_net']}</td>"
                f"<td style='padding:6px 8px;font-size:13px;font-weight:700;color:{an_col}'>"
                f"{_pct(alpha_net)}{an_sub}</td>"
                f"</tr>"
            )
        attr_rows += (
            f"<tr style='background:#f0f4ff'>"
            f"<td colspan='2' style='padding:6px 8px;font-size:12px;font-weight:700'>"
            f"{'Portfolio total' if lang == 'en' else '组合总收益'}</td>"
            f"<td style='padding:6px 8px;font-size:13px;font-weight:700;color:{total_col}'>"
            f"{_pct(total)}</td>"
            f"</tr>"
        )
        err_note = ""
        if attr_error is not None and attr_error == attr_error:
            err_note = (f"\n  <p style='font-size:10px;color:#bbb;margin:6px 0 0'>"
                        f"{t['attr_error'].format(e=_pct(attr_error, 2))}</p>")
        ols_note = ""
        if p_ols:
            ols_note = (f"\n  <p style='font-size:10px;color:#bbb;margin:2px 0 0'>"
                        f"{t['attr_ols_note'].format(n=p_ols['n_days'])}</p>")
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['attribution'].format(n=atr['window'])}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">
    {t['attribution_note'].format(n=atr['window'], start=atr['start'], end=atr['end'])}
  </p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:38%">{t['attr_factor']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:28%">β</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:34%">{t['attr_contrib']}</th>
    </tr>{attr_rows}
  </table>{err_note}{ols_note}
  <p style="font-size:10px;color:#ccc;margin:4px 0 0">{t['attr_sig_note']}</p>
</div>""")

    # ── Factor model quality (IC + collinearity) ─────────────────────────────
    if ic_result or (collinearity and collinearity.get("n_obs", 0) > 0):
        qrows = ""

        # IC analysis row
        if ic_result:
            ic_m   = ic_result.get("ic_mean")
            icir_v = ic_result.get("icir")
            pval   = ic_result.get("pvalue")
            nobs   = ic_result.get("n_obs", 0)
            fwd    = ic_result.get("forward_days", 21)

            ic_col = "#27ae60" if (ic_m or 0) > 0.05 else ("#e67e22" if (ic_m or 0) > 0 else "#c0392b")
            pval_str = _f(pval, 3) if (pval is not None and pval == pval) else "N/A"
            pval_col = "#27ae60" if (pval or 1.0) < 0.05 else ("#e67e22" if (pval or 1.0) < 0.10 else "#bbb")
            icir_str = _f(icir_v, 2) if (icir_v is not None and icir_v == icir_v) else "N/A"
            ic_label    = t["fmq_ic_label"].format(fwd=fwd)
            icir_label  = t["fmq_icir_label"]
            pval_label  = t["fmq_pval_label"]
            nobs_label  = t["fmq_nobs_label"]

            qrows += (
                f"<tr style='background:#f0fdf4'>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;font-weight:600'>"
                f"{ic_label}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{ic_col};font-weight:700'>{_f(ic_m, 3) if ic_m is not None and ic_m == ic_m else 'N/A'}</td>"
                f"</tr>"
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{icir_label}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;font-weight:600'>{icir_str}</td>"
                f"</tr>"
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{pval_label}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{pval_col};font-weight:600'>{pval_str}</td>"
                f"</tr>"
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#888'>{nobs_label}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#888'>{nobs}</td>"
                f"</tr>"
            )

        # Collinearity section
        if collinearity and collinearity.get("n_obs", 0) > 0:
            cond_num   = collinearity.get("condition_number")
            vif_dict   = collinearity.get("vif", {})
            high_pairs = collinearity.get("high_pairs", [])
            has_warn   = collinearity.get("has_warning", False)

            warn_badge = ""
            if has_warn:
                warn_badge = (f"<span style='background:#e67e22;color:#fff;padding:1px 7px;"
                              f"border-radius:8px;font-size:10px;margin-left:6px'>"
                              f"{t['fmq_warn']}</span>")
            cond_col = "#c0392b" if (cond_num or 0) > 30 else ("#e67e22" if (cond_num or 0) > 10 else "#27ae60")
            cond_str = _f(cond_num, 1) if (cond_num is not None and cond_num != float("inf")) else "∞"

            qrows += (
                f"<tr style='background:#f8f9ff'>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;font-weight:600'>"
                f"{t['fmq_cond']}{warn_badge}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{cond_col};font-weight:700'>{cond_str}</td>"
                f"</tr>"
            )

            # VIF table
            if vif_dict:
                vif_cells = ""
                for fname, vval in vif_dict.items():
                    vc = "#c0392b" if vval > 10 else ("#e67e22" if vval > 5 else "#27ae60")
                    vif_str = _f(vval, 1) if vval != float("inf") else "∞"
                    vif_cells += (
                        f"<span style='margin-right:10px;white-space:nowrap'>"
                        f"<span style='color:#888'>{fname}:</span> "
                        f"<b style='color:{vc}'>{vif_str}</b></span>"
                    )
                qrows += (
                    f"<tr>"
                    f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{t['fmq_vif']}</td>"
                    f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:11px'>{vif_cells}</td>"
                    f"</tr>"
                )

            # High-correlation pairs
            if high_pairs:
                pair_strs = []
                for fa, fb, cv in high_pairs:
                    pc = "#c0392b" if abs(cv) > 0.7 else "#e67e22"
                    pair_strs.append(
                        f"<span style='color:{pc};margin-right:8px'>{fa}↔{fb}: {cv:+.2f}</span>"
                    )
                qrows += (
                    f"<tr>"
                    f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{t['fmq_high_pairs']}</td>"
                    f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:11px'>{''.join(pair_strs)}</td>"
                    f"</tr>"
                )

        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['fmq_title']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['fmq_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:50%">{t['metric']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666;width:50%">{t['value']}</th>
    </tr>{qrows}
  </table>
</div>""")

    # ── Stock factor rankings ──────────────────────────────────────────────────
    if stock_factors:
        _SIG_COL = {"BUY": "#27ae60", "SELL": "#c0392b", "HOLD": "#95a5a6"}
        _SIG_ZH  = {"BUY": "买入", "SELL": "卖出", "HOLD": "持有"}
        sf_rows  = ""
        for s in stock_factors:
            # Prefer robust_signal; fall back to naive signal
            robust_sig = s.get("robust_signal")
            naive_sig  = s.get("signal", "HOLD")
            sig        = robust_sig if robust_sig else naive_sig
            sig_col    = _SIG_COL.get(sig, "#888")
            sig_lbl    = _SIG_ZH.get(sig, sig) if lang == "zh" else sig
            # Mark if robust signal differs from naive
            if robust_sig and robust_sig != naive_sig:
                naive_lbl = (_SIG_ZH.get(naive_sig, naive_sig) if lang == "zh" else naive_sig)
                sig_lbl  += f"<br><span style='font-size:10px;color:#bbb'>({naive_lbl})</span>"

            price  = f"${s['price']:.2f}" if s.get("price") else "—"
            alpha  = _pct(s.get("alpha_ann"))
            t_stat = _f(s.get("t_alpha"))
            # Use deflated IR when available, raw IR as fallback
            ir_val = s.get("ir_deflated", s.get("ir"))
            ir_str = (_f(ir_val, decimals=2) + ("*" if "ir_deflated" in s else "")
                      if ir_val is not None and ir_val == ir_val else "N/A")
            r2     = _f(s.get("r_squared"), decimals=2)
            bmkt   = _f(s.get("beta_mkt"))

            # Build robustness sub-text for sub-row
            robust_parts = []
            psr = s.get("psr")
            if psr is not None and psr == psr:
                psr_col = "#27ae60" if psr > 0.75 else ("#e67e22" if psr > 0.5 else "#c0392b")
                robust_parts.append(
                    f"<span style='color:{psr_col}'>PSR={psr*100:.0f}%</span>"
                )
            cons = s.get("alpha_consistency")
            nw   = s.get("n_windows", len((63, 126, 252)))
            if cons is not None:
                cons_col = "#27ae60" if cons >= nw - 1 else ("#e67e22" if cons >= 1 else "#c0392b")
                w_label  = "w一致" if lang == "zh" else "w consist."
                robust_parts.append(
                    f"<span style='color:{cons_col}'>{cons}/{nw}{w_label}</span>"
                )
            oos = s.get("oos_hit_rate")
            if oos is not None:
                oos_col = "#27ae60" if oos > 0.55 else ("#e67e22" if oos > 0.50 else "#c0392b")
                oos_lbl = "OOS命中" if lang == "zh" else "OOS hit"
                robust_parts.append(
                    f"<span style='color:{oos_col}'>{oos_lbl}={oos*100:.0f}%</span>"
                )

            # Build driver + robustness combined sub-row
            driver   = s.get("driver_zh" if lang == "zh" else "driver_en", "")
            sub_lines = []
            if driver and driver != "—":
                sub_lines.append(f"↳ {driver}")
            if robust_parts:
                sep = " &nbsp;·&nbsp; "
                sub_lines.append(sep.join(robust_parts))
            driver_row = ""
            if sub_lines:
                driver_row = (
                    f"<tr>"
                    f"<td colspan='8' style='padding:2px 8px 8px 20px;font-size:11px;"
                    f"background:#fafafa;border-bottom:1px solid #f0f0f0'>"
                    + "<br>".join(sub_lines)
                    + "</td></tr>"
                )

            bd = "0" if driver_row else "1px solid #f0f0f0"
            sf_rows += (
                f"<tr>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px'><b>{s['ticker']}</b></td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;color:#555'>{price}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;"
                f"font-weight:700;color:{sig_col}'>{sig_lbl}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;color:#555'>{alpha}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;color:#555'>{bmkt}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;color:#555'>{t_stat}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;font-weight:600'>{ir_str}</td>"
                f"<td style='padding:5px 8px;border-bottom:{bd};font-size:12px;color:#888'>{r2}</td>"
                f"</tr>"
                + driver_row
            )
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['factor_rankings']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['factor_rankings_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ticker']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['price']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['signal']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">α (ann.)</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">β MKT</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">t(α)</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ir_col']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">R²</th>
    </tr>{sf_rows}
  </table>
  <p style="font-size:10px;color:#ccc;margin:6px 0 0">{t['ir_deflated_note']}</p>
</div>""")

    # ── Signal weight allocation ──────────────────────────────────────────────
    if signal_weights:
        sw_rows = ""
        for e in signal_weights:
            w_pct   = f"{e['weight']*100:.1f}%"
            psr_v   = e.get("psr")
            psr_str = f"{psr_v*100:.0f}%" if (psr_v is not None and psr_v == psr_v) else "N/A"
            psr_col = "#27ae60" if (psr_v or 0) > 0.75 else ("#e67e22" if (psr_v or 0) > 0.5 else "#c0392b")
            ir_d    = e.get("ir_deflated")
            ir_str  = _f(ir_d, 2) if (ir_d is not None and ir_d == ir_d) else "N/A"
            cons    = e.get("alpha_consistency")
            cons_str = str(cons) if cons is not None else "—"
            sw_rows += (
                f"<tr>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'><b>{e['ticker']}</b></td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"font-weight:700;color:#2980b9'>{w_pct}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{psr_col}'>{psr_str}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{ir_str}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#888'>{cons_str}</td>"
                f"</tr>"
            )

        # Portfolio-level factor exposure warnings
        exp_warn_html = ""
        if signal_weights:
            port_exp  = signal_weights[0].get("port_factor_exposures", {})
            exp_warns = signal_weights[0].get("exposure_warnings", {})
            if port_exp:
                exp_cells = ""
                for fname, fval in port_exp.items():
                    fc = "#c0392b" if fname in exp_warns else "#555"
                    exp_cells += (
                        f"<span style='margin-right:10px;white-space:nowrap'>"
                        f"<span style='color:#888'>{fname}:</span> "
                        f"<b style='color:{fc}'>{_f(fval, 2)}</b></span>"
                    )
                warn_note = ""
                if exp_warns:
                    warn_names = ", ".join(exp_warns.keys())
                    warn_note = (
                        f"<p style='font-size:11px;color:#c0392b;margin:4px 0 0'>"
                        f"{t['sw_exp_warn'].format(factors=warn_names)}</p>"
                    )
                exp_warn_html = (
                    f"<p style='font-size:11px;color:#888;margin:6px 0 2px'>{t['sw_port_exp']}: "
                    f"{exp_cells}</p>{warn_note}"
                )

        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['sw_title']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['sw_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ticker']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['sw_weight']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">PSR</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ir_col']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['sw_consist']}</th>
    </tr>{sw_rows}
  </table>
  {exp_warn_html}
</div>""")

    # ── Execution / laddering plan ─────────────────────────────────────────────
    if execution:
        held_plans  = [e for e in execution if e.get("side") == "exit"]
        watch_plans = [e for e in execution if e.get("side") == "entry"]
        groups = []
        if held_plans:
            groups.append((t['exec_held'], held_plans))
        if watch_plans:
            groups.append((t['exec_watch'], watch_plans))

        group_html = ""
        for grp_label, plans in groups:
            group_html += (f"<p style='font-size:12px;font-weight:700;color:#555;"
                           f"margin:10px 0 4px'>{grp_label}</p>")
            for e in plans:
                group_html += _exec_card(e, t)

        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['exec_title']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['exec_note']}</p>
  {group_html}
</div>""")

    # ── Rebalancing suggestions ────────────────────────────────────────────────
    if suggestions:
        sug_rows = ""
        tier_labels = {
            1: (t['tier_act'],   _TIER_COLOR[1]),
            2: (t['tier_watch'], _TIER_COLOR[2]),
            3: (t['tier_hold'],  _TIER_COLOR[3]),
        }
        for s in suggestions:
            tier      = s["tier"]
            lbl, col  = tier_labels.get(tier, ("", "#888"))
            badge     = (f"<span style='background:{col};color:#fff;padding:2px 7px;"
                         f"border-radius:10px;font-size:11px;font-weight:700'>{lbl}</span>")
            direction = s["direction_zh"] if lang == "zh" else s["direction"]
            dir_col   = "#27ae60" if s["delta"] > 0 else "#c0392b"
            cgt_tag   = (f" <span style='color:#e67e22;font-size:10px'>{t['cgt_flag']}</span>"
                         if s.get("cgt_flag") else "")
            reason = s.get("reason_zh" if lang == "zh" else "reason", "")
            reason_html = (f"<br><span style='color:#aaa;font-size:10px'>{reason}</span>"
                           if reason else "")
            sug_rows += (
                f"<tr style='background:{'#fff8f8' if tier == 1 else '#fff'}'>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0'>{badge}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:13px'>"
                f"<b>{s['ticker']}</b>{cgt_tag}{reason_html}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555'>"
                f"{_pct(s['cur_weight'])}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555'>"
                f"{_pct(s['opt_weight'])}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{dir_col};font-weight:600'>{direction} {_pct(abs(s['delta']))}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555'>"
                f"{_eur(s['eur_change'])}</td>"
                f"</tr>"
            )
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['rebalance']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['rebalance_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['action']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ticker']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['cur_wt']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['opt_wt']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['change']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['eur_chg']}</th>
    </tr>{sug_rows}
  </table>
</div>""")

    # ── Diversification candidates (bonds / gold) ─────────────────────────────
    if div_candidates:
        usdeur = (fx_rates or {}).get("USDEUR", 0.92)
        gbpeur = (fx_rates or {}).get("GBPEUR", 1.17)

        _FX_RATE = {"USD": usdeur, "GBP": gbpeur, "EUR": 1.0}
        _CLS_LABEL = {
            "bond": ("Bond", "债券"),
            "gold": ("Gold", "黄金"),
        }
        _ASSET_ICON = {"bond": "🏦", "gold": "🥇"}

        dc_rows = ""
        for c in div_candidates:
            ticker    = c["ticker"]
            sw        = c["suggested_weight"]
            ds        = c["delta_sharpe"]
            dv        = c["delta_vol"]
            corr      = c["corr_to_portfolio"]
            ann_ret   = c["ann_ret"]
            ann_vol   = c["ann_vol"]
            ccy       = c.get("currency", "USD")
            cls_key   = c.get("asset_class", "bond")
            name      = c.get("name", ticker)

            cls_en, cls_zh = _CLS_LABEL.get(cls_key, (cls_key, cls_key))
            cls_label = cls_zh if lang == "zh" else cls_en

            # FX note for non-EUR assets
            fx_rate   = _FX_RATE.get(ccy, usdeur)
            fx_note   = (f"<br><span style='color:#aaa;font-size:10px'>"
                         f"{ccy} · 1{ccy}=€{fx_rate:.3f}</span>"
                         if ccy != "EUR" else "")

            # Suggested weight: show dash if at optimizer lower bound
            sw_str = f"{sw*100:.1f}%" if sw > 0.006 else ("—" if lang == "en" else "—")

            # Colour: green if positive impact, amber if neutral, grey if minimal
            if sw > 0.006 and ds > 0:
                row_bg  = "#f0fdf4"
                ds_col  = "#27ae60"
                verdict = t["dc_add"] if lang == "zh" else t["dc_add"]
            elif sw > 0.006:
                row_bg  = "#fef9f0"
                ds_col  = "#e67e22"
                verdict = t["dc_watch"]
            else:
                row_bg  = "#f9f9f9"
                ds_col  = "#95a5a6"
                verdict = t["dc_skip"]

            dv_col  = "#27ae60" if dv < 0 else "#e67e22"
            corr_col = "#27ae60" if corr < 0.3 else ("#e67e22" if corr < 0.6 else "#c0392b")

            dc_rows += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
                f"<b>{ticker}</b>{fx_note}<br>"
                f"<span style='color:#888;font-size:10px'>{name}</span></td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:11px;"
                f"color:#666'>{cls_label}</td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"font-weight:600'>{sw_str}</td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{ds_col};font-weight:600'>{'+' if ds>0 else ''}{ds:.2f}</td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{dv_col}'>{'+' if dv>0 else ''}{dv*100:.1f}%</td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;"
                f"color:{corr_col}'>{corr:+.2f}</td>"
                f"<td style='padding:7px 8px;border-bottom:1px solid #f0f0f0;font-size:11px;"
                f"color:#666'>{_pct(ann_ret)} / {_pct(ann_vol)}</td>"
                f"</tr>"
            )

        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['div_candidates']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">{t['div_note']}</p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ticker']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['dc_type']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['dc_weight']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">ΔSharpe</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">ΔVol</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['dc_corr']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['dc_stats']}</th>
    </tr>{dc_rows}
  </table>
</div>""")

    # ── Top 10 equity positions ────────────────────────────────────────────────
    equity_h = [h for h in holdings if h.get("asset_class", "equity") == "equity"]
    top10    = sorted(equity_h, key=lambda h: h["market_value_eur"], reverse=True)[:10]
    pos_rows = ""
    for h in top10:
        pnl_pct = h["unrealised_pnl_eur"] / h["cost_basis_eur"] if h.get("cost_basis_eur") else 0
        c       = "#27ae60" if pnl_pct >= 0 else "#c0392b"
        pos_rows += (
            f"<tr>"
            f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'><b>{h['ticker']}</b></td>"
            f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555'>{h['platform']}</td>"
            f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{_pct(h['weight'])}</td>"
            f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:{c}'>{_pct(pnl_pct)}</td>"
            f"</tr>"
        )

    sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['top10']}
  </h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['ticker']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['platform']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['weight']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['pnl_pct']}</th>
    </tr>{pos_rows}
  </table>
</div>""")

    # ── Bond holdings ──────────────────────────────────────────────────────────
    if bond_holdings:
        bond_rows = ""
        for h in sorted(bond_holdings, key=lambda h: h["market_value_eur"], reverse=True):
            desc    = h.get("description") or h["ticker"]
            mat     = h.get("maturity", "")
            cpn_raw = h.get("coupon", "")
            try:
                cpn_str = f"{float(cpn_raw):.3f}%" if cpn_raw else ""
            except (ValueError, TypeError):
                cpn_str = str(cpn_raw)
            detail  = " · ".join(filter(None, [cpn_str, mat]))
            pnl_pct = h["unrealised_pnl_eur"] / h["cost_basis_eur"] if h.get("cost_basis_eur") else 0
            c       = "#27ae60" if pnl_pct >= 0 else "#c0392b"
            detail_html = (f"<br><span style='color:#888;font-size:11px'>{detail}</span>"
                           if detail else "")
            bond_rows += (
                f"<tr>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>"
                f"<b>{desc}</b>{detail_html}"
                f"</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555'>{h['platform']}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{_pct(h['weight'])}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px'>{_eur(h['market_value_eur'])}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:{c}'>{_pct(pnl_pct)}</td>"
                f"</tr>"
            )
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['bond_holdings']}
  </h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['bond_desc']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['platform']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['weight']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['mkt_val']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['pnl_pct']}</th>
    </tr>{bond_rows}
  </table>
</div>""")

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    mc = (stress or {}).get("monte_carlo", {})
    if mc:
        mc_model   = mc.get("model", "GARCH-CC")
        mc_margin  = mc.get("margin_call_prob", 0.0) or 0.0
        margin_row = ""
        if mc_margin > 0.001:
            m_col = "#c0392b" if mc_margin > 0.05 else "#e67e22"
            margin_row = (
                f"<tr><td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;"
                f"font-size:12px;color:#555'>{t['mc_margin']}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid #f0f0f0;"
                f"font-size:12px;color:{m_col};font-weight:600'>"
                f"{_pct(mc_margin)}</td></tr>"
            )

        cgt_eur = mc.get("cgt_median_eur", 0.0) or 0.0
        tc_eur  = mc.get("trading_cost_eur", 0.0) or 0.0

        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['monte_carlo']}
  </h2>
  <p style="font-size:11px;color:#aaa;margin:0 0 8px">
    {mc.get('paths',0):,} {t['mc_paths']} · {mc.get('horizon_days',21)}{t['mc_days']} · {mc_model}
  </p>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f9ff">
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['metric']}</th>
      <th style="padding:5px 8px;text-align:left;font-size:12px;color:#666">{t['value']}</th>
    </tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      VaR (95%)</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;
          color:#c0392b;font-weight:600">
        {_pct(mc.get('var_95'))} &nbsp;<span style="color:#888;font-weight:400">({_eur(mc.get('eur_var_95'))})</span>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      VaR (99%)</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;
          color:#c0392b;font-weight:600">
        {_pct(mc.get('var_99'))} &nbsp;<span style="color:#888;font-weight:400">({_eur(mc.get('eur_var_99'))})</span>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      CVaR (95%)</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;
          color:#c0392b;font-weight:600">
        {_pct(mc.get('cvar_95'))} &nbsp;<span style="color:#888;font-weight:400">({_eur(mc.get('cvar_95_eur'))})</span>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      {t['mc_max_dd']}</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#e67e22;font-weight:600">
        {_pct(mc.get('max_dd_mean'))} &nbsp;<span style="color:#888;font-weight:400">{t['mc_p95']}: {_pct(mc.get('max_dd_p95'))}</span>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      {t['mc_loss_probs']}</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
        P(&gt;5%) <b>{_pct(mc.get('prob_loss_5pct',mc.get('var_95',0)))}</b> &nbsp;
        P(&gt;10%) <b>{_pct(mc.get('prob_loss_10pct'))}</b> &nbsp;
        P(&gt;20%) <b>{_pct(mc.get('prob_loss_20pct'))}</b>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      {t['mc_decision']}</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
        P(+) <b style="color:#27ae60">{_pct(mc.get('p_positive'))}</b> &nbsp;
        Sharpe <b>{_f(mc.get('sharpe_median'))}</b> &nbsp;
        Sortino <b>{_f(mc.get('sortino_median'))}</b>
      </td></tr>
    <tr><td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
      {t['mc_costs']}</td>
      <td style="padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#555">
        {t['mc_tc']} <b>{_eur(tc_eur)}</b> &nbsp; {t['mc_cgt']} <b>{_eur(cgt_eur)}</b>
      </td></tr>
    {margin_row}
  </table>
</div>""")

    # ── Correlation breakdown ──────────────────────────────────────────────────
    cb = (stress or {}).get("corr_breakdown", {})
    if cb:
        decay = cb.get("diversification_decay")
        decay_color = "#c0392b" if decay and decay > 1.3 else "#27ae60"
        sections.append(f"""
<div style="padding:0 20px 16px">
  <h2 style="font-size:14px;color:#444;margin:0 0 10px;border-bottom:1px solid #eee;padding-bottom:6px">
    {t['corr_breakdown']}
  </h2>
  <p style="font-size:13px;color:#555;margin:4px 0">
    {t['normal_sigma']} <b>{_pct(cb.get('normal_sigma'))}</b> &nbsp;→&nbsp;
    {t['crisis_sigma']} <b style="color:#c0392b">{_pct(cb.get('crisis_sigma'))}</b><br>
    {t['decay_ratio']} <b style="color:{decay_color}">{_f(decay)}×</b>
  </p>
</div>""")

    # ── Data quality footer ────────────────────────────────────────────────────
    excluded = [tk for tk, pd_obj in price_data.items() if pd_obj.flag]
    flags_html = ""
    if excluded:
        flags_html = "".join(
            f"<div style='font-size:11px;color:#888'>{tk} {price_data[tk].flag}</div>"
            for tk in excluded
        )

    sections.append(f"""
<div style="padding:12px 20px;background:#f8f9ff;border-top:1px solid #eee;border-radius:0 0 12px 12px">
  <p style="font-size:11px;color:#aaa;margin:0 0 4px">{t['footer_note']}</p>
  {flags_html}
</div>""")

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t['title']}</title></head>
<body style="{_S}background:#f5f6fa;margin:0;padding:16px">
<div style="max-width:600px;margin:0 auto;background:#fff;border-radius:12px;
     box-shadow:0 2px 12px rgba(0,0,0,.08);overflow:hidden">
{body}
</div>
</body></html>"""
    return html


# ── i18n strings ──────────────────────────────────────────────────────────────

_T = {
    "en": {
        "title":          "Portfolio Risk Report",
        "generated":      "Generated",
        "risk_level":     "Risk Level",
        "alert_banner":   "⚠ Active Alerts",
        "snapshot":       "Snapshot",
        "nav":            "Total NAV",
        "pnl":            "Unrealised P&L",
        "var95":          "VaR (95%)",
        "sharpe":         "Sharpe Ratio",
        "stress_top3":    "Stress Tests — Top 3 Worst Scenarios",
        "scenario":       "Scenario",
        "eur_loss":       "Est. Loss",
        "pct_loss":       "% Loss",
        "rr_chart":       "Risk–Return Chart",
        "rr_chart_note":  "Orange circle = current portfolio · Green diamond = max-Sharpe optimal · Dots = individual positions",
        "risk_indicators":"Risk Indicators",
        "metric":         "Metric",
        "value":          "Value",
        "rebalance":      "Portfolio Rebalancing Suggestions",
        "rebalance_note": "Act: |Δ|≥5% · Watch: 2–5% or Irish CGT (33%) applies · Hold: |Δ|<2%",
        "action":         "Action",
        "cur_wt":         "Current",
        "opt_wt":         "Optimal",
        "change":         "Direction",
        "eur_chg":        "Est. EUR",
        "tier_act":       "Act",
        "tier_watch":     "Watch",
        "tier_hold":      "Hold",
        "cgt_flag":       "⚑ CGT 33%",
        "top10":          "Top 10 Positions",
        "ticker":         "Ticker",
        "platform":       "Platform",
        "weight":         "Weight",
        "pnl_pct":        "P&L%",
        "monte_carlo":    "Monte Carlo Simulation",
        "mc_paths":       "paths",
        "mc_days":        "-day horizon",
        "mc_max_dd":      "Max Drawdown (avg paths)",
        "mc_p95":         "95th pct",
        "mc_loss_probs":  "Loss Probabilities",
        "mc_decision":    "Decision Metrics",
        "mc_costs":       "Liquidation Costs",
        "mc_tc":          "Trade:",
        "mc_cgt":         "CGT (median):",
        "mc_margin":      "Margin Call Probability",
        "corr_breakdown": "Correlation Breakdown",
        "normal_sigma":   "Normal σ:",
        "crisis_sigma":   "Crisis σ:",
        "decay_ratio":    "Diversification decay:",
        "bond_exposure":  "Bond Exposure",
        "bond_holdings":  "Bond Holdings",
        "bond_desc":      "Description",
        "mkt_val":        "Market Value",
        "div_candidates": "Diversification Candidates",
        "div_note":       "Marginal impact on equity portfolio at optimizer-suggested weight. ΔSharpe > 0 = beneficial. Corr < 0.3 = good diversifier.",
        "dc_type":        "Type",
        "dc_weight":      "Sug. Weight",
        "dc_corr":        "Corr",
        "dc_stats":       "Ret / Vol",
        "dc_add":         "Consider",
        "dc_watch":       "Monitor",
        "dc_skip":        "Low benefit",
        "port_exposure":       "Portfolio Factor Exposures (FF5)",
        "port_exposure_note":  "Market-value-weighted average of Fama-French 5-factor betas across equity holdings.",
        "pe_factor":           "Factor",
        "pe_t":                "t-stat",
        "pe_sig":              "Sig.",
        "pe_se_note":          "±SE: pooled from holdings (independence assumed — lower bound).",
        "attribution":         "Performance Attribution (FF5 — last {n} days)",
        "attribution_note":    "{start} → {end} · {n} trading days · equity holdings only · betas from 252-day OLS",
        "attr_factor":         "Factor",
        "attr_factor_ret":     "Factor return",
        "attr_contrib":        "Contribution",
        "attr_alpha_gross":    "Alpha (gross, stock selection)",
        "attr_trading_cost":   "Trading cost",
        "attr_alpha_net":      "Alpha (net of cost)",
        "attr_error":          "Attribution error: ±{e}",
        "attr_ols_note":       "t-stats via direct portfolio OLS ({n} days)",
        "attr_sig_note":       "★★★ |t|≥3.0 highly significant · ★★ |t|≥2.0 significant · ★ |t|≥1.5 marginal",
        "factor_rankings":      "Stock Factor Rankings (FF5 — sorted by IR)",
        "factor_rankings_note": "Signal uses robust filter: PSR>85% + multi-window consistency + deflated IR. Naive t(α) signal shown in brackets if it differs. Sub-row: 63-day attribution · PSR · window consistency · OOS hit rate.",
        "ir_col":               "IR*",
        "ir_deflated_note":     "IR* = deflated for multiple-testing (Bailey–Lopez de Prado 2014). PSR = P(true α>0) corrected for non-normality. OOS = rolling 126d-train/21d-test hit rate.",
        "price":               "Price",
        "signal":              "Signal",
        "fmq_title":           "Factor Model Quality",
        "fmq_note":            "IC = cross-sectional Spearman rank correlation between predicted and realised alpha ranks. VIF > 5 indicates collinearity concerns.",
        "fmq_ic_label":        "IC mean ({fwd}-day forward)",
        "fmq_icir_label":      "IC-IR (annualised)",
        "fmq_pval_label":      "p-value (two-tailed)",
        "fmq_nobs_label":      "# evaluation periods",
        "fmq_cond":            "Condition number (FF5 corr)",
        "fmq_vif":             "VIF per factor",
        "fmq_high_pairs":      "High-corr pairs (|r|>0.4)",
        "fmq_warn":            "Warning",
        "sw_title":            "Signal-Weighted Allocation (BUY signals)",
        "sw_note":             "Score = PSR × deflated IR × consistency boost. Weights capped at 20% and renormalised. Eligible: robust_signal=BUY, PSR≥65%, IR_deflated>0.",
        "sw_weight":           "Weight",
        "sw_consist":          "Consistency",
        "sw_port_exp":         "Portfolio factor exposure",
        "sw_exp_warn":         "Exposure warning: {factors} exceed threshold",
        "exec_title":          "Execution Plan — Laddered Entry / Exit",
        "exec_note":           "Trade execution only — the BUY/SELL decision comes from the factor model. Every name gets both a sell ladder (above price) and a buy ladder (below price); ★ marks the primary intent. Rungs are limit levels spaced by ATR, snapped to high-volume nodes (HVN), and sized by liquidity (×shares). The ADV cap is the hard constraint: ! flags a rung above the daily cap.",
        "exec_held":           "Holdings (primary: sell ladder)",
        "exec_watch":          "Watchlist BUY (primary: buy ladder)",
        "exec_verdict":        "Action",
        "exec_anchor":         "vs anchored VWAP",
        "exec_ladder":         "Ladder",
        "exec_ladder_up":      "↑ Sell ladder (scale-out)",
        "exec_ladder_down":    "↓ Buy ladder (scale-in)",
        "exec_adv_cap":        "ADV cap",
        "exec_days":           "≈{d}d to exit",
        "exec_opex":           "OPEX",
        "exec_maxpain":        "max-pain",
        "exec_gamma":          "γ-peak",
        "exec_hvn_tag":        "HVN",
        "exec_v_reduce":       "REDUCE",
        "exec_v_observe":      "OBSERVE",
        "exec_v_accumulate":   "ACCUMULATE",
        "exec_v_wait":         "WAIT",
        "footer_note":    "For informational purposes only. Not investment advice.",
        "alert_labels": {
            "var_95":    "VaR (95%) > 5% NAV",
            "cvar_ratio":"CVaR/VaR ratio > 1.8×",
            "max_dd":    "Max Drawdown > 20%",
            "max_pos":   "Single position > 30%",
            "hhi":       "HHI > 0.25",
            "beta":      "Beta > 1.5",
            "sharpe":    "Sharpe < 0.5",
            "daily_loss":"Single-day loss > 3%",
        },
    },
    "zh": {
        "title":          "投资组合风险报告",
        "generated":      "生成于",
        "risk_level":     "风险等级",
        "alert_banner":   "⚠ 当前预警",
        "snapshot":       "核心指标快照",
        "nav":            "总市值",
        "pnl":            "未实现盈亏",
        "var95":          "VaR（95%）",
        "sharpe":         "夏普比率",
        "stress_top3":    "压力测试 — 最差三个情景",
        "scenario":       "情景",
        "eur_loss":       "预估损失",
        "pct_loss":       "损失比例",
        "rr_chart":       "风险–收益图",
        "rr_chart_note":  "橙圈=当前组合 · 绿菱=最优组合（最大夏普） · 蓝点=单个持仓",
        "risk_indicators":"风险指标",
        "metric":         "指标",
        "value":          "数值",
        "rebalance":      "组合再平衡建议",
        "rebalance_note": "立即行动：|Δ|≥5% · 观察：2–5%或涉及爱尔兰CGT(33%) · 维持：|Δ|<2%",
        "action":         "行动",
        "cur_wt":         "当前权重",
        "opt_wt":         "最优权重",
        "change":         "方向",
        "eur_chg":        "预估金额(€)",
        "tier_act":       "立即行动",
        "tier_watch":     "观察",
        "tier_hold":      "维持",
        "cgt_flag":       "⚑ CGT 33%",
        "top10":          "前十大持仓",
        "ticker":         "代码",
        "platform":       "平台",
        "weight":         "权重",
        "pnl_pct":        "盈亏%",
        "monte_carlo":    "蒙特卡洛模拟",
        "mc_paths":       "条路径",
        "mc_days":        "天期限",
        "mc_max_dd":      "最大回撤（路径均值）",
        "mc_p95":         "95分位",
        "mc_loss_probs":  "亏损概率",
        "mc_decision":    "决策指标",
        "mc_costs":       "清仓成本",
        "mc_tc":          "交易成本：",
        "mc_cgt":         "CGT税（中位）：",
        "mc_margin":      "追加保证金概率",
        "corr_breakdown": "相关性击穿分析",
        "normal_sigma":   "正常市场σ:",
        "crisis_sigma":   "危机市场σ:",
        "decay_ratio":    "分散化折损:",
        "bond_exposure":  "债券敞口",
        "bond_holdings":  "债券持仓",
        "bond_desc":      "债券名称",
        "mkt_val":        "市值",
        "div_candidates": "多元化候选资产",
        "div_note":       "按优化器建议权重加入时对股票组合的边际影响。ΔSharpe > 0 = 有益，相关性 < 0.3 = 良好分散效果。",
        "dc_type":        "类型",
        "dc_weight":      "建议配置",
        "dc_corr":        "相关性",
        "dc_stats":       "收益/波动",
        "dc_add":         "可考虑",
        "dc_watch":       "持续观察",
        "dc_skip":        "效益有限",
        "port_exposure":       "组合因子暴露（FF5）",
        "port_exposure_note":  "按市值加权的Fama-French五因子Beta，反映当前股票持仓的整体风格偏向。",
        "pe_factor":           "因子",
        "pe_t":                "t值",
        "pe_sig":              "显著性",
        "pe_se_note":          "±SE为各持仓加权传播值，独立性假设下为下界。",
        "attribution":         "业绩归因（FF5 — 近{n}个交易日）",
        "attribution_note":    "{start} → {end} · {n}个交易日 · 仅股票持仓 · Beta来自252日OLS回归",
        "attr_factor":         "因子",
        "attr_factor_ret":     "因子收益",
        "attr_contrib":        "贡献度",
        "attr_alpha_gross":    "Alpha（毛，选股贡献）",
        "attr_trading_cost":   "交易成本",
        "attr_alpha_net":      "Alpha（扣成本后）",
        "attr_error":          "归因误差：±{e}",
        "attr_ols_note":       "t值来自组合直接OLS回归（{n}天）",
        "attr_sig_note":       "★★★ |t|≥3.0高度显著 · ★★ |t|≥2.0显著 · ★ |t|≥1.5边际显著",
        "factor_rankings":      "个股因子排名（FF5 — 按IR排序）",
        "factor_rankings_note": "信号使用稳健过滤：PSR>85% + 多窗口一致性 + 多重检验修正IR。括号内为单因子朴素信号（如不同）。子行：63天归因 · PSR · 窗口一致性 · OOS命中率。",
        "ir_col":               "IR*",
        "ir_deflated_note":     "IR* = 多重检验修正后IR（Bailey–Lopez de Prado 2014）。PSR = 考虑残差非正态性后的真实α>0概率。OOS = 滚动126天训练/21天测试的信号准确率。",
        "price":               "价格",
        "signal":              "信号",
        "fmq_title":           "因子模型质量",
        "fmq_note":            "IC = 预测Alpha排名与实际Alpha排名的截面Spearman秩相关。VIF > 5 表示存在共线性风险。",
        "fmq_ic_label":        "IC均值（{fwd}日前瞻）",
        "fmq_icir_label":      "IC-IR（年化）",
        "fmq_pval_label":      "p值（双尾检验）",
        "fmq_nobs_label":      "评估期数",
        "fmq_cond":            "条件数（FF5相关矩阵）",
        "fmq_vif":             "各因子VIF",
        "fmq_high_pairs":      "高相关因子对（|r|>0.4）",
        "fmq_warn":            "警告",
        "sw_title":            "信号加权配置（买入信号）",
        "sw_note":             "评分 = PSR × 多重检验修正IR × 一致性加成。权重上限20%，已归一化。条件：robust_signal=BUY，PSR≥65%，IR_deflated>0。",
        "sw_weight":           "权重",
        "sw_consist":          "一致性",
        "sw_port_exp":         "组合因子敞口",
        "sw_exp_warn":         "敞口警告：{factors} 超过阈值",
        "exec_title":          "执行计划 — 分批建仓 / 减仓阶梯",
        "exec_note":           "仅为交易执行方案，买卖决策由因子模型给出。每只标的同时给出上升减仓阶梯（价上方）与下降建仓阶梯（价下方）；★ 为主要方向。阶梯档位为限价单，按 ATR 间距排列、贴近高成交量区（HVN），并按流动性分配股数（×股数）。ADV 上限为硬约束：! 表示该档位超过单日上限。",
        "exec_held":           "持仓（主：减仓阶梯）",
        "exec_watch":          "观察名单买入（主：建仓阶梯）",
        "exec_verdict":        "操作",
        "exec_anchor":         "相对锚定VWAP",
        "exec_ladder":         "阶梯",
        "exec_ladder_up":      "↑ 减仓阶梯（分批卖出）",
        "exec_ladder_down":    "↓ 建仓阶梯（分批买入）",
        "exec_adv_cap":        "ADV上限",
        "exec_days":           "≈{d}天退出",
        "exec_opex":           "期权到期",
        "exec_maxpain":        "最大痛点",
        "exec_gamma":          "γ峰值",
        "exec_hvn_tag":        "HVN",
        "exec_v_reduce":       "减仓",
        "exec_v_observe":      "观望",
        "exec_v_accumulate":   "建仓",
        "exec_v_wait":         "等待",
        "footer_note":    "本报告仅供参考，不构成投资建议。",
        "alert_labels": {
            "var_95":    "VaR(95%) > 组合市值5%",
            "cvar_ratio":"CVaR/VaR > 1.8倍",
            "max_dd":    "最大回撤 > 20%",
            "max_pos":   "单仓权重 > 30%",
            "hhi":       "HHI > 0.25",
            "beta":      "贝塔 > 1.5",
            "sharpe":    "夏普 < 0.5",
            "daily_loss":"单日亏损 > 3%",
        },
    },
}
