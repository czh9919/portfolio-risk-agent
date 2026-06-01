"""
M7 — Alert system.
Checks metrics against thresholds and sends bilingual alert emails.
Deduplication: same alert suppressed for 24h via a local JSON file.
"""
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from notify.mailer import send_combined_alert

logger = logging.getLogger(__name__)

DEDUP_FILE = Path("cache/alert_dedup.json")


def _load_dedup() -> dict:
    if DEDUP_FILE.exists():
        try:
            return json.loads(DEDUP_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_dedup(data: dict):
    DEDUP_FILE.parent.mkdir(exist_ok=True)
    DEDUP_FILE.write_text(json.dumps(data))


def _is_suppressed(key: str) -> bool:
    dedup = _load_dedup()
    if key not in dedup:
        return False
    fired_at = datetime.fromisoformat(dedup[key])
    return datetime.utcnow() - fired_at < timedelta(hours=24)


def _mark_fired(key: str):
    dedup = _load_dedup()
    dedup[key] = datetime.utcnow().isoformat()
    _save_dedup(dedup)


# risk_engine uses short alert keys; thresholds.yaml uses longer descriptive keys
_ALERT_TO_THRESH_KEY = {
    "var_95":    "var_95_pct",
    "cvar_ratio":"cvar_var_ratio",
    "max_dd":    "max_drawdown",
    "max_pos":   "max_single_position",
    "daily_loss":"single_day_loss",
    # hhi, beta, sharpe match directly
}


def check_and_alert(metrics: dict, thresholds: dict):
    """
    Collect all RED metrics not yet sent today, then fire ONE combined email.
    Reduces alert volume from 2×N emails to a single digest per check cycle.
    """
    alert_map = metrics.get("alerts", {})
    t_cfg     = thresholds.get("alerts", {})

    to_fire: list[dict] = []
    for key, rag in alert_map.items():
        if rag != "RED":
            continue
        dedup_key = f"{date.today()}_{key}"
        if _is_suppressed(dedup_key):
            logger.info(f"Alert {key} suppressed (already sent today)")
            continue

        cfg      = t_cfg.get(_ALERT_TO_THRESH_KEY.get(key, key), {})
        value    = _get_value(metrics, key)
        severity = cfg.get("severity", "HIGH")
        logger.warning(
            f"ALERT [{severity}] {cfg.get('label_en', key)}: "
            f"{value} vs threshold {cfg.get('threshold', '?')}"
        )
        to_fire.append({
            "key":       key,
            "dedup_key": dedup_key,
            "label_en":  cfg.get("label_en", key),
            "label_zh":  cfg.get("label_zh", key),
            "value":     value,
            "threshold": str(cfg.get("threshold", "?")),
            "severity":  severity,
        })

    if not to_fire:
        return False

    send_combined_alert(to_fire)
    for a in to_fire:
        _mark_fired(a["dedup_key"])
    return True


def _get_value(metrics: dict, key: str) -> str:
    mapping = {
        "var_95":     lambda m: f"{m.get('var_95_cf', m.get('var_95_ewma', 0))*100:.1f}%",
        "cvar_ratio": lambda m: f"{m.get('cvar_var_ratio', 0):.2f}×",
        "max_dd":     lambda m: f"{abs(m.get('max_drawdown', 0))*100:.1f}%",
        "max_pos":    lambda m: f"{m.get('max_position_wt', 0)*100:.1f}%",
        "hhi":        lambda m: f"{m.get('hhi', 0):.3f}",
        "beta":       lambda m: f"{m.get('beta', 0):.2f}",
        "sharpe":     lambda m: f"{m.get('sharpe', 0):.2f}",
        "daily_loss": lambda m: f"{-m.get('daily_return', 0)*100:.1f}%",
    }
    fn = mapping.get(key)
    return fn(metrics) if fn else "?"
