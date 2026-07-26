import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

BACKUP_DIR = Path("data/email_backup")


class Mailer:
    def __init__(self, config: dict):
        self.config = config

    def send(self, html_body: str, subject: str | None = None) -> bool:
        """Send report or alert email. Writes local backup on failure. Returns True on success."""
        if subject is None:
            subject = f"{self.config.get('subject_prefix', '[Stock Agent]')} Daily Report {date.today()}"
        recipient = self.config.get("recipient") or os.environ.get("RECIPIENT_EMAIL")

        if not recipient:
            logger.error("No recipient email configured")
            return False

        use_smtp = self.config.get("use_smtp", False)
        try:
            if use_smtp:
                success = self._send_smtp(recipient, subject, html_body)
            else:
                success = self._send_sendgrid(recipient, subject, html_body)

            if success:
                logger.info(f"Email sent to {recipient}")
            return success
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            self._backup(html_body)
            return False

    def _send_sendgrid(self, recipient: str, subject: str, html: str) -> bool:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        api_key = os.environ.get("SENDGRID_API_KEY") or self.config.get("sendgrid_api_key")
        if not api_key:
            logger.error("SENDGRID_API_KEY not set")
            self._backup(html)
            return False

        sender = self.config.get("sender", "stock-agent@example.com")
        message = Mail(
            from_email=sender,
            to_emails=recipient,
            subject=subject,
            html_content=html,
        )
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        return response.status_code in (200, 202)

    def _send_smtp(self, recipient: str, subject: str, html: str) -> bool:
        host     = os.environ.get("SMTP_HOST") or self.config.get("smtp_host", "smtp.gmail.com")
        port     = int(os.environ.get("SMTP_PORT") or self.config.get("smtp_port", 587))
        user     = os.environ.get("SMTP_USER") or self.config.get("smtp_user", "")
        password = os.environ.get("SMTP_PASS") or os.environ.get("SMTP_PASSWORD") or self.config.get("smtp_password", "")
        sender   = self.config.get("sender", user)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(sender, recipient, msg.as_string())
        return True

    def _backup(self, html: str):
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        path = BACKUP_DIR / f"report_{date.today()}.html"
        path.write_text(html, encoding="utf-8")
        logger.info(f"Email backup written to {path}")


# ── Portfolio risk helpers (standalone, no config object needed) ──────────────

def _smtp_send(html: str, subject: str, plain: str = "",
               chart_bytes: bytes = b"", cid: str = "rrChart",
               images: list[tuple[str, bytes]] | None = None) -> bool:
    """Low-level SMTP send using env vars directly.

    Inline images can be embedded either via the legacy single-image slot
    (`chart_bytes` + `cid`) or via `images=[(cid, png_bytes), ...]` for
    multi-chart emails (e.g. macro panel). If both are given, the legacy
    slot is treated as one more entry in `images`.
    """
    from email.mime.image import MIMEImage

    host      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port      = int(os.environ.get("SMTP_PORT", 587))
    user      = os.environ.get("SMTP_USER", "")
    password  = os.environ.get("SMTP_PASS", "")
    recipient = os.environ.get("REPORT_TO_EMAIL") or os.environ.get("RECIPIENT_EMAIL", "")

    if not recipient:
        logger.error("No recipient configured (REPORT_TO_EMAIL / RECIPIENT_EMAIL)")
        return False

    all_images: list[tuple[str, bytes]] = list(images) if images else []
    if chart_bytes:
        all_images.append((cid, chart_bytes))

    if all_images:
        # multipart/related wraps the HTML + inline PNGs so Gmail renders them
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = user
        msg["To"]      = recipient
        alt = MIMEMultipart("alternative")
        if plain:
            alt.attach(MIMEText(plain, "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
        for img_cid, img_bytes in all_images:
            img = MIMEImage(img_bytes, _subtype="png")
            img.add_header("Content-ID", f"<{img_cid}>")
            img.add_header("Content-Disposition", "inline",
                           filename=f"{img_cid}.png")
            msg.attach(img)
    else:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = user
        msg["To"]      = recipient
        if plain:
            msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(user, recipient, msg.as_string())
        logger.info(f"Portfolio email sent: {subject[:60]}")
        return True
    except Exception as e:
        logger.error(f"Portfolio email failed: {e}")
        backup = BACKUP_DIR / f"portfolio_{date.today()}.html"
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup.write_text(html, encoding="utf-8")
        return False


def send_report(html_en: str, html_zh: str, rag: str = "GREEN",
                chart_en: bytes = b"", chart_zh: bytes = b""):
    """Send bilingual daily portfolio risk report with optional inline charts."""
    rag_label    = {"RED": "RED", "AMBER": "AMBER", "GREEN": "GREEN"}.get(rag, rag)
    rag_label_zh = {"RED": "红",  "AMBER": "黄",    "GREEN": "绿"}.get(rag, rag)
    today = date.today().isoformat()
    _smtp_send(html_en, f"[Daily] Portfolio Report — {today} — Risk: {rag_label}",
               chart_bytes=chart_en)
    _smtp_send(html_zh, f"[日报] 投资组合报告 — {today} — 风险等级: {rag_label_zh}",
               chart_bytes=chart_zh)


def send_alert(metric: str, value: str, threshold: str, metric_zh: str = ""):
    """Send bilingual threshold breach alert (single metric, legacy)."""
    html_en = f"<p><b>{metric}</b> breached threshold.<br>Current: {value} | Threshold: {threshold}</p>"
    html_zh = f"<p><b>{metric_zh or metric}</b> 触发预警。<br>当前值: {value}，阈值: {threshold}</p>"
    _smtp_send(html_en, f"[ALERT] {metric} breached — {value} vs {threshold}")
    _smtp_send(html_zh, f"[预警] {metric_zh or metric} 触发阈值 — {value} / {threshold}")


def send_combined_alert(alerts: list[dict]):
    """
    Send ONE bilingual email summarising all currently-RED metrics.

    Each alert dict must contain:
      label_en, label_zh, value, threshold, severity
    """
    if not alerts:
        return
    today = date.today().isoformat()
    count = len(alerts)

    sev_color = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#f39c12"}
    sev_bg    = {"HIGH": "#fdf0ef", "MEDIUM": "#fef5ec", "LOW": "#fefbe8"}

    def _alert_rows() -> str:
        rows = ""
        for a in alerts:
            sev  = a.get("severity", "HIGH")
            fc   = sev_color.get(sev, "#e74c3c")
            bg   = sev_bg.get(sev, "#fdf0ef")
            rows += f"""
<tr style="background:{bg}">
  <td style="padding:12px 10px;border-bottom:1px solid #eee;word-break:break-word">
    <span style="font-size:14px;font-weight:bold;color:#2c3e50">{a['label_en']}</span><br>
    <span style="font-size:12px;color:#7f8c8d">{a.get('label_zh') or a['label_en']}</span>
  </td>
  <td style="padding:12px 10px;border-bottom:1px solid #eee;text-align:center;
             font-size:16px;font-weight:bold;color:{fc};white-space:nowrap">
    {a['value']}
  </td>
  <td style="padding:12px 10px;border-bottom:1px solid #eee;text-align:center;
             font-size:13px;color:#7f8c8d;white-space:nowrap">
    {a['threshold']}
  </td>
  <td style="padding:12px 10px;border-bottom:1px solid #eee;text-align:center;
             white-space:nowrap">
    <span style="background:{fc};color:#fff;font-size:11px;font-weight:bold;
                 padding:3px 7px;border-radius:3px">{sev}</span>
  </td>
</tr>"""
        return rows

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Portfolio Alert</title>
</head>
<body style="margin:0;padding:0;background:#f5f6fa;font-family:Arial,Helvetica,sans-serif">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f6fa">
<tr><td align="center" style="padding:16px 8px">

  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:580px;background:#fff;border-radius:6px;
                overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)">

    <!-- Header -->
    <tr><td style="background:#e74c3c;padding:20px 20px 16px">
      <p style="margin:0;font-size:22px;font-weight:bold;color:#fff">
        &#9888; Portfolio Alert
      </p>
      <p style="margin:4px 0 0;font-size:13px;color:rgba(255,255,255,.8)">
        {count} metric(s) breached &nbsp;·&nbsp; {today}
      </p>
      <p style="margin:2px 0 0;font-size:12px;color:rgba(255,255,255,.7)">
        {count} 项指标触发阈值
      </p>
    </td></tr>

    <!-- Alert table -->
    <tr><td style="padding:0">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr style="background:#2c3e50">
          <th style="padding:10px;text-align:left;font-size:12px;
                     color:#fff;font-weight:bold;width:40%">
            Metric / 指标
          </th>
          <th style="padding:10px;text-align:center;font-size:12px;
                     color:#fff;font-weight:bold;width:20%">
            Current / 当前
          </th>
          <th style="padding:10px;text-align:center;font-size:12px;
                     color:#fff;font-weight:bold;width:20%">
            Limit / 阈值
          </th>
          <th style="padding:10px;text-align:center;font-size:12px;
                     color:#fff;font-weight:bold;width:20%">
            Level
          </th>
        </tr>
        {_alert_rows()}
      </table>
    </td></tr>

    <!-- Footer -->
    <tr><td style="padding:14px 20px;border-top:1px solid #ecf0f1">
      <p style="margin:0;font-size:11px;color:#bdc3c7;text-align:center">
        Stock AI Agent &nbsp;·&nbsp; Review full report for details.
      </p>
    </td></tr>

  </table>
</td></tr>
</table>
</body></html>
"""

    subject_en = f"[ALERT] {count} metric(s) breached — {today}"
    _smtp_send(html, subject_en)
