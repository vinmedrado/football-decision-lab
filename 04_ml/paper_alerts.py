#!/usr/bin/env python3
"""Gera alerta local e, opcionalmente, envia resumo por SMTP."""
from __future__ import annotations

import hashlib
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MONITOR_PATH = BASE_DIR / "reports" / "paper_monitor.json"
AUTOMATION_PATH = BASE_DIR / "reports" / "automation_state.json"
ALERT_PATH = BASE_DIR / "reports" / "paper_alert_status.json"
SENT_PATH = BASE_DIR / "reports" / "paper_alert_sent.json"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def pct(value) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def build_alert() -> dict:
    monitor = read_json(MONITOR_PATH, {})
    automation = read_json(AUTOMATION_PATH, {})
    metrics = monitor.get("metrics", {})
    severity = "INFO"
    reasons = []
    if monitor.get("calibration_status") == "ALERTA":
        severity = "ALERTA"
        reasons.append("Erro de calibração acima de 10% com amostra mínima.")
    if monitor.get("status") == "EVIDENCIA_NEGATIVA":
        severity = "CRITICO"
        reasons.append("Intervalo de ROI ficou totalmente abaixo de zero.")
    failed = []
    for name, job in (automation.get("jobs", {}) or {}).items():
        slots = job.get("slots", {}) if isinstance(job, dict) else {}
        valid_slots = [(key, value) for key, value in slots.items() if isinstance(value, dict)]
        latest = max(valid_slots, key=lambda item: item[0])[1] if valid_slots else {}
        if latest.get("status") == "failed":
            failed.append(name)
    if failed:
        severity = "CRITICO"
        reasons.append("Rotinas com falha: " + ", ".join(sorted(set(failed))))
    if not reasons:
        reasons.append("Rotinas sem alerta crítico; experimento continua coletando dados.")
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "severity": severity,
        "cycle_id": monitor.get("cycle_id"),
        "status": monitor.get("status"),
        "reasons": reasons,
        "summary": {
            "settled": metrics.get("settled", 0),
            "pending": metrics.get("pending", 0),
            "profit": metrics.get("profit", 0),
            "roi": metrics.get("roi"),
            "roi_ci95": [metrics.get("ci95_low"), metrics.get("ci95_high")],
            "brier": metrics.get("brier"),
            "calibration_error": metrics.get("calibration_error"),
        },
    }


def maybe_send(payload: dict) -> bool:
    enabled = os.getenv("ALERT_EMAIL_ENABLED", "").strip().lower() in {"1", "true", "sim", "yes"}
    if not enabled:
        return False
    required = ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Configuração de e-mail incompleta: " + ", ".join(missing))
    day_key = datetime.now().astimezone().date().isoformat()
    fingerprint_source = json.dumps({"day": day_key, "severity": payload["severity"], "summary": payload["summary"]}, sort_keys=True)
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    sent = read_json(SENT_PATH, {})
    if sent.get("fingerprint") == fingerprint:
        return False
    summary = payload["summary"]
    message = EmailMessage()
    message["Subject"] = f"[Football Lab] {payload['severity']} — ROI paper {pct(summary.get('roi'))}"
    message["From"] = os.environ["ALERT_EMAIL_FROM"]
    message["To"] = os.environ["ALERT_EMAIL_TO"]
    message.set_content(
        "\n".join([
            f"Ciclo: {payload.get('cycle_id')}",
            f"Status: {payload.get('status')}",
            f"Liquidadas: {summary.get('settled')} | Pendentes: {summary.get('pending')}",
            f"Lucro: R$ {float(summary.get('profit') or 0):.2f} | ROI: {pct(summary.get('roi'))}",
            f"Brier: {summary.get('brier')} | Erro de calibração: {pct(summary.get('calibration_error'))}",
            "",
            *payload["reasons"],
        ])
    )
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    use_ssl = os.getenv("SMTP_SSL", "").strip().lower() in {"1", "true", "yes"}
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context()) as server:
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(message)
    SENT_PATH.write_text(json.dumps({"fingerprint": fingerprint, "sent_at": payload["generated_at"]}, indent=2), encoding="utf-8")
    return True


def main() -> int:
    payload = build_alert()
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sent = maybe_send(payload)
    print(json.dumps({**payload, "email_sent": sent}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
