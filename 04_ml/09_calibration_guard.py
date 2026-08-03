#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarda operacional de calibração/modelo.

Bloqueia ou alerta a operação quando o histórico real mostra degradação de
calibração acima do limite. Usa os arquivos gerados pelo 04_monitor_drift.py.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DRIFT_SUMMARY_JSON = REPORTS_DIR / "resumo_deriva.json"
GUARD_STATE_JSON = REPORTS_DIR / "estado_guard_calibracao.json"
BASELINE_METRICS_JSON = MODELS_DIR / "baseline_metrics.json"

# Política conservadora: warning permite operar; critical bloqueia auto-fill.
BLOCK_ON_ALERT_LEVELS = {"CRITICO"}
MAX_BRIER_DELTA_PCT = 0.05


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_alert(value) -> str:
    return str(value or "NO_DATA").strip().upper()


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    drift = load_json(DRIFT_SUMMARY_JSON)
    baseline = load_json(BASELINE_METRICS_JSON)

    status = "OK"
    motivos = []

    if not drift:
        status = "NO_DRIFT_REPORT"
        motivos.append("resumo_deriva.json não encontrado; rode 04_monitor_drift.py antes do guard.")
    else:
        overall = drift.get("overall") or {}
        alert = normalize_alert(overall.get("alert_level"))
        if alert in BLOCK_ON_ALERT_LEVELS:
            status = "BLOQUEADA"
            motivos.append(f"alerta de calibração em {alert}")

        current_brier = overall.get("brier_score")
        baseline_brier = (baseline.get("overall") or {}).get("brier_score") or baseline.get("brier_score")
        try:
            if current_brier is not None and baseline_brier is not None and float(baseline_brier) > 0:
                delta_pct = (float(current_brier) - float(baseline_brier)) / float(baseline_brier)
                if delta_pct > MAX_BRIER_DELTA_PCT:
                    status = "BLOQUEADA"
                    motivos.append(f"Brier Score piorou {delta_pct:.1%} vs baseline")
        except Exception:
            pass

    state = {
        "generated_at": utc_now_iso(),
        "status": status,
        "blocked": status == "BLOQUEADA",
        "motivos": motivos,
        "drift_summary_json": str(DRIFT_SUMMARY_JSON),
        "baseline_metrics_json": str(BASELINE_METRICS_JSON),
        "policy": {
            "block_on_alert_levels": sorted(BLOCK_ON_ALERT_LEVELS),
            "max_brier_delta_pct": MAX_BRIER_DELTA_PCT,
        },
    }
    GUARD_STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    if state["blocked"]:
        print("🚫 Guarda de calibração BLOQUEOU operação automática.")
        for motivo in motivos:
            print(f"  - {motivo}")
        print(f"Estado salvo em: {GUARD_STATE_JSON}")
        return 2

    if status == "NO_DRIFT_REPORT":
        print("⚠️ Guarda de calibração sem relatório de drift; operação não foi bloqueada, mas precisa rodar monitoramento.")
        print(f"Estado salvo em: {GUARD_STATE_JSON}")
        return 0

    print("✅ Guarda de calibração OK. Operação automática liberada.")
    print(f"Estado salvo em: {GUARD_STATE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
