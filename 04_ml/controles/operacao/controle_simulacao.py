#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controle de simulação e proteção operacional.

Avalia risco histórico por mercado e drawdown para impedir uso cego do sistema.
Este arquivo gera estado de proteção em modo simulação/analytics, sem executar aposta.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
ALERTS_DIR = BASE_DIR / "alerts"
ESTADO_FILE = BASE_DIR / "banca" / "banca_estado.json"
MARKET_PERF_FILE = REPORTS_DIR / "performance_por_mercado.json"
DRIFT_FILE = REPORTS_DIR / "resumo_deriva.json"
CALIBRATION_GUARD_FILE = REPORTS_DIR / "estado_guard_calibracao.json"

ROI_BLOCK = -0.03
MIN_BETS_BLOCK = 100
DRAWDOWN_CAUTION = 0.10
DRAWDOWN_PROTECTION = 0.20


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def is_stale(older: Path, newer: Path) -> bool:
    try:
        return older.exists() and newer.exists() and older.stat().st_mtime < newer.stat().st_mtime
    except Exception:
        return False


def drawdown_state(estado: dict[str, Any]) -> dict[str, Any]:
    atual = float(estado.get("banca_atual", 0) or 0)
    pico = float(estado.get("banca_pico", atual) or atual or 0)
    dd = ((pico - atual) / pico) if pico > 0 else 0.0
    if dd >= DRAWDOWN_PROTECTION:
        status = "PROTECTION"
    elif dd >= DRAWDOWN_CAUTION:
        status = "CAUTION"
    else:
        status = "OK"
    return {"banca_atual": round(atual, 2), "banca_pico": round(pico, 2), "drawdown": round(dd, 6), "status": status}


def mercado_blocks(perf: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = []
    for row in perf:
        apostas = int(row.get("apostas", row.get("registros", 0)) or 0)
        roi = float(row.get("roi", 0) or 0)
        if apostas >= MIN_BETS_BLOCK and roi < ROI_BLOCK:
            blocked.append({
                "mercado": row.get("mercado", ""),
                "apostas": apostas,
                "roi": round(roi, 6),
                "motivo": f"ROI abaixo de {ROI_BLOCK:.0%} com amostra >= {MIN_BETS_BLOCK}",
                "mode": "blocked_in_guard_simulacao",
            })
    return blocked


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    estado = read_json(ESTADO_FILE, {})
    perf = read_json(MARKET_PERF_FILE, [])
    drift = read_json(DRIFT_FILE, {})
    guard_calibracao = read_json(CALIBRATION_GUARD_FILE, {})
    simulation_file = REPORTS_DIR / "estado_guard_simulacao.json"
    stale = is_stale(simulation_file, CALIBRATION_GUARD_FILE)

    dd = drawdown_state(estado)
    blocks = mercado_blocks(perf if isinstance(perf, list) else [])
    drift_level = str(((drift.get("overall") or {}).get("alert_level") or "DESCONHECIDA")).upper() if isinstance(drift, dict) else "DESCONHECIDA"
    calibration_blocked = bool(guard_calibracao.get("blocked", False)) if isinstance(guard_calibracao, dict) else False

    risk_flags = []
    alertas = []
    removed_flags = []
    if stale:
        alertas.append("GUARD_SIMULACAO_DESATUALIZADO")
    if not calibration_blocked:
        removed_flags.append("CALIBRATION_GUARD_BLOQUEADA")
    if blocks:
        risk_flags.append("DEGRADACAO_MERCADO")
    if dd["status"] != "OK":
        risk_flags.append(f"DRAWDOWN_{dd['status']}")
    if drift_level == "CRITICO":
        risk_flags.append("DERIVA_CRITICA")
    if calibration_blocked:
        risk_flags.append("CALIBRATION_GUARD_BLOQUEADA")

    status = "PROTECTION" if any(f in risk_flags for f in ["DERIVA_CRITICA", "CALIBRATION_GUARD_BLOQUEADA", "DRAWDOWN_PROTECTION"]) or blocks else ("CAUTION" if risk_flags else "OK")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "guard_simulacao_only",
        "status": status,
        "risk_flags": risk_flags,
        "drawdown": dd,
        "mercados_bloqueados": blocks,
        "drift_level": drift_level,
        "guard_calibracao_blocked": calibration_blocked,
        "guard_calibracao_status": guard_calibracao.get("status") if isinstance(guard_calibracao, dict) else "DESCONHECIDA",
        "guard_calibracao_source": guard_calibracao.get("source") if isinstance(guard_calibracao, dict) else "DESCONHECIDA",
        "modo_simulacao": True,
        "apostas_reais_habilitadas": False,
        "recomendacoes_habilitadas": False,
        "modo_seguro": True,
        "alertas": alertas,
        "removed_flags": removed_flags,
        "staleness": {
            "guard_simulacao_stale": stale,
            "guard_simulacao_path": str(simulation_file),
            "guard_calibracao_path": str(CALIBRATION_GUARD_FILE),
        },
        "rules": {
            "mercado_block": {"roi_below": ROI_BLOCK, "min_records": MIN_BETS_BLOCK},
            "drawdown_caution": DRAWDOWN_CAUTION,
            "drawdown_protection": DRAWDOWN_PROTECTION,
        },
        "note": "Estado analítico para revisão humana. Não executa apostas nem recomenda entradas.",
    }
    (REPORTS_DIR / "estado_guard_simulacao.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ALERTS_DIR / "guard_simulacao_alerts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
