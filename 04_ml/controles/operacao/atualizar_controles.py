#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sincronização segura dos controles operacionais.

Não libera apostas reais, não altera stake/banca/modelo e não gera recomendações.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT_DIR / "04_ml"
REPORTS_DIR = BASE_DIR / "reports"

CALIBRATION_GUARD = REPORTS_DIR / "estado_guard_calibracao.json"
SIMULATION_GUARD = REPORTS_DIR / "estado_guard_simulacao.json"
DRIFT_SUMMARY = REPORTS_DIR / "resumo_deriva.json"
MARKET_GOVERNANCE = REPORTS_DIR / "relatorio_governanca_mercados.json"
EXPOSURE_GUARD = REPORTS_DIR / "guard_exposicao.json"
OPERATIONAL_STATUS = REPORTS_DIR / "status_operacional.json"
REFRESH_REPORT = REPORTS_DIR / "relatorio_atualizacao_guards.json"

SAFE_FLAGS = {
    "modo_simulacao": True,
    "apostas_reais_habilitadas": False,
    "recomendacoes_habilitadas": False,
    "modo_seguro": True,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_is_stale(older: Path, newer: Path) -> bool:
    try:
        return older.exists() and newer.exists() and older.stat().st_mtime < newer.stat().st_mtime
    except Exception:
        return False


def run_python(relative_path: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, relative_path],
        cwd=ROOT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def refresh_guard_simulacao() -> tuple[dict, list[str], list[str]]:
    before = read_json(SIMULATION_GUARD, {}) or {}
    before_flags = set(before.get("risk_flags", []) if isinstance(before, dict) else [])
    alertas: list[str] = []
    if file_is_stale(SIMULATION_GUARD, CALIBRATION_GUARD):
        alertas.append("GUARD_SIMULACAO_DESATUALIZADO")
    code, out, err = run_python("04_ml/13_simulation_guard.py")
    after = read_json(SIMULATION_GUARD, {}) or {}
    after_flags = set(after.get("risk_flags", []) if isinstance(after, dict) else [])
    removed = sorted(before_flags - after_flags)
    remaining = sorted(after_flags)
    if code != 0:
        alertas.append(f"SIMULATION_GUARD_REFRESH_ERRO:{err.strip() or out.strip()}")
    if alertas:
        after.setdefault("alertas", [])
        after["alertas"] = sorted(set(after.get("alertas", []) + alertas))
        after.update(SAFE_FLAGS)
        write_json(SIMULATION_GUARD, after)
    return after, removed, remaining


def refresh_operational_status() -> dict:
    calibration = read_json(CALIBRATION_GUARD, {}) or {}
    simulation = read_json(SIMULATION_GUARD, {}) or {}
    drift = read_json(DRIFT_SUMMARY, {}) or {}
    mercado = read_json(MARKET_GOVERNANCE, {}) or {}
    exposure = read_json(EXPOSURE_GUARD, {}) or {}

    motivos: list[str] = []
    alertas: list[str] = []

    if bool(calibration.get("blocked", False)):
        motivos.append("CALIBRACAO_BLOQUEADA")
    if str(simulation.get("status", "")).upper() == "PROTECTION":
        motivos.append("PROTECAO_SIMULACAO")
    for flag in simulation.get("risk_flags", []) if isinstance(simulation, dict) else []:
        if flag and flag not in motivos:
            motivos.append(str(flag))
    if isinstance(drift, dict):
        level = str(((drift.get("overall") or {}).get("alert_level") or "")).upper()
        # Se o calibration recovery já resolveu a calibração, não transformar alerta antigo em bloqueio por calibração.
        if level == "CRITICO" and bool(calibration.get("blocked", True)):
            if "DERIVA_CRITICA" not in motivos:
                motivos.append("DERIVA_CRITICA")
    if isinstance(mercado, dict):
        if str(mercado.get("status", "")).upper() in {"BLOQUEADA", "CRITICO", "PROTECTION"}:
            motivos.append("MARKET_GOVERNANCE_BLOQUEADA")
        alertas.extend(mercado.get("alertas", []) or [])
    if isinstance(exposure, dict):
        if str(exposure.get("status", "")).upper() in {"BLOQUEADA", "CRITICO", "PROTECTION"}:
            motivos.append("EXPOSURE_BLOQUEADA")
        alertas.extend(exposure.get("alertas", []) or [])
    alertas.extend(simulation.get("alertas", []) if isinstance(simulation, dict) else [])

    # Hotfix seguro: status diagnóstico sincronizado, mas sem habilitar recomendações.
    primary_motivo = motivos[0] if motivos else "SIMULATION_MODE_ATIVA"
    status = "BLOQUEADA" if motivos else "SIMULATION_ONLY"
    payload = {
        "timestamp": now_iso(),
        "status": status,
        "permitir_previsoes": False,
        "motivo": primary_motivo,
        "guard_calibracao": calibration.get("status", "DESCONHECIDA"),
        "calibration_blocked": bool(calibration.get("blocked", False)),
        "raw_calibration_error": calibration.get("raw_calibration_error"),
        "calibrated_calibration_error": calibration.get("calibrated_calibration_error"),
        "calibration_source": calibration.get("source"),
        "guard_simulacao": simulation.get("status", "DESCONHECIDA"),
        "simulation_risk_flags": simulation.get("risk_flags", []) if isinstance(simulation, dict) else [],
        "mercado_governance_status": mercado.get("status", "DESCONHECIDA") if isinstance(mercado, dict) else "DESCONHECIDA",
        "exposure_status": exposure.get("status", "DESCONHECIDA") if isinstance(exposure, dict) else "DESCONHECIDA",
        "mercados_bloqueados": mercado.get("mercados_bloqueados", []) if isinstance(mercado, dict) else [],
        "mercados_permitidos": mercado.get("mercados_permitidos", []) if isinstance(mercado, dict) else [],
        "alertas": sorted(set([str(w) for w in alertas if w])),
        **SAFE_FLAGS,
        "note": "Hotfix seguro: artefatos sincronizados em modo simulação. Não habilita apostas reais nem recomendações operacionais.",
    }
    write_json(OPERATIONAL_STATUS, payload)
    return payload


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    calibration = read_json(CALIBRATION_GUARD, {}) or {}
    simulation, removed, remaining = refresh_guard_simulacao()
    operational = refresh_operational_status()

    report = {
        "timestamp": now_iso(),
        "status": "OK",
        "scope": "safe_guard_artifact_refresh",
        "guard_calibracao": {
            "status": calibration.get("status"),
            "blocked": calibration.get("blocked"),
            "source": calibration.get("source"),
            "raw_calibration_error": calibration.get("raw_calibration_error"),
            "calibrated_calibration_error": calibration.get("calibrated_calibration_error"),
        },
        "guard_simulacao": {
            "status": simulation.get("status"),
            "risk_flags": simulation.get("risk_flags", []),
            "removed_flags": removed,
            "remaining_flags": remaining,
            "alertas": simulation.get("alertas", []),
        },
        "operational_status": {
            "status": operational.get("status"),
            "motivo": operational.get("motivo"),
            "permitir_previsoes": operational.get("permitir_previsoes"),
            "modo_simulacao": operational.get("modo_simulacao"),
            "apostas_reais_habilitadas": operational.get("apostas_reais_habilitadas"),
            "recomendacoes_habilitadas": operational.get("recomendacoes_habilitadas"),
            "modo_seguro": operational.get("modo_seguro"),
        },
    }
    write_json(REFRESH_REPORT, report)

    print("✅ Guards e relatórios sincronizados em modo simulação.")
    print(f"Guard de Calibração: {report['guard_calibracao']['status']} | blocked={str(report['guard_calibracao']['blocked']).lower()}")
    print(f"Guard de Simulação: {report['guard_simulacao']['status']}")
    print(f"Status Operacional: {report['operational_status']['status']} | motivo={report['operational_status']['motivo']}")
    print(f"Flags removidas: {', '.join(removed) if removed else 'nenhuma'}")
    print(f"Flags restantes: {', '.join(remaining) if remaining else 'nenhuma'}")
    print("modo_simulacao=true | apostas_reais_habilitadas=false | recomendacoes_habilitadas=false")
    print(f"Relatório: {REFRESH_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
