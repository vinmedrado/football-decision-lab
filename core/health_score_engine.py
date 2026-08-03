#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operational Pontuação de Saúde auditavel para governanca."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR_FALLBACK = Path(__file__).resolve().parents[1]
if str(ROOT_DIR_FALLBACK) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR_FALLBACK))

from core.project_paths import REPORTS_DIR, now_iso, read_json, write_json

BREAKDOWN_PATH = REPORTS_DIR / "detalhamento_pontuacao_saude.json"
HEALTH_PATH = REPORTS_DIR / "saude_operacional.json"

DEDUCTIONS = {
    "CALIBRACAO_BLOQUEADA": -20,
    "PROTECAO_SIMULACAO": -20,
    "DERIVA_FEATURES_CRITICA": -15,
    "EXPOSURE_BLOQUEADA": -15,
    "MARKET_GOVERNANCE_BLOQUEADA": -15,
    "REGISTRO_MODELO_INVALIDO": -15,
    "MODELO_ATIVO_NAO_ENCONTRADO": -15,
    "BASELINE_NAO_ENCONTRADA": -15,
    "ENVIRONMENT_INVALID": -15,
}


def classify_health(score: int) -> str:
    if score >= 95:
        return "EXCELLENT"
    if score >= 80:
        return "HEALTHY"
    if score >= 60:
        return "ALERTA"
    if score >= 40:
        return "DEGRADED"
    return "CRITICO"


def _status_is_blocked(value: Any) -> bool:
    return str(value or "").upper() in {"BLOQUEADA", "CRITICO", "INVALID", "PROTECTION"}


def infer_deductions(status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    deductions: List[Dict[str, Any]] = []
    motivo = str(status_payload.get("motivo") or "").upper()
    if motivo in DEDUCTIONS:
        deductions.append({"motivo": motivo, "points": DEDUCTIONS[motivo], "source": "operational_status.motivo"})

    if str(status_payload.get("guard_calibracao") or "").upper() == "BLOQUEADA" and not any(d["motivo"] == "CALIBRACAO_BLOQUEADA" for d in deductions):
        deductions.append({"motivo": "CALIBRACAO_BLOQUEADA", "points": DEDUCTIONS["CALIBRACAO_BLOQUEADA"], "source": "guard_calibracao"})
    if str(status_payload.get("guard_simulacao") or "").upper() == "PROTECTION" and not any(d["motivo"] == "PROTECAO_SIMULACAO" for d in deductions):
        deductions.append({"motivo": "PROTECAO_SIMULACAO", "points": DEDUCTIONS["PROTECAO_SIMULACAO"], "source": "guard_simulacao"})
    if str(status_payload.get("saude_features") or "").upper() == "CRITICO" and not any(d["motivo"] == "DERIVA_FEATURES_CRITICA" for d in deductions):
        deductions.append({"motivo": "DERIVA_FEATURES_CRITICA", "points": DEDUCTIONS["DERIVA_FEATURES_CRITICA"], "source": "saude_features"})
    if _status_is_blocked(status_payload.get("exposure_status")):
        deductions.append({"motivo": "EXPOSURE_BLOQUEADA", "points": DEDUCTIONS["EXPOSURE_BLOQUEADA"], "source": "exposure_status"})
    if _status_is_blocked(status_payload.get("mercado_governance_status")):
        deductions.append({"motivo": "MARKET_GOVERNANCE_BLOQUEADA", "points": DEDUCTIONS["MARKET_GOVERNANCE_BLOQUEADA"], "source": "mercado_governance_status"})
    if str(status_payload.get("model_registry") or "").upper() in {"INVALID", "BLOQUEADA"} and not any(d["motivo"] == "REGISTRO_MODELO_INVALIDO" for d in deductions):
        deductions.append({"motivo": "REGISTRO_MODELO_INVALIDO", "points": DEDUCTIONS["REGISTRO_MODELO_INVALIDO"], "source": "model_registry"})
    if str(status_payload.get("environment_status") or "").upper() == "BLOQUEADA" and not any(d["motivo"] == "ENVIRONMENT_INVALID" for d in deductions):
        deductions.append({"motivo": "ENVIRONMENT_INVALID", "points": DEDUCTIONS["ENVIRONMENT_INVALID"], "source": "environment_status"})

    # Deduplicar por motivo, mantendo a primeira fonte.
    seen = set()
    out = []
    for item in deductions:
        if item["motivo"] not in seen:
            out.append(item)
            seen.add(item["motivo"])
    return out


def calculate_health_score(status_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if status_payload is None:
        status_payload = read_json(REPORTS_DIR / "status_operacional.json", {}) or {}
    base_score = 100
    deductions = infer_deductions(status_payload)
    final_score = max(0, min(100, base_score + sum(int(d.get("points", 0)) for d in deductions)))
    label = classify_health(final_score)
    breakdown = {
        "timestamp": now_iso(),
        "base_score": base_score,
        "deductions": deductions,
        "final_score": final_score,
        "label": label,
        "classification_rules": {
            "95-100": "EXCELLENT",
            "80-94": "HEALTHY",
            "60-79": "ALERTA",
            "40-59": "DEGRADED",
            "0-39": "CRITICO",
        },
        "components": {
            "calibration": status_payload.get("guard_calibracao"),
            "simulation": status_payload.get("guard_simulacao"),
            "drift": status_payload.get("saude_features"),
            "exposure": status_payload.get("exposure_status"),
            "mercado_governance": status_payload.get("mercado_governance_status"),
            "model_registry": status_payload.get("model_registry"),
            "environment": status_payload.get("environment_status"),
        },
    }
    health = {
        "timestamp": now_iso(),
        "score": final_score,
        "label": label,
        "status": status_payload.get("status", "DESCONHECIDA"),
        "permitir_previsoes": bool(status_payload.get("permitir_previsoes", False)),
        "motivo": status_payload.get("motivo"),
        "breakdown_file": str(BREAKDOWN_PATH),
        "deductions_total": sum(int(d.get("points", 0)) for d in deductions),
        "deductions": deductions,
    }
    write_json(BREAKDOWN_PATH, breakdown)
    write_json(HEALTH_PATH, health)
    return health


if __name__ == "__main__":
    print(json.dumps(calculate_health_score(), ensure_ascii=False, indent=2))
