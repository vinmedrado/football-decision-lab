#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Football Lab — Controle operacional central.

Camada central de governança operacional. Não treina modelo, não altera
backtest e não modifica histórico; apenas lê artefatos existentes para decidir
se o predict pode gerar previsões em modo fail-safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR_FALLBACK = Path(__file__).resolve().parents[3]
ML_DIR_FALLBACK = Path(__file__).resolve().parents[2]
if str(ROOT_DIR_FALLBACK) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR_FALLBACK))
if str(ML_DIR_FALLBACK) not in sys.path:
    sys.path.insert(0, str(ML_DIR_FALLBACK))

from core.project_paths import (
    ROOT_DIR,
    ML_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    MODEL_REGISTRY_DIR,
    validate_environment,
    now_iso,
    read_json,
    write_json,
)
from core.block_motivo_catalog import get_motivo_info
from core.health_score_engine import calculate_health_score
from core.roi_consistency_engine import evaluate_roi_consistency
from controles.mercados.status import lifecycle_summary

REGISTRY_PATH = MODEL_REGISTRY_DIR / "model_registry.json"
MODEL_DIR = MODELS_DIR
BASELINE_METRICS_PATHS = [
    MODEL_DIR / "baseline_metrics.json",
    MODEL_REGISTRY_DIR / "baseline_metrics.json",
    REPORTS_DIR / "baseline_metrics.json",
]
OPERATIONAL_STATUS_PATH = REPORTS_DIR / "status_operacional.json"
GOVERNANCE_DIAGNOSTICS_PATH = REPORTS_DIR / "diagnostico_governanca.json"
OPERATIONAL_HEALTH_PATH = REPORTS_DIR / "saude_operacional.json"
GOVERNANCE_EXPLAINABILITY_PATH = REPORTS_DIR / "explicabilidade_governanca.json"
GOVERNANCE_EXECUTIVE_REPORT_PATH = REPORTS_DIR / "relatorio_executivo_governanca.json"
SAFE_MODE_FLAGS = {"modo_simulacao": True, "apostas_reais_habilitadas": False, "recomendacoes_habilitadas": False, "modo_seguro": True}

FAIL_SAFE_REASONS = {
    "CALIBRACAO_BLOQUEADA",
    "PROTECAO_SIMULACAO",
    "DERIVA_FEATURES_CRITICA",
    "REGISTRO_MODELO_INVALIDO",
    "MODELO_ATIVO_NAO_ENCONTRADO",
    "BASELINE_NAO_ENCONTRADA",
    "ENVIRONMENT_INVALID",
}


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_model_path(modelo_ativo: Optional[str]) -> Optional[Path]:
    if not modelo_ativo:
        return None
    candidate = Path(str(modelo_ativo))
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([MODEL_DIR / candidate, ML_DIR / candidate, ROOT_DIR / candidate])
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _load_mercado_governance() -> Dict[str, Any]:
    try:
        from market_governance import evaluate_mercado_governance
        return evaluate_mercado_governance()
    except Exception as exc:
        return {
            "status": "ALERTA",
            "mercados_bloqueados": [],
            "mercados_permitidos": [],
            "alertas": [f"MARKET_GOVERNANCE_ERRO:{exc}"],
        }


def _extract_registry_models(registry: Dict[str, Any]) -> tuple[Any, Any, Any]:
    modelo_ativo = registry.get("modelo_ativo") or registry.get("active")
    modelo_campeao = registry.get("modelo_campeao") or registry.get("champion")
    modelo_base = registry.get("modelo_base") or registry.get("baseline")
    if isinstance(modelo_base, dict):
        modelo_base = modelo_base.get("model") or modelo_base.get("path") or modelo_base.get("name")
    return modelo_ativo, modelo_campeao, modelo_base


def _write_governance_diagnostics(payload: Dict[str, Any]) -> Dict[str, Any]:
    diag = {
        "timestamp": now_iso(),
        "normalizer_status": "OK",
        "performance_provider": "OK" if payload.get("performance_source") not in {None, "empty"} else "ALERTA",
        "environment_validation": payload.get("environment_status", "DESCONHECIDA"),
        "path_validation": "OK" if not payload.get("stale_path_references") else "ALERTA",
        "mercado_governance": payload.get("mercado_governance_status", "DESCONHECIDA"),
        "alertas": payload.get("alertas", []),
    }
    write_json(GOVERNANCE_DIAGNOSTICS_PATH, diag)
    return diag


def _write_governance_explainability(payload: Dict[str, Any]) -> Dict[str, Any]:
    motivo = payload.get("motivo") or ("OK" if payload.get("status") == "OK" else "DESCONHECIDA")
    info = get_motivo_info(str(motivo)) if motivo != "OK" else {
        "motivo": "OK",
        "severity": "OK",
        "description": "Nenhum bloqueio operacional ativo.",
        "acao_recomendada": "Manter monitoramento regular dos relatórios de governança.",
    }
    mercado_motivos = payload.get("mercado_block_motivos", {}) or {}
    mercado_explanations = {
        mercado: [get_motivo_info(r) for r in motivos]
        for mercado, motivos in mercado_motivos.items()
    }
    explainability = {
        "timestamp": now_iso(),
        "status": payload.get("status", "DESCONHECIDA"),
        "motivo": motivo,
        "severity": info.get("severity"),
        "explanation": info.get("description"),
        "acao_recomendada": info.get("acao_recomendada"),
        "mercado_explanations": mercado_explanations,
        "alertas": payload.get("alertas", []),
    }
    write_json(GOVERNANCE_EXPLAINABILITY_PATH, explainability)
    return explainability


def _write_governance_executive_report(payload: Dict[str, Any], health: Dict[str, Any], roi_summary: Dict[str, Any]) -> Dict[str, Any]:
    env = read_json(REPORTS_DIR / "status_ambiente.json", {}) or {}
    executive = {
        "timestamp": now_iso(),
        "operational_health": health.get("score"),
        "health_status": health.get("label"),
        "environment_status": env.get("status", payload.get("environment_status")),
        "roi_consistency": roi_summary.get("status", "DESCONHECIDA"),
        "operational_status": payload.get("status"),
        "blocked_motivo": payload.get("motivo"),
        "permitir_previsoes": payload.get("permitir_previsoes", True),
        "performance_source": payload.get("performance_source"),
        "mercado_governance_status": payload.get("mercado_governance_status"),
        "exposure_status": payload.get("exposure_status"),
        "mercados_bloqueados": payload.get("mercados_bloqueados", []),
        "alertas": payload.get("alertas", []),
    }
    write_json(GOVERNANCE_EXECUTIVE_REPORT_PATH, executive)
    return executive


def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.update(SAFE_MODE_FLAGS)
    write_json(OPERATIONAL_STATUS_PATH, payload)
    _write_governance_diagnostics(payload)
    roi_result = evaluate_roi_consistency()
    health = calculate_health_score(payload)
    _write_governance_explainability(payload)
    _write_governance_executive_report(payload, health, roi_result.get("summary", {}))
    return payload


def _block(motivo: str, *, alertas: Optional[List[str]] = None, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(partial or {})
    payload.update({
        "timestamp": now_iso(),
        "status": "BLOQUEADA",
        "permitir_previsoes": False,
        "motivo": motivo,
        "alertas": sorted(set(list(alertas or []))),
    })
    payload.setdefault("mercados_bloqueados", [])
    payload.setdefault("mercados_permitidos", [])
    return _finalize(payload)


def evaluate_operational_guard() -> Dict[str, Any]:
    """Avalia todos os guards obrigatórios e retorna decisão fail-safe."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = validate_environment()
    alertas: List[str] = list(env.get("alertas", []))

    calibration = read_json(REPORTS_DIR / "estado_guard_calibracao.json", {}) or {}
    simulation = read_json(REPORTS_DIR / "estado_guard_simulacao.json", {}) or {}
    saude_features = read_json(REPORTS_DIR / "relatorio_saude_features.json", {}) or {}
    registry = read_json(REGISTRY_PATH, {}) or {}
    mercado_gov = _load_mercado_governance()

    modelos_registrados = registry.get("models", [])
    modelos_aprovados = [
        m for m in modelos_registrados
        if isinstance(m, dict)
        and str(m.get("status", "")).lower() in {"approved", "aprovado"}
        and m.get("deployable") is True
        and m.get("model_file")
    ]

    modelo_ativo = modelos_aprovados[0]["model_file"] if modelos_aprovados else None
    modelo_campeao = modelo_ativo
    modelo_base = str(MODEL_DIR / "baseline_metrics.json")
    feature_status = saude_features.get("overall_status", saude_features.get("status", "DESCONHECIDA"))
    resumo_ciclo_vida_mercados = lifecycle_summary()
    calibration_status = "BLOQUEADA" if calibration.get("blocked") is True else calibration.get("status", "DESCONHECIDA")
    simulation_status = simulation.get("status", "DESCONHECIDA")

    base_status = {
        "timestamp": now_iso(),
        "modelo_ativo": modelo_ativo,
        "modelo_campeao": modelo_campeao,
        "modelo_base": modelo_base,
        "guard_calibracao": calibration_status,
        "guard_simulacao": simulation_status,
        "saude_features": feature_status,
        "model_registry": "OK" if modelos_aprovados else "INVALID",
        "modelos_registrados": len(modelos_registrados),
        "modelos_aprovados": len(modelos_aprovados),
        "environment_status": env.get("status", "DESCONHECIDA"),
        "stale_path_references": env.get("stale_path_references", []),
        "mercado_governance_status": mercado_gov.get("status", "DESCONHECIDA"),
        "exposure_status": mercado_gov.get("exposure_status", "DESCONHECIDA"),
        "performance_source": mercado_gov.get("performance_source", "DESCONHECIDA"),
        "mercados_bloqueados": mercado_gov.get("mercados_bloqueados", []),
        "mercados_permitidos": mercado_gov.get("mercados_permitidos", []),
        "mercado_block_motivos": mercado_gov.get("mercado_motivos", {}),
        "resumo_ciclo_vida_mercados": resumo_ciclo_vida_mercados,
    }
    alertas.extend(mercado_gov.get("alertas", []))

    if env.get("status") == "BLOQUEADA":
        return _block("ENVIRONMENT_INVALID", alertas=alertas, partial=base_status)
    if calibration.get("blocked") is True:
        return _block("CALIBRACAO_BLOQUEADA", alertas=alertas, partial=base_status)
    if str(simulation_status).upper() == "PROTECTION":
        return _block("PROTECAO_SIMULACAO", alertas=alertas, partial=base_status)
    if str(feature_status).upper() == "CRITICO":
        return _block("DERIVA_FEATURES_CRITICA", alertas=alertas, partial=base_status)
    if not modelos_aprovados:
        return _block("REGISTRO_MODELO_INVALIDO", alertas=alertas, partial=base_status)
    if resumo_ciclo_vida_mercados and resumo_ciclo_vida_mercados.get("active", 0) <= 0:
        alertas.append("SEM_MERCADOS_ATIVOS")
        return _block("SEM_MERCADOS_ATIVOS", alertas=alertas, partial=base_status)
    modelo_ativo_path = _resolve_model_path(str(modelo_ativo))
    if modelo_ativo_path is None:
        return _block("MODELO_ATIVO_NAO_ENCONTRADO", alertas=alertas, partial=base_status)
    baseline_path = _first_existing(BASELINE_METRICS_PATHS)
    if baseline_path is None:
        return _block("BASELINE_NAO_ENCONTRADA", alertas=alertas, partial=base_status)

    payload = dict(base_status)
    payload.update({
        "timestamp": now_iso(),
        "status": "SIMULATION_ONLY",
        "permitir_previsoes": True,
        "motivo": "SIMULATION_MODE_ATIVA",
        "alertas": sorted(set(alertas)),
        "modelo_ativo_path": str(modelo_ativo_path),
        "baseline_metrics_path": str(baseline_path),
    })
    return _finalize(payload)


if __name__ == "__main__":
    print(json.dumps(evaluate_operational_guard(), ensure_ascii=False, indent=2))
