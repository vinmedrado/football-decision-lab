#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Governança Operacional Engine.

Transforms the auditoria de desempenho real reality audit into a mercado lifecycle state for
simulation/audit governance. It does not train models, alter strategy, settle
bets, modify bankroll, relax guards, or unlock any real betting flow.

Safe mode is explicit:
- modo_simulacao = true
- apostas_reais_habilitadas = false
- recomendacoes_habilitadas = false
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import REPORTS_DIR, ROOT_DIR, now_iso, read_json, write_json  # noqa: E402

MARKET_REALITY_AUDIT_PATH = REPORTS_DIR / "auditoria_operacional_mercados.json"
LACUNA_OPERACIONAL_SUMMARY_PATH = REPORTS_DIR / "resumo_lacuna_operacional.json"
MARKET_GOVERNANCE_PATH = REPORTS_DIR / "relatorio_governanca_mercados.json"
PERFORMANCE_BY_MARKET_PATH = REPORTS_DIR / "performance_por_mercado.json"

MARKET_LIFECYCLE_STATE_PATH = REPORTS_DIR / "ciclo_vida_mercados.json"
REALITY_GOVERNANCE_REPORT_PATH = REPORTS_DIR / "relatorio_governanca_operacional.json"
MARKET_LIFECYCLE_AUDIT_TRAIL_PATH = REPORTS_DIR / "historico_ciclo_vida_mercados.json"
OPERATIONAL_STATUS_PATH = REPORTS_DIR / "status_operacional.json"

MIN_REAL_SAMPLE = 10
ALLOWED_STATUSES = {"ATIVA", "OBSERVACAO", "APOSENTADA", "BLOQUEADA", "DESCONHECIDA"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _load_mercado_reality_rows() -> List[Dict[str, Any]]:
    data = read_json(MARKET_REALITY_AUDIT_PATH, {})
    rows = data.get("mercados", []) if isinstance(data, dict) else []
    return [x for x in rows if isinstance(x, dict)]


def _blocked_by_mercado_governance() -> Dict[str, List[str]]:
    data = read_json(MARKET_GOVERNANCE_PATH, {})
    blocked: Dict[str, List[str]] = {}
    if not isinstance(data, dict):
        return blocked

    for key in ("mercados_bloqueados", "mercado_block_motivos"):
        value = data.get(key)
        if isinstance(value, dict):
            for mercado, motivos in value.items():
                if isinstance(motivos, list):
                    blocked[str(mercado)] = [str(r) for r in motivos]
                else:
                    blocked[str(mercado)] = [str(motivos)]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    mercado = item.get("mercado") or item.get("mercado")
                    if mercado:
                        motivo = item.get("motivo") or item.get("motivo") or "MARKET_GOVERNANCE_BLOCK"
                        blocked[str(mercado)] = [str(motivo)]
                elif item:
                    blocked[str(item)] = ["MARKET_GOVERNANCE_BLOCK"]
    return blocked


def _classify_mercado(row: Dict[str, Any], blocked_map: Dict[str, List[str]]) -> Tuple[str, str, str]:
    mercado = str(row.get("mercado") or "").strip()
    prod = row.get("production", {}) if isinstance(row.get("production"), dict) else {}
    status_operacional_real = str(row.get("status_operacional_real") or "").upper()
    action = str(row.get("acao_recomendada") or "").upper()
    roi_real = _as_float(prod.get("roi_real"))
    apostas_reais = _as_int(prod.get("total_apostas_real"))

    # Reality evidence has precedence over generic governance labels.
    # A proven negative reality gap retires the mercado; insufficient real sample
    # stays on watchlist instead of being treated as active.
    if status_operacional_real == "SUSPEITA_DE_SOBREAJUSTE" or action == "APOSENTAR_MERCADO" or (roi_real < 0 and apostas_reais >= MIN_REAL_SAMPLE):
        return "APOSENTADA", status_operacional_real or "NEGATIVE_REAL_ROI", "APOSENTAR_MERCADO"

    if roi_real > 0 and apostas_reais < MIN_REAL_SAMPLE:
        return "OBSERVACAO", "AMOSTRA_REAL_INSUFICIENTE", "OBSERVACAO"

    if mercado in blocked_map:
        motivos = set(blocked_map.get(mercado, []))
        if motivos - {"INSUFFICIENT_SAMPLE"}:
            return "BLOQUEADA", "MARKET_GOVERNANCE_BLOCK", "BLOQUEADA_BY_GOVERNANCE"

    if (
        roi_real > 0
        and apostas_reais >= MIN_REAL_SAMPLE
        and (
            status_operacional_real == "APROVADA_NA_OPERACAO"
            or mercado not in blocked_map
        )
    ):
        return "ATIVA", "APROVADA_NA_OPERACAO", "MANTER_ATIVA"

    if apostas_reais <= 0:
        return "DESCONHECIDA", "NO_REAL_SAMPLE", "OBSERVACAO"

    return "DESCONHECIDA", status_operacional_real or "REALITY_UNCLEAR", "OBSERVACAO"


def _previous_statuses() -> Dict[str, str]:
    state = read_json(MARKET_LIFECYCLE_STATE_PATH, {})
    mercados = state.get("mercados", {}) if isinstance(state, dict) else {}
    if not isinstance(mercados, dict):
        return {}
    return {str(m): str((info or {}).get("status_ciclo_vida") or "DESCONHECIDA") for m, info in mercados.items()}


def build_lifecycle_state() -> Dict[str, Any]:
    rows = _load_mercado_reality_rows()
    blocked_map = _blocked_by_mercado_governance()
    previous = _previous_statuses()
    mercados: Dict[str, Dict[str, Any]] = {}
    changes: List[Dict[str, Any]] = []
    now = now_iso()

    for row in rows:
        mercado = str(row.get("mercado") or "").strip()
        if not mercado:
            continue
        backtest = row.get("backtest", {}) if isinstance(row.get("backtest"), dict) else {}
        prod = row.get("production", {}) if isinstance(row.get("production"), dict) else {}
        comp = row.get("comparison", {}) if isinstance(row.get("comparison"), dict) else {}

        status, motivo, action = _classify_mercado(row, blocked_map)
        if status not in ALLOWED_STATUSES:
            status = "DESCONHECIDA"

        roi_backtest = _as_float(backtest.get("roi_backtest"))
        roi_real = _as_float(prod.get("roi_real"))
        apostas_reais = _as_int(prod.get("total_apostas_real"))
        roi_gap_signed = roi_real - roi_backtest

        mercados[mercado] = {
            "status_ciclo_vida": status,
            "motivo": motivo,
            "acao_recomendada": action,
            "status_operacional_real": row.get("status_operacional_real"),
            "roi_backtest": roi_backtest,
            "roi_real": roi_real,
            "roi_gap_signed": roi_gap_signed,
            "roi_gap_abs": abs(roi_gap_signed),
            "apostas_reais": apostas_reais,
            "taxa_acerto_real": _as_float(prod.get("taxa_acerto_real")),
            "odd_media_real": _as_float(prod.get("odd_media_real")),
            "simulation_only": True,
            "apostas_reais_habilitadas": False,
            "recomendacoes_habilitadas": False,
        }

        old = previous.get(mercado, "DESCONHECIDA")
        if old != status:
            changes.append({
                "timestamp": now,
                "mercado": mercado,
                "from": old,
                "to": status,
                "motivo": motivo,
                "roi_backtest": roi_backtest,
                "roi_real": roi_real,
                "apostas_reais": apostas_reais,
            })

    summary = {
        "active": sum(1 for x in mercados.values() if x["status_ciclo_vida"] == "ATIVA"),
        "watchlist": sum(1 for x in mercados.values() if x["status_ciclo_vida"] == "OBSERVACAO"),
        "retired": sum(1 for x in mercados.values() if x["status_ciclo_vida"] == "APOSENTADA"),
        "blocked": sum(1 for x in mercados.values() if x["status_ciclo_vida"] == "BLOQUEADA"),
        "unknown": sum(1 for x in mercados.values() if x["status_ciclo_vida"] == "DESCONHECIDA"),
    }

    state = {
        "generated_at": now,
        "source": "mercado_reality_audit",
        "scope": "simulation_lifecycle_governance",
        "modo_simulacao": True,
        "apostas_reais_habilitadas": False,
        "recomendacoes_habilitadas": False,
        "min_real_sample": MIN_REAL_SAMPLE,
        "summary": summary,
        "mercados": mercados,
    }

    report = {
        "timestamp": now,
        "status": "OK" if mercados else "NO_DATA",
        "scope": "simulation_lifecycle_governance",
        "modo_seguro": True,
        "modo_simulacao": True,
        "apostas_reais_habilitadas": False,
        "recomendacoes_habilitadas": False,
        "source_files": {
            "mercado_reality_audit": str(MARKET_REALITY_AUDIT_PATH.relative_to(ROOT_DIR)),
            "reality_gap_summary": str(LACUNA_OPERACIONAL_SUMMARY_PATH.relative_to(ROOT_DIR)),
            "mercado_governance": str(MARKET_GOVERNANCE_PATH.relative_to(ROOT_DIR)),
            "performance_by_mercado": str(PERFORMANCE_BY_MARKET_PATH.relative_to(ROOT_DIR)),
        },
        "summary": summary,
        "retired_mercados": [m for m, x in mercados.items() if x["status_ciclo_vida"] == "APOSENTADA"],
        "watchlist_mercados": [m for m, x in mercados.items() if x["status_ciclo_vida"] == "OBSERVACAO"],
        "active_mercados": [m for m, x in mercados.items() if x["status_ciclo_vida"] == "ATIVA"],
        "mercados_bloqueados": [m for m, x in mercados.items() if x["status_ciclo_vida"] == "BLOQUEADA"],
        "unknown_mercados": [m for m, x in mercados.items() if x["status_ciclo_vida"] == "DESCONHECIDA"],
        "changes": changes,
    }

    previous_trail = read_json(MARKET_LIFECYCLE_AUDIT_TRAIL_PATH, [])
    if not isinstance(previous_trail, list):
        previous_trail = []
    audit_trail = previous_trail + changes

    write_json(MARKET_LIFECYCLE_STATE_PATH, state)
    write_json(REALITY_GOVERNANCE_REPORT_PATH, report)
    write_json(MARKET_LIFECYCLE_AUDIT_TRAIL_PATH, audit_trail)

    _refresh_operational_status(summary, report)
    return report


def _refresh_operational_status(summary: Dict[str, int], report: Dict[str, Any]) -> None:
    status = read_json(OPERATIONAL_STATUS_PATH, {})
    if not isinstance(status, dict):
        status = {}
    status["timestamp"] = now_iso()
    status["resumo_ciclo_vida_mercados"] = summary
    status["mercado_lifecycle_source"] = "reality_governance_report"
    status["modo_simulacao"] = True
    status["apostas_reais_habilitadas"] = False
    status["recomendacoes_habilitadas"] = False
    status["modo_seguro"] = True
    if summary.get("active", 0) <= 0:
        status["status"] = "BLOQUEADA"
        status["permitir_previsoes"] = False
        status["motivo"] = "SEM_MERCADOS_ATIVOS"
        alertas = status.get("alertas", [])
        if not isinstance(alertas, list):
            alertas = []
        if "SEM_MERCADOS_ATIVOS" not in alertas:
            alertas.append("SEM_MERCADOS_ATIVOS")
        status["alertas"] = alertas
    write_json(OPERATIONAL_STATUS_PATH, status)


def main() -> None:
    report = build_lifecycle_state()
    print("\n" + "=" * 72)
    print("Governança Operacional Engine")
    print("=" * 72)
    print("Modo: SIMULAÇÃO/AUDITORIA")
    print("Apostas reais habilitadas: false")
    print("Recomendações habilitadas: false")
    print(f"Status: {report.get('status')}")
    print("\nMercados aposentados:")
    for mercado in report.get("retired_mercados", []):
        print(f"  - {mercado}")
    print("\nMercados em watchlist:")
    for mercado in report.get("watchlist_mercados", []):
        print(f"  - {mercado}")
    print("\nMercados ativos:")
    active = report.get("active_mercados", [])
    if active:
        for mercado in active:
            print(f"  - {mercado}")
    else:
        print("  - nenhum")
    print("\nArquivos gerados:")
    print(f"  - {MARKET_LIFECYCLE_STATE_PATH.relative_to(ROOT_DIR)}")
    print(f"  - {REALITY_GOVERNANCE_REPORT_PATH.relative_to(ROOT_DIR)}")
    print(f"  - {MARKET_LIFECYCLE_AUDIT_TRAIL_PATH.relative_to(ROOT_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
