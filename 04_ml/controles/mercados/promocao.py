#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 18 — Motor de Promoção de Mercados (simulation/safety mode).

Evaluates OBSERVACAO mercados as promotion candidates using real-production evidence,
calibration, reality audit, drawdown and stability. This implementation is deliberately
safe-by-default: it DOES NOT enable real betting, does NOT set recomendacoes_habilitadas,
and does NOT change any mercado lifecycle to ATIVA automatically.

Outputs are analytical artifacts only:
- reports/relatorio_promocao_mercados.json
- reports/classificacao_mercados.json
- reports/candidatos_substituicao.json
- lifecycle state annotated with promotion_simulation only
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import REPORTS_DIR, ROOT_DIR, read_json, write_json, now_iso  # noqa: E402
from core.result_normalizer import normalize_result  # noqa: E402

LIFECYCLE_PATH = REPORTS_DIR / "ciclo_vida_mercados.json"
REALITY_AUDIT_PATH = REPORTS_DIR / "auditoria_operacional_mercados.json"
CALIBRATION_RECOVERY_PATH = REPORTS_DIR / "relatorio_recuperacao_calibracao.json"
MARKET_GOVERNANCE_PATH = REPORTS_DIR / "relatorio_governanca_mercados.json"
HISTORY_PATH = ROOT_DIR / "04_ml" / "banca" / "historico_apostas.csv"

PROMOTION_REPORT_PATH = REPORTS_DIR / "relatorio_promocao_mercados.json"
MARKET_RANKING_PATH = REPORTS_DIR / "classificacao_mercados.json"
REPLACEMENT_CANDIDATES_PATH = REPORTS_DIR / "candidatos_substituicao.json"
MARKET_LIFECYCLE_AUDIT_TRAIL_PATH = REPORTS_DIR / "historico_ciclo_vida_mercados.json"

MIN_BETS_FOR_ATIVA = 100
MIN_PROMOTION_SCORE = 80
MAX_ACCEPTABLE_DRAWDOWN = 0.25
MIN_WINRATE = 0.50

SAFE_FLAGS = {
    "modo_simulacao": True,
    "apostas_reais_habilitadas": False,
    "recomendacoes_habilitadas": False,
    "modo_seguro": True,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _mercado_rows_by_name() -> Dict[str, Dict[str, Any]]:
    data = read_json(REALITY_AUDIT_PATH, {})
    rows = data.get("mercados", []) if isinstance(data, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        mercado = str(row.get("mercado") or "").strip()
        if mercado:
            out[mercado] = row
    return out


def _governance_motivos() -> Dict[str, List[str]]:
    data = read_json(MARKET_GOVERNANCE_PATH, {})
    motivos = data.get("mercado_motivos", {}) if isinstance(data, dict) else {}
    out: Dict[str, List[str]] = {}
    if isinstance(motivos, dict):
        for mercado, values in motivos.items():
            if isinstance(values, list):
                out[str(mercado)] = [str(v) for v in values]
            elif values:
                out[str(mercado)] = [str(values)]
    return out


def _calibration_by_mercado() -> Dict[str, Dict[str, Any]]:
    data = read_json(CALIBRATION_RECOVERY_PATH, {})
    mercados = data.get("mercados", {}) if isinstance(data, dict) else {}
    return mercados if isinstance(mercados, dict) else {}


def _history_stability_by_mercado() -> Dict[str, Dict[str, Any]]:
    """Compute simple monthly ROI stability from closed historical records."""
    if not HISTORY_PATH.exists():
        return {}
    by_mercado_month: Dict[str, Dict[str, Dict[str, float]]] = {}
    with HISTORY_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mercado = str(row.get("mercado") or row.get("mercado") or row.get("Mercado") or "").strip()
            if not mercado:
                continue
            result = normalize_result(row.get("resultado") or row.get("result") or row.get("status"))
            if result not in {"WIN", "LOSS"}:
                continue
            date_raw = str(row.get("data") or row.get("date") or row.get("Data") or "")[:10]
            month = date_raw[:7] if len(date_raw) >= 7 else "unknown"
            stake = _as_float(row.get("stake") or row.get("valor_aposta") or row.get("valor") or 1.0, 1.0)
            lucro = row.get("lucro") or row.get("profit") or row.get("pnl")
            if lucro is None or lucro == "":
                odd = _as_float(row.get("odd") or row.get("odds") or 0)
                lucro_val = stake * (odd - 1.0) if result == "WIN" else -stake
            else:
                lucro_val = _as_float(lucro)
            slot = by_mercado_month.setdefault(mercado, {}).setdefault(month, {"stake": 0.0, "profit": 0.0, "bets": 0.0})
            slot["stake"] += max(stake, 0.0)
            slot["profit"] += lucro_val
            slot["bets"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for mercado, months in by_mercado_month.items():
        rois = []
        profitable = 0
        for slot in months.values():
            if slot["stake"] > 0:
                roi = slot["profit"] / slot["stake"]
                rois.append(roi)
                if roi > 0:
                    profitable += 1
        if not rois:
            continue
        vol = statistics.pstdev(rois) if len(rois) > 1 else 0.0
        positive_ratio = profitable / len(rois)
        stability = max(0.0, min(1.0, positive_ratio * 0.7 + (1.0 - min(vol, 1.0)) * 0.3))
        out[mercado] = {
            "months": len(rois),
            "monthly_roi_mean": statistics.mean(rois),
            "monthly_roi_stdev": vol,
            "positive_month_ratio": positive_ratio,
            "stability_score_0_1": stability,
        }
    return out


def _component_scores(row: Dict[str, Any], calibration: Dict[str, Any], stability: Dict[str, Any]) -> Dict[str, Any]:
    prod = row.get("production", {}) if isinstance(row.get("production"), dict) else {}
    status_operacional_real = str(row.get("status_operacional_real") or "DESCONHECIDA").upper()
    roi_real = _as_float(prod.get("roi_real"))
    winrate = _as_float(prod.get("taxa_acerto_real"))
    bets = _as_int(prod.get("total_apostas_real"))
    drawdown = _as_float(prod.get("drawdown_real"), 1.0)

    cal_status = str(calibration.get("status") or "DESCONHECIDA").upper()
    cal_error = _as_float(calibration.get("calibration_error_after"), _as_float(calibration.get("calibration_error_before"), 1.0))
    cal_ok = cal_status == "RECOVERED" and cal_error <= 0.10

    # Explicit weighted scoring. It is diagnostic only and cannot unlock a mercado.
    roi_score = max(0.0, min(25.0, roi_real * 50.0)) if roi_real > 0 else 0.0
    winrate_score = max(0.0, min(20.0, (winrate - MIN_WINRATE) / 0.25 * 20.0)) if winrate > MIN_WINRATE else 0.0
    calibration_score = 20.0 if cal_ok else max(0.0, min(20.0, (0.20 - cal_error) / 0.20 * 20.0))
    drawdown_score = max(0.0, min(15.0, (MAX_ACCEPTABLE_DRAWDOWN - drawdown) / MAX_ACCEPTABLE_DRAWDOWN * 15.0)) if drawdown < MAX_ACCEPTABLE_DRAWDOWN else 0.0
    sample_score = max(0.0, min(10.0, bets / MIN_BETS_FOR_ATIVA * 10.0))
    stability_score = max(0.0, min(10.0, _as_float(stability.get("stability_score_0_1")) * 10.0))

    total = round(roi_score + winrate_score + calibration_score + drawdown_score + sample_score + stability_score, 2)
    return {
        "pontuacao_promocao": total,
        "components": {
            "roi_score": round(roi_score, 2),
            "winrate_score": round(winrate_score, 2),
            "calibration_score": round(calibration_score, 2),
            "drawdown_score": round(drawdown_score, 2),
            "sample_score": round(sample_score, 2),
            "stability_score": round(stability_score, 2),
        },
        "evidence": {
            "roi_real": roi_real,
            "taxa_acerto_real": winrate,
            "apostas_reais": bets,
            "drawdown_real": drawdown,
            "calibration_status": cal_status,
            "calibration_error_after": cal_error,
            "status_operacional_real": status_operacional_real,
            **stability,
        },
    }


def _candidate_status(status_ciclo_vida: str, row: Dict[str, Any], scores: Dict[str, Any], governance_motivos: List[str]) -> Tuple[str, str, bool]:
    prod = row.get("production", {}) if isinstance(row.get("production"), dict) else {}
    status_operacional_real = str(row.get("status_operacional_real") or "DESCONHECIDA").upper()
    action = str(row.get("acao_recomendada") or "").upper()
    roi_real = _as_float(prod.get("roi_real"))
    bets = _as_int(prod.get("total_apostas_real"))
    drawdown = _as_float(prod.get("drawdown_real"), 1.0)
    score = _as_float(scores.get("pontuacao_promocao"))
    cal_ok = str(scores.get("evidence", {}).get("calibration_status") or "").upper() == "RECOVERED" and _as_float(scores.get("evidence", {}).get("calibration_error_after"), 1.0) <= 0.10

    if status_ciclo_vida == "APOSENTADA" or status_operacional_real in {"SUSPEITA_DE_SOBREAJUSTE", "REPROVADA_NA_OPERACAO"} or action == "APOSENTAR_MERCADO":
        return "APOSENTADA_LOCKED", "MERCADO_APOSENTADO_OR_REPROVADA_NA_OPERACAO", False
    if status_ciclo_vida == "BLOQUEADA" or any(r not in {"INSUFFICIENT_SAMPLE"} for r in governance_motivos):
        return "BLOQUEADA_BY_GOVERNANCE", "MARKET_GOVERNANCE_BLOCK", False
    if bets < MIN_BETS_FOR_ATIVA:
        return "OBSERVACAO_NEEDS_SAMPLE", "AMOSTRA_REAL_INSUFICIENTE", False
    if roi_real <= 0:
        return "OBSERVACAO_NAO_PRONTA", "NON_POSITIVE_REAL_ROI", False
    if not cal_ok:
        return "OBSERVACAO_NAO_PRONTA", "CALIBRATION_NOT_OK", False
    if drawdown > MAX_ACCEPTABLE_DRAWDOWN:
        return "OBSERVACAO_NAO_PRONTA", "DRAWDOWN_ABOVE_LIMIT", False
    if status_operacional_real != "APROVADA_NA_OPERACAO":
        return "OBSERVACAO_NAO_PRONTA", "REALITY_STATUS_NOT_OK", False
    if score < MIN_PROMOTION_SCORE:
        return "OBSERVACAO_NAO_PRONTA", "PROMOTION_SCORE_BELOW_THRESHOLD", False

    # Safe mode: this is an analytical candidate only. Manual review and the existing
    # guards are still required; no lifecycle status is changed to ATIVA here.
    return "ELIGIBLE_FOR_MANUAL_REVIEW", "PROMOTION_CRITERIA_MET_SIMULATION_ONLY", True


def run_promotion_engine() -> Dict[str, Any]:
    lifecycle = read_json(LIFECYCLE_PATH, {})
    mercados_state = lifecycle.get("mercados", {}) if isinstance(lifecycle, dict) else {}
    reality_rows = _mercado_rows_by_name()
    calibration = _calibration_by_mercado()
    gov_motivos = _governance_motivos()
    stability_map = _history_stability_by_mercado()
    now = now_iso()

    report_mercados: Dict[str, Dict[str, Any]] = {}
    classificacao_rows: List[Dict[str, Any]] = []
    replacement_candidates: List[Dict[str, Any]] = []

    all_mercados = sorted(set(mercados_state.keys()) | set(reality_rows.keys()))
    for mercado in all_mercados:
        lifecycle_info = mercados_state.get(mercado, {}) if isinstance(mercados_state, dict) else {}
        status_ciclo_vida = str((lifecycle_info or {}).get("status_ciclo_vida") or "DESCONHECIDA").upper()
        row = reality_rows.get(mercado, {"mercado": mercado, "production": {}, "status_operacional_real": "DESCONHECIDA"})
        scores = _component_scores(row, calibration.get(mercado, {}), stability_map.get(mercado, {}))
        candidate_status, motivo, eligible = _candidate_status(status_ciclo_vida, row, scores, gov_motivos.get(mercado, []))

        item = {
            "mercado": mercado,
            "status_ciclo_vida": status_ciclo_vida,
            "candidate_status": candidate_status,
            "pontuacao_promocao": scores["pontuacao_promocao"],
            "motivo": motivo,
            "eligible_for_manual_review": eligible,
            "would_change_lifecycle": False,
            "target_status_ciclo_vida": status_ciclo_vida,
            "simulation_only": True,
            "apostas_reais_habilitadas": False,
            "recomendacoes_habilitadas": False,
            "score_breakdown": scores["components"],
            "evidence": scores["evidence"],
            "governance_motivos": gov_motivos.get(mercado, []),
        }
        report_mercados[mercado] = item
        classificacao_rows.append({
            "rank": 0,
            "mercado": mercado,
            "pontuacao_promocao": scores["pontuacao_promocao"],
            "status_ciclo_vida": status_ciclo_vida,
            "candidate_status": candidate_status,
            "motivo": motivo,
            "eligible_for_manual_review": eligible,
        })
        if status_ciclo_vida != "APOSENTADA":
            replacement_candidates.append({
                "rank": 0,
                "mercado": mercado,
                "pontuacao_promocao": scores["pontuacao_promocao"],
                "candidate_status": candidate_status,
                "motivo": motivo,
                "eligible_for_manual_review": eligible,
            })

    classificacao_rows.sort(key=lambda x: (_as_float(x.get("pontuacao_promocao")), bool(x.get("eligible_for_manual_review"))), reverse=True)
    replacement_candidates.sort(key=lambda x: (_as_float(x.get("pontuacao_promocao")), bool(x.get("eligible_for_manual_review"))), reverse=True)
    for i, item in enumerate(classificacao_rows, 1):
        item["rank"] = i
    for i, item in enumerate(replacement_candidates, 1):
        item["rank"] = i

    promoted = [m for m, x in report_mercados.items() if x.get("eligible_for_manual_review")]
    watchlist = [m for m, x in report_mercados.items() if str(x.get("status_ciclo_vida")).upper() == "OBSERVACAO"]
    retired = [m for m, x in report_mercados.items() if str(x.get("status_ciclo_vida")).upper() == "APOSENTADA"]

    report = {
        "generated_at": now,
        "phase": "FASE_18_MARKET_PROMOTION_ENGINE",
        "scope": "simulation_analysis_only",
        **SAFE_FLAGS,
        "status": "OK" if report_mercados else "NO_DATA",
        "promotion_policy": {
            "min_bets_for_active": MIN_BETS_FOR_ATIVA,
            "min_pontuacao_promocao": MIN_PROMOTION_SCORE,
            "max_acceptable_drawdown": MAX_ACCEPTABLE_DRAWDOWN,
            "min_winrate": MIN_WINRATE,
            "requires_calibration_ok": True,
            "requires_reality_ok": True,
            "automatic_lifecycle_promotion_enabled": False,
        },
        "summary": {
            "mercados_evaluated": len(report_mercados),
            "eligible_for_manual_review": len(promoted),
            "watchlist": len(watchlist),
            "retired": len(retired),
            "active_created_by_this_engine": 0,
        },
        "mercados": report_mercados,
    }

    classificacao = {
        "generated_at": now,
        "scope": "simulation_analysis_only",
        **SAFE_FLAGS,
        "classificacao_basis": "pontuacao_promocao_diagnostic_only",
        "mercados": classificacao_rows,
    }
    replacements = {
        "generated_at": now,
        "scope": "simulation_analysis_only",
        **SAFE_FLAGS,
        "retired_mercado_to_replace": "DC_12",
        "note": "Candidates are analytical only; this report does not activate mercados or enable recommendations.",
        "candidates": replacement_candidates,
    }

    write_json(PROMOTION_REPORT_PATH, report)
    write_json(MARKET_RANKING_PATH, classificacao)
    write_json(REPLACEMENT_CANDIDATES_PATH, replacements)

    # Annotate lifecycle with promotion_simulation but do not change status_ciclo_vida.
    if isinstance(lifecycle, dict) and isinstance(mercados_state, dict):
        for mercado, item in report_mercados.items():
            if mercado not in mercados_state or not isinstance(mercados_state.get(mercado), dict):
                mercados_state[mercado] = {}
            mercados_state[mercado]["promotion_simulation"] = {
                "last_evaluated_at": now,
                "candidate_status": item["candidate_status"],
                "pontuacao_promocao": item["pontuacao_promocao"],
                "motivo": item["motivo"],
                "eligible_for_manual_review": item["eligible_for_manual_review"],
                "automatic_promotion_enabled": False,
            }
            mercados_state[mercado]["modo_simulacao"] = True
            mercados_state[mercado]["apostas_reais_habilitadas"] = False
            mercados_state[mercado]["recomendacoes_habilitadas"] = False
        lifecycle["mercados"] = mercados_state
        lifecycle["last_promotion_simulation_at"] = now
        lifecycle["automatic_lifecycle_promotion_enabled"] = False
        lifecycle["apostas_reais_habilitadas"] = False
        lifecycle["recomendacoes_habilitadas"] = False
        write_json(LIFECYCLE_PATH, lifecycle)

    # Record audit trail entry without lifecycle transition.
    trail = read_json(MARKET_LIFECYCLE_AUDIT_TRAIL_PATH, [])
    if not isinstance(trail, list):
        trail = []
    trail.append({
        "timestamp": now,
        "event": "PROMOTION_ENGINE_SIMULATION_RUN",
        "mercados_evaluated": len(report_mercados),
        "eligible_for_manual_review": promoted,
        "automatic_lifecycle_promotion_enabled": False,
        "apostas_reais_habilitadas": False,
        "recomendacoes_habilitadas": False,
    })
    write_json(MARKET_LIFECYCLE_AUDIT_TRAIL_PATH, trail)
    return report


def main() -> None:
    report = run_promotion_engine()
    classificacao = read_json(MARKET_RANKING_PATH, {}).get("mercados", [])
    replacements = read_json(REPLACEMENT_CANDIDATES_PATH, {}).get("candidates", [])

    print("\n" + "=" * 72)
    print("FASE 18 — Motor de Promoção de Mercados")
    print("=" * 72)
    print("Modo: SIMULAÇÃO/AUDITORIA")
    print("Apostas reais habilitadas: false")
    print("Recomendações habilitadas: false")
    print("Promoção automática para ATIVA: false")
    print(f"Status: {report.get('status')}")
    print("\nMercados elegíveis para revisão manual:")
    eligible = [m for m, x in report.get("mercados", {}).items() if x.get("eligible_for_manual_review")]
    if eligible:
        for mercado in eligible:
            print(f"  - {mercado}")
    else:
        print("  - nenhum")
    print("\nRanking diagnóstico:")
    for item in classificacao[:5]:
        print(f"  {item.get('rank')}. {item.get('mercado')} — score {item.get('pontuacao_promocao')} — {item.get('candidate_status')}")
    print("\nCandidatos analíticos para substituir DC_12:")
    for item in replacements[:5]:
        print(f"  {item.get('rank')}. {item.get('mercado')} — score {item.get('pontuacao_promocao')} — {item.get('motivo')}")
    print("\nArquivos gerados:")
    print(f"  - {PROMOTION_REPORT_PATH.relative_to(ROOT_DIR)}")
    print(f"  - {MARKET_RANKING_PATH.relative_to(ROOT_DIR)}")
    print(f"  - {REPLACEMENT_CANDIDATES_PATH.relative_to(ROOT_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
