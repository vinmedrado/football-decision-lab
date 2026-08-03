#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compara ROI do provider operacional contra ROI recalculado no historico."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR_FALLBACK = Path(__file__).resolve().parents[1]
if str(ROOT_DIR_FALLBACK) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR_FALLBACK))

from core.performance_provider import _read_history_performance, _read_performance_json
from core.project_paths import REPORTS_DIR, now_iso, write_json

REPORT_PATH = REPORTS_DIR / "relatorio_consistencia_roi.json"
SUMMARY_PATH = REPORTS_DIR / "resumo_consistencia_roi.json"
ALERTA_LIMIT = 0.05
CRITICO_LIMIT = 0.15


def _pct_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def evaluate_roi_consistency() -> Dict[str, Any]:
    provider = _read_performance_json()
    historical, hist_alertas = _read_history_performance()
    rows: List[Dict[str, Any]] = []
    alertas: List[str] = list(hist_alertas)
    mercados = sorted(set(provider.keys()) | set(historical.keys()))
    for mercado in mercados:
        p = provider.get(mercado)
        h = historical.get(mercado)
        provider_roi = _pct_value(p.get("roi")) if p else None
        historical_roi = _pct_value(h.get("roi")) if h else None
        if p is None:
            status = "ALERTA"
            difference = None
            motivo = "MISSING_PROVIDER_MARKET"
        elif h is None:
            status = "ALERTA"
            difference = None
            motivo = "MISSING_HISTORICAL_MARKET"
        else:
            difference = abs(float(provider_roi) - float(historical_roi))
            if difference > CRITICO_LIMIT:
                status = "CRITICO"
                motivo = "INCONSISTENCIA_ROI_CRITICO"
            elif difference > ALERTA_LIMIT:
                status = "ALERTA"
                motivo = "INCONSISTENCIA_ROI"
            else:
                status = "OK"
                motivo = None
        rows.append({
            "mercado": mercado,
            "provider_roi": provider_roi,
            "historical_roi": historical_roi,
            "difference": round(difference, 6) if difference is not None else None,
            "status": status,
            "motivo": motivo,
            "provider_bets": p.get("bets") if p else None,
            "historical_bets": h.get("bets") if h else None,
        })
    summary = {
        "timestamp": now_iso(),
        "checked_mercados": len(rows),
        "consistent": sum(1 for r in rows if r["status"] == "OK"),
        "alertas": sum(1 for r in rows if r["status"] == "ALERTA"),
        "critical": sum(1 for r in rows if r["status"] == "CRITICO"),
        "status": "CRITICO" if any(r["status"] == "CRITICO" for r in rows) else ("ALERTA" if any(r["status"] == "ALERTA" for r in rows) else "OK"),
        "warning_limit": ALERTA_LIMIT,
        "critical_limit": CRITICO_LIMIT,
        "alertas_detail": alertas,
    }
    write_json(REPORT_PATH, {"timestamp": now_iso(), "mercados": rows})
    write_json(SUMMARY_PATH, summary)
    return {"summary": summary, "mercados": rows}


if __name__ == "__main__":
    print(json.dumps(evaluate_roi_consistency(), ensure_ascii=False, indent=2))
