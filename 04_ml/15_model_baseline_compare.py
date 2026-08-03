#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 9 — Comparação baseline vs métricas atuais.

Compara baseline_metrics.json, resumo_deriva.json e registry para apontar
risco de degradação. Não troca modelo automaticamente.
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

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ALERTS_DIR = BASE_DIR / "alerts"

MAX_BRIER_WORSENING = 0.05
MIN_EVALUATED_RECORDS = 100


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = read_json(MODELS_DIR / "baseline_metrics.json", {})
    drift = read_json(REPORTS_DIR / "resumo_deriva.json", {})
    registry = read_json(REPORTS_DIR / "relatorio_registro_modelos.json", {})

    base_overall = baseline.get("overall") or {}
    drift_overall = drift.get("overall") or {}
    base_brier = num(base_overall.get("brier_score") or base_overall.get("brier"))
    current_brier = num(drift_overall.get("brier_score") or drift_overall.get("brier"))
    evaluated = int(drift_overall.get("evaluated_records", 0) or 0) if isinstance(drift_overall, dict) else 0

    checks: list[dict[str, Any]] = []
    status = "OK"

    if base_brier is None:
        checks.append({"check": "baseline_brier", "status": "NO_DATA", "detail": "baseline_metrics.json sem Brier Score"})
        status = "NO_DATA"
    if current_brier is None:
        checks.append({"check": "current_brier", "status": "NO_DATA", "detail": "resumo_deriva.json sem Brier atual"})
        status = "NO_DATA"
    if evaluated and evaluated < MIN_EVALUATED_RECORDS:
        checks.append({"check": "sample_size", "status": "ALERTA", "detail": f"amostra avaliada baixa: {evaluated}"})
        if status == "OK":
            status = "ALERTA"

    delta_pct = None
    if base_brier is not None and current_brier is not None and base_brier > 0:
        delta_pct = (current_brier - base_brier) / base_brier
        if delta_pct > MAX_BRIER_WORSENING:
            status = "BLOQUEADA_ANALYTICS"
            checks.append({"check": "brier_delta", "status": "CRITICO", "detail": f"Brier piorou {delta_pct:.2%} vs baseline"})
        else:
            checks.append({"check": "brier_delta", "status": "OK", "detail": f"delta {delta_pct:.2%}"})

    model_count = int(registry.get("deployable_models", 0) or 0) if isinstance(registry, dict) else 0
    if model_count == 0:
        checks.append({
            "check": "model_registry",
            "status": "NO_DEPLOYABLE_MODEL",
            "detail": "nenhum modelo presente no resumo ativo; artefatos antigos não contam como publicáveis",
        })
        if status == "OK":
            status = "NO_DATA"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "analytical_baseline_compare_only",
        "status": status,
        "baseline_brier": base_brier,
        "current_brier": current_brier,
        "brier_delta_pct": round(delta_pct, 6) if delta_pct is not None else None,
        "evaluated_records": evaluated,
        "deployable_models": model_count,
        "checks": checks,
        "policy": {"max_brier_worsening": MAX_BRIER_WORSENING, "min_evaluated_records": MIN_EVALUATED_RECORDS},
        "note": "Comparação analítica. Não troca modelo nem recomenda entrada.",
    }
    (REPORTS_DIR / "comparacao_modelo_base.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ALERTS_DIR / "model_baseline_alerts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
