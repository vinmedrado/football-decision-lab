#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 10 — Saúde das features/datasets.

Avalia datasets usados no ML: colunas ausentes, nulos, constantes e possíveis
colunas de leakage por nome. Não altera os dados.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"

LEAKAGE_TERMS = ["resultado", "result", "placar", "score", "profit", "lucro", "stake", "green", "red", "win", "loss", "target"]
MAX_NULL_RATE_ALERTA = 0.30


def inspect_csv(path: Path) -> dict[str, Any]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"file": str(path.relative_to(BASE_DIR)), "status": "READ_ERRO", "error": str(exc)}

    rows = len(df)
    cols = list(df.columns)
    null_rates = df.isna().mean().sort_values(ascending=False) if cols else pd.Series(dtype=float)
    high_nulls = [{"column": c, "null_rate": round(float(v), 6)} for c, v in null_rates.items() if float(v) >= MAX_NULL_RATE_ALERTA]
    constants = []
    for c in cols:
        try:
            if df[c].nunique(dropna=True) <= 1:
                constants.append(c)
        except Exception:
            pass
    suspicious = [c for c in cols if any(term in str(c).lower() for term in LEAKAGE_TERMS)]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    status = "OK"
    if suspicious:
        status = "ALERTA"
    if rows == 0:
        status = "EMPTY"

    return {
        "file": str(path.relative_to(BASE_DIR)),
        "status": status,
        "rows": rows,
        "columns": len(cols),
        "numeric_columns": len(numeric_cols),
        "high_null_columns": high_nulls[:50],
        "constant_columns": constants[:100],
        "suspicious_leakage_columns": suspicious[:100],
    }


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATASET_DIR.rglob("*.csv")) if DATASET_DIR.exists() else []
    inspections = [inspect_csv(path) for path in files]
    warning_files = [x for x in inspections if x.get("status") not in {"OK"}]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "saude_features_analytics_only",
        "dataset_dir": str(DATASET_DIR),
        "files_found": len(files),
        "files_with_alertas": len(warning_files),
        "inspections": inspections,
        "rules": {"max_null_rate_warning": MAX_NULL_RATE_ALERTA, "leakage_terms": LEAKAGE_TERMS},
        "note": "Relatório de saúde de features. Não altera dados nem modelos.",
    }
    (REPORTS_DIR / "relatorio_saude_features.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files_found": len(files), "files_with_alertas": len(warning_files)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
