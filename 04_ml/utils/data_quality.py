#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality checks locais para o Football Lab.

Este módulo só audita arquivos e gera relatórios. Não recomenda entradas,
não altera histórico e não executa apostas.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
HISTORICO_FILE = BASE_DIR / "banca" / "historico_apostas.csv"
BACKTEST_SUMMARY = BASE_DIR.parent / "03_backtest" / "results" / "resumo.csv"


def _load_valid_markets() -> set[str]:
    """Carrega o catálogo produzido pelo backtest em vez de manter uma
    whitelist local que envelhece separadamente do restante do projeto.
    """
    valid = {"DC_12", "TG_HT_U05", "TG_HT_U15", "TG_HT_U25"}
    if BACKTEST_SUMMARY.exists():
        try:
            summary = pd.read_csv(BACKTEST_SUMMARY, usecols=["mercado"], low_memory=False)
            valid.update(summary["mercado"].dropna().astype(str).str.strip())
        except (ValueError, pd.errors.EmptyDataError):
            pass
    return {market for market in valid if market}


VALID_MARKETS = _load_valid_markets()
VALID_MARKET_PATTERN = re.compile(
    r"^(?:R_(?:FT|HT)_[HDA]|TG_(?:FT|HT)_[OU]\d+|BTTS_[YN]|DC_(?:12|1X|X2)|"
    r"AH_(?:HOME|AWAY)_(?:POS|NEG)_\d+_\d+|EH_(?:HOME|DRAW|AWAY)_(?:POS|NEG)_\d+_\d+|"
    r"CS_\d+_\d+)$"
)

CRITICO_COLUMNS = ["data", "jogo", "liga", "mercado", "odd", "valor_apostado", "resultado", "lucro"]
VALID_RESULTS = {
    "ganhou", "perdeu", "pendente", "push", "anulada", "anulado",
    "win", "loss", "pending", "void",
}


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_history(path: Path = HISTORICO_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CRITICO_COLUMNS)
    return pd.read_csv(path)


def run_quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    rows = int(len(df))

    missing_columns = [c for c in CRITICO_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append({"severity": "critical", "check": "missing_columns", "columns": missing_columns})

    work = df.copy()
    for col in CRITICO_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA

    # Normalizações locais apenas para auditoria.
    work["odd_num"] = _safe_float_series(work["odd"])
    work["stake_num"] = _safe_float_series(work["valor_apostado"])
    work["data_dt"] = pd.to_datetime(work["data"], errors="coerce")

    duplicates_subset = [c for c in ["data", "jogo", "liga", "mercado", "odd"] if c in work.columns]
    duplicated_rows = int(work.duplicated(subset=duplicates_subset, keep=False).sum()) if duplicates_subset else 0
    if duplicated_rows:
        issues.append({"severity": "high", "check": "duplicated_games", "rows": duplicated_rows})

    invalid_dates = int(work["data_dt"].isna().sum())
    if invalid_dates:
        issues.append({"severity": "high", "check": "invalid_dates", "rows": invalid_dates})

    invalid_odds = int((work["odd_num"].isna() | (work["odd_num"] <= 1.0)).sum())
    if invalid_odds:
        issues.append({"severity": "high", "check": "invalid_or_low_odds", "rows": invalid_odds})

    invalid_stake = int((work["stake_num"].isna() | (work["stake_num"] <= 0)).sum())
    if invalid_stake:
        issues.append({"severity": "medium", "check": "invalid_stake", "rows": invalid_stake})

    empty_ligas = int(work["liga"].astype(str).str.strip().isin(["", "nan", "None"]).sum())
    if empty_ligas:
        issues.append({"severity": "medium", "check": "empty_ligas", "rows": empty_ligas})

    resultados = work["resultado"].fillna("").astype(str).str.strip().str.lower()
    invalid_results_mask = ~resultados.isin(VALID_RESULTS)
    invalid_results = int(invalid_results_mask.sum())
    if invalid_results:
        examples = sorted(resultados[invalid_results_mask].unique().tolist())[:20]
        issues.append({
            "severity": "high",
            "check": "invalid_results",
            "rows": invalid_results,
            "examples": examples,
        })

    mercados_normalizados = work["mercado"].astype(str).str.strip().str.upper()
    invalid_mercados_mask = ~(
        mercados_normalizados.isin(VALID_MARKETS)
        | mercados_normalizados.str.match(VALID_MARKET_PATTERN)
    )
    invalid_mercados = int(invalid_mercados_mask.sum())
    if invalid_mercados:
        examples = sorted(work.loc[invalid_mercados_mask, "mercado"].astype(str).dropna().unique().tolist())[:20]
        issues.append({"severity": "medium", "check": "invalid_mercados", "rows": invalid_mercados, "examples": examples})

    nulls_by_col = {c: int(work[c].isna().sum()) for c in CRITICO_COLUMNS if int(work[c].isna().sum()) > 0}
    if nulls_by_col:
        issues.append({"severity": "medium", "check": "nulls_in_critical_columns", "columns": nulls_by_col})

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    max_sev = max((severity_order.get(i["severity"], 0) for i in issues), default=0)
    status = "OK" if not issues else ("CRITICO" if max_sev >= 4 else "ALERTA")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "rows": rows,
        "issues_count": len(issues),
        "issues": issues,
    }


def write_quality_report(output_path: Path | None = None) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_history()
    report = run_quality_checks(df)
    output_path = output_path or (REPORTS_DIR / "relatorio_qualidade_dados.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = write_quality_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
