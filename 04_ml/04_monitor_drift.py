#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
DRIFT_REPORT_CSV = REPORTS_DIR / "relatorio_deriva.csv"
DRIFT_SUMMARY_JSON = REPORTS_DIR / "resumo_deriva.json"
CALIBRATION_GUARD_JSON = REPORTS_DIR / "estado_guard_calibracao.json"

HISTORICO_CANDIDATES = [
    BASE_DIR / "historico_apostas.csv",
    BASE_DIR / "banca" / "historico_apostas.csv",
]

CALIBRATION_OK_MAX = 0.05
CALIBRATION_ALERTA_MAX = 0.10
MIN_EVALUATED_FOR_ALERT = 30

COLUMN_ALIASES = {
    "probability": [
        "probabilidade_calibrada", "prob_modelo", "prob_sim", "prob", "probabilidade", "probabilidade_prevista",
        "predicted_probability", "avg_predicted_probability", "confianca", "confianca", "p_modelo", "p",
    ],
    "raw_probability": [
        "probabilidade_bruta", "prob_modelo", "prob_sim", "prob", "probabilidade", "probabilidade_prevista",
        "predicted_probability", "avg_predicted_probability", "confianca", "confianca", "p_modelo", "p",
    ],
    "result": [
        "resultado", "result", "status", "resultado_aposta", "outcome", "ganhou",
        "win", "acerto", "hit",
    ],
    "stake": ["valor_apostado", "stake", "aposta", "valor", "bet_amount"],
    "profit": ["lucro", "profit", "pnl", "resultado_financeiro", "lucro_realizado"],
    "mercado": ["mercado", "mercado"],
    "liga": ["liga", "liga", "competition", "competicao", "campeonato"],
}

POSITIVE_RESULTS = {"ganhou", "win", "won", "green", "acerto", "hit", "true", "1", "sim", "s", "yes", "y"}
NEGATIVE_RESULTS = {"perdeu", "loss", "lost", "red", "erro", "miss", "false", "0", "nao", "não", "n", "no"}
PENDING_RESULTS = {"pendente", "pending", "open", "aberta", "em aberto", "void", "cancelada", "cancelado", "push", ""}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_first_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        found = by_lower.get(alias.lower())
        if found is not None:
            return found
    return None


def detect_csv_separator(path: Path) -> str:
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return ","
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_history() -> tuple[pd.DataFrame, Path | None, str | None]:
    for path in HISTORICO_CANDIDATES:
        if not path.exists():
            continue
        try:
            sep = detect_csv_separator(path)
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            df = df.dropna(how="all")
            return df, path, None
        except UnicodeDecodeError:
            try:
                sep = detect_csv_separator(path)
                df = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
                df = df.dropna(how="all")
                return df, path, None
            except Exception as exc:
                return pd.DataFrame(), path, str(exc)
        except Exception as exc:
            return pd.DataFrame(), path, str(exc)
    return pd.DataFrame(), None, "Nenhum histórico encontrado em 04_ml/historico_apostas.csv ou 04_ml/banca/historico_apostas.csv."


def parse_probability(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(numeric <= 1, numeric / 100.0)
    return numeric.clip(lower=0, upper=1)


def parse_result(value) -> float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except Exception:
        pass

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    text = str(value).strip().lower()
    if text in POSITIVE_RESULTS:
        return 1.0
    if text in NEGATIVE_RESULTS:
        return 0.0
    if text in PENDING_RESULTS:
        return math.nan

    numeric = pd.to_numeric(pd.Series([text.replace(",", ".")]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return math.nan
    if numeric == 1:
        return 1.0
    if numeric == 0:
        return 0.0
    return math.nan


def alert_level(calibration_error: float, evaluated_records: int) -> str:
    if evaluated_records <= 0 or pd.isna(calibration_error):
        return "NO_DATA"
    if evaluated_records < MIN_EVALUATED_FOR_ALERT:
        return "ALERTA"
    if calibration_error <= CALIBRATION_OK_MAX:
        return "OK"
    if calibration_error <= CALIBRATION_ALERTA_MAX:
        return "ALERTA"
    return "CRITICO"


def build_metrics(df: pd.DataFrame, columns: dict[str, str | None], group_cols: list[str]) -> list[dict]:
    generated_at = utc_now_iso()
    rows: list[dict] = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]

    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))

        evaluated = grp.dropna(subset=["_predicted_probability", "_observed_result"])
        evaluated_records = int(len(evaluated))

        avg_prob = float(evaluated["_predicted_probability"].mean()) if evaluated_records else math.nan
        observed = float(evaluated["_observed_result"].mean()) if evaluated_records else math.nan
        calibration_error = abs(avg_prob - observed) if evaluated_records else math.nan
        brier_score = float(((evaluated["_predicted_probability"] - evaluated["_observed_result"]) ** 2).mean()) if evaluated_records else math.nan

        roi = math.nan
        stake_col = columns.get("stake")
        profit_col = columns.get("profit")
        if stake_col and profit_col and evaluated_records:
            stake = pd.to_numeric(evaluated[stake_col], errors="coerce").fillna(0).sum()
            profit = pd.to_numeric(evaluated[profit_col], errors="coerce").fillna(0).sum()
            if stake > 0:
                roi = float(profit / stake)

        mercado_value = key_map.get(columns.get("mercado"), "ALL") if columns.get("mercado") in group_cols else "ALL"
        liga_value = key_map.get(columns.get("liga"), "ALL") if columns.get("liga") in group_cols else "ALL"

        rows.append({
            "generated_at": generated_at,
            "total_records": int(len(grp)),
            "evaluated_records": evaluated_records,
            "mercado": "ALL" if pd.isna(mercado_value) else str(mercado_value),
            "liga": "ALL" if pd.isna(liga_value) else str(liga_value),
            "avg_predicted_probability": round(avg_prob, 6) if not pd.isna(avg_prob) else None,
            "observed_frequency": round(observed, 6) if not pd.isna(observed) else None,
            "calibration_error": round(calibration_error, 6) if not pd.isna(calibration_error) else None,
            "brier_score": round(brier_score, 6) if not pd.isna(brier_score) else None,
            "roi": round(roi, 6) if not pd.isna(roi) else None,
            "alert_level": alert_level(calibration_error, evaluated_records),
        })

    return rows



def load_existing_guard_calibracao() -> dict:
    try:
        if CALIBRATION_GUARD_JSON.exists():
            data = json.loads(CALIBRATION_GUARD_JSON.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def operational_calibration_view(summary: dict) -> dict:
    """Retorna visão operacional pós-recovery sem relaxar limites."""
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    raw_error = summary.get("raw_overall", {}).get("calibration_error") if isinstance(summary, dict) else None
    calibrated_error = overall.get("calibration_error") if isinstance(overall, dict) else None
    source = "drift_history"

    existing_guard = load_existing_guard_calibracao()
    if existing_guard.get("source") == "calibration_recovery_report":
        raw_error = existing_guard.get("raw_calibration_error", raw_error)
        calibrated_error = existing_guard.get("calibrated_calibration_error", calibrated_error)
        source = "calibration_recovery_report"
    else:
        recovery_path = REPORTS_DIR / "relatorio_recuperacao_calibracao.json"
        if recovery_path.exists():
            try:
                recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
                raw_error = (recovery.get("global_before") or {}).get("calibration_error", raw_error)
                calibrated_error = (recovery.get("global_after") or {}).get("calibration_error", calibrated_error)
                source = "calibration_recovery_report"
            except Exception:
                pass

    status = "OK" if calibrated_error is not None and float(calibrated_error) <= CALIBRATION_ALERTA_MAX else ("NO_DATA" if calibrated_error is None else "CRITICO")
    blocked = not (calibrated_error is not None and float(calibrated_error) <= CALIBRATION_ALERTA_MAX)
    return {
        "raw_calibration_error": raw_error,
        "calibrated_calibration_error": calibrated_error,
        "status": status,
        "blocked": blocked,
        "source": source,
    }

def save_outputs(report_rows: list[dict], summary: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(DRIFT_REPORT_CSV, index=False, encoding="utf-8-sig")
    DRIFT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    view = operational_calibration_view(summary)
    guard_state = {
        "generated_at": summary.get("generated_at") if isinstance(summary, dict) else utc_now_iso(),
        "raw_calibration_error": view.get("raw_calibration_error"),
        "calibrated_calibration_error": view.get("calibrated_calibration_error"),
        "status": view.get("status"),
        "blocked": view.get("blocked"),
        "limit": CALIBRATION_ALERTA_MAX,
        "evaluated_records": overall.get("evaluated_records"),
        "source": view.get("source"),
    }
    CALIBRATION_GUARD_JSON.write_text(json.dumps(guard_state, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    df, history_path, load_error = load_history()
    generated_at = utc_now_iso()

    empty_report = [{
        "generated_at": generated_at,
        "total_records": 0,
        "evaluated_records": 0,
        "mercado": "ALL",
        "liga": "ALL",
        "avg_predicted_probability": None,
        "observed_frequency": None,
        "calibration_error": None,
        "brier_score": None,
        "roi": None,
        "alert_level": "NO_DATA",
    }]

    if load_error and history_path is None:
        summary = {
            "generated_at": generated_at,
            "status": "NO_HISTORY",
            "message": load_error,
            "history_path": None,
            "reports": {
                "drift_report_csv": str(DRIFT_REPORT_CSV.relative_to(BASE_DIR.parent)),
                "drift_summary_json": str(DRIFT_SUMMARY_JSON.relative_to(BASE_DIR.parent)),
            },
        }
        save_outputs(empty_report, summary)
        print(f"⚠️  {load_error}")
        print(f"Relatório vazio salvo em: {DRIFT_REPORT_CSV}")
        return 0

    if load_error:
        summary = {
            "generated_at": generated_at,
            "status": "READ_ERRO",
            "message": load_error,
            "history_path": str(history_path) if history_path else None,
        }
        save_outputs(empty_report, summary)
        print(f"⚠️  Histórico encontrado, mas não foi possível ler: {load_error}")
        return 0

    if df.empty:
        summary = {
            "generated_at": generated_at,
            "status": "EMPTY_HISTORY",
            "message": "O histórico existe, mas está vazio.",
            "history_path": str(history_path),
        }
        save_outputs(empty_report, summary)
        print("⚠️  Histórico vazio. Nada para avaliar ainda.")
        return 0

    columns = {name: find_first_column(df, aliases) for name, aliases in COLUMN_ALIASES.items()}
    missing_required = [name for name in ["probability", "result"] if columns.get(name) is None]

    if missing_required:
        summary = {
            "generated_at": generated_at,
            "status": "MISSING_COLUMNS",
            "message": "Colunas obrigatórias ausentes para calcular drift/calibração.",
            "missing_required": missing_required,
            "detected_columns": columns,
            "available_columns": list(df.columns),
            "history_path": str(history_path),
        }
        empty_report[0]["total_records"] = int(len(df))
        save_outputs(empty_report, summary)
        print("⚠️  Não foi possível calcular drift: faltam colunas de probabilidade e/ou resultado.")
        print(f"Colunas detectadas: {columns}")
        return 0

    df = df.copy()
    df["_predicted_probability"] = parse_probability(df[columns["probability"]])
    if columns.get("raw_probability"):
        df["_raw_predicted_probability"] = parse_probability(df[columns["raw_probability"]])
    else:
        df["_raw_predicted_probability"] = df["_predicted_probability"]
    df["_observed_result"] = df[columns["result"]].apply(parse_result)

    evaluated_records = int(df.dropna(subset=["_predicted_probability", "_observed_result"]).shape[0])
    if evaluated_records == 0:
        summary = {
            "generated_at": generated_at,
            "status": "NO_EVALUATED_RECORDS",
            "message": "Ainda não há apostas/previsões liquidadas com probabilidade e resultado realizados.",
            "total_records": int(len(df)),
            "evaluated_records": 0,
            "detected_columns": columns,
            "history_path": str(history_path),
        }
        empty_report[0]["total_records"] = int(len(df))
        save_outputs(empty_report, summary)
        print("⚠️  Histórico carregado, mas não há registros liquidados suficientes para avaliar.")
        return 0

    group_sets = [[], [columns["mercado"]] if columns.get("mercado") else []]
    if columns.get("mercado") and columns.get("liga"):
        group_sets.append([columns["mercado"], columns["liga"]])
    elif columns.get("liga"):
        group_sets.append([columns["liga"]])

    report_rows: list[dict] = []
    for group_cols in group_sets:
        if group_cols or not report_rows:
            report_rows.extend(build_metrics(df, columns, group_cols))

    overall = report_rows[0]
    raw_eval = df.dropna(subset=["_raw_predicted_probability", "_observed_result"])
    raw_overall = {
        "calibration_error": None,
        "brier_score": None,
        "evaluated_records": int(len(raw_eval)),
    }
    if len(raw_eval):
        raw_overall["calibration_error"] = round(float(abs(raw_eval["_raw_predicted_probability"].mean() - raw_eval["_observed_result"].mean())), 6)
        raw_overall["brier_score"] = round(float(((raw_eval["_raw_predicted_probability"] - raw_eval["_observed_result"]) ** 2).mean()), 6)
    summary = {
        "generated_at": generated_at,
        "status": "OK",
        "history_path": str(history_path),
        "total_records": int(len(df)),
        "evaluated_records": evaluated_records,
        "detected_columns": columns,
        "thresholds": {
            "ok_max_calibration_error": CALIBRATION_OK_MAX,
            "warning_max_calibration_error": CALIBRATION_ALERTA_MAX,
            "min_evaluated_for_alert": MIN_EVALUATED_FOR_ALERT,
        },
        "overall": overall,
        "raw_overall": raw_overall,
        "probability_column_used": columns.get("probability"),
        "raw_probability_column_used": columns.get("raw_probability"),
        "reports": {
            "drift_report_csv": str(DRIFT_REPORT_CSV.relative_to(BASE_DIR.parent)),
            "drift_summary_json": str(DRIFT_SUMMARY_JSON.relative_to(BASE_DIR.parent)),
        },
    }
    save_outputs(report_rows, summary)

    print("✅ Monitoramento de drift/calibração finalizado.")
    print(f"Histórico usado: {history_path}")
    print(f"Registros totais: {len(df)}")
    print(f"Registros avaliados: {evaluated_records}")
    view = operational_calibration_view(summary)
    print(f"Erro calibração bruto: {view.get('raw_calibration_error')}")
    print(f"Erro calibração calibrado: {view.get('calibrated_calibration_error')}")
    print(f"Guard de Calibração: {view.get('status')}")
    print(f"Bloqueado: {str(view.get('blocked')).lower()}")
    print(f"Fonte: {view.get('source')}")
    print(f"Brier Score geral: {overall.get('brier_score')}")
    print(f"Alerta geral: {view.get('status')}")
    print(f"Relatório CSV: {DRIFT_REPORT_CSV}")
    print(f"Resumo JSON: {DRIFT_SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
