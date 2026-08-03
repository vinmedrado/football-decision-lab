#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Football Lab — Governança de mercados e exposição.

FASE 13:
- Normaliza resultado real para WIN/LOSS/PENDING.
- Usa performance_por_mercado.json como fonte primária operacional.
- Faz validação cruzada de ROI sem bloquear por inconsistência.
- Usa paths centralizados via core/project_paths.py.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.performance_provider import get_mercado_performance
from core.project_paths import BANCA_DIR, REPORTS_DIR, validate_environment, now_iso, read_json, write_json
from core.result_normalizer import normalize_result

HISTORICO_PATH = BANCA_DIR / "historico_apostas.csv"
EXPOSURE_GUARD_PATH = REPORTS_DIR / "guard_exposicao.json"
MARKET_GOVERNANCE_PATH = REPORTS_DIR / "relatorio_governanca_mercados.json"

MIN_MARKET_WINRATE = 0.50
MIN_MARKET_BETS = 10
MAX_MARKET_SHARE = 0.30
MAX_LEAGUE_SHARE = 0.35
MAX_DAILY_RECOMMENDATIONS = 20
ROI_DEGRADATION_LIMIT = 0.30
RECENT_DAYS = 30


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    return None


def _read_history() -> Tuple[pd.DataFrame, List[str]]:
    if not HISTORICO_PATH.exists():
        return pd.DataFrame(), ["HISTORICO_APOSTAS_NOT_FOUND"]
    try:
        df = pd.read_csv(HISTORICO_PATH)
    except Exception as exc:
        return pd.DataFrame(), [f"HISTORICO_READ_ERRO:{exc}"]
    return df, ["HISTORICO_EMPTY"] if df.empty else []


def _prepare_history(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    alertas: List[str] = []
    cols = {
        "mercado": _find_col(df, ["mercado", "mercado"]),
        "liga": _find_col(df, ["liga", "liga", "League_std"]),
        "date": _find_col(df, ["data", "date", "Date"]),
        "resultado": _find_col(df, ["resultado", "result", "status"]),
        "lucro": _find_col(df, ["lucro", "profit", "pnl", "resultado_financeiro"]),
        "stake": _find_col(df, ["stake", "valor_aposta", "unidade"]),
        "odd": _find_col(df, ["odd", "odds"]),
    }
    if not cols["mercado"] or not cols["resultado"]:
        alertas.append("HISTORICO_MISSING_COLUMNS:mercado,resultado")
        return df.copy(), {k: v or "" for k, v in cols.items()}, alertas

    out = df.copy()
    out["_result_norm"] = out[cols["resultado"]].map(normalize_result)
    out["_is_win"] = out["_result_norm"] == "WIN"
    out["_is_loss"] = out["_result_norm"] == "LOSS"
    out["_is_closed"] = out["_result_norm"].isin(["WIN", "LOSS"])

    if cols["date"]:
        out["_date"] = pd.to_datetime(out[cols["date"]], errors="coerce")
    else:
        out["_date"] = pd.NaT

    if cols["lucro"]:
        out["_profit"] = pd.to_numeric(out[cols["lucro"]], errors="coerce").fillna(0.0)
    else:
        stake = pd.to_numeric(out[cols["stake"]], errors="coerce").fillna(1.0) if cols["stake"] else 1.0
        odd = pd.to_numeric(out[cols["odd"]], errors="coerce").fillna(2.0) if cols["odd"] else 2.0
        out["_profit"] = 0.0
        out.loc[out["_is_win"], "_profit"] = stake * (odd - 1.0)
        out.loc[out["_is_loss"], "_profit"] = -stake
    out["_stake"] = pd.to_numeric(out[cols["stake"]], errors="coerce").fillna(1.0) if cols["stake"] else 1.0
    return out, {k: v or "" for k, v in cols.items()}, alertas


def _evaluate_mercado_performance(performance: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    alertas: List[str] = list(performance.get("alertas", []))
    mercados = performance.get("mercados", {}) or {}
    blocked: List[str] = []
    motivos: Dict[str, List[str]] = {}
    if not mercados:
        return blocked, motivos, alertas + ["MARKET_GOVERNANCE_NO_PERFORMANCE_DATA"]

    for mercado, item in mercados.items():
        bets = int(float(item.get("bets", 0) or 0))
        roi = float(item.get("roi", 0.0) or 0.0)
        winrate = float(item.get("winrate", 0.0) or 0.0)
        mercado_motivos: List[str] = []
        if bets < MIN_MARKET_BETS:
            mercado_motivos.append("INSUFFICIENT_SAMPLE")
        if roi <= 0:
            mercado_motivos.append("ROI_NEGATIVO")
        if winrate < MIN_MARKET_WINRATE:
            mercado_motivos.append("BAIXA_TAXA_ACERTO_MERCADO")
        if mercado_motivos:
            blocked.append(str(mercado))
            motivos[str(mercado)] = mercado_motivos
    return blocked, motivos, alertas


def _evaluate_roi_degradation(df: pd.DataFrame, cols: Dict[str, str], motivos: Dict[str, List[str]]) -> List[str]:
    alertas: List[str] = []
    mercado_col = cols.get("mercado")
    if not mercado_col or mercado_col not in df.columns:
        return ["ROI_DEGRADATION_SKIPPED_NO_MARKET_COLUMN"]
    closed = df[df["_is_closed"]].copy()
    if closed.empty:
        return ["ROI_DEGRADATION_SKIPPED_NO_CLOSED_BETS"]
    max_date = closed["_date"].max() if "_date" in closed.columns else pd.NaT
    if pd.isna(max_date):
        return ["ROI_DEGRADATION_SKIPPED_NO_VALID_DATE"]
    recent = closed[closed["_date"] >= (max_date - timedelta(days=RECENT_DAYS))].copy()
    if recent.empty:
        return alertas
    for mercado, group in closed.groupby(mercado_col):
        name = str(mercado)
        full_stake = float(group["_stake"].sum()) or float(len(group))
        full_roi = float(group["_profit"].sum()) / full_stake if full_stake else 0.0
        rg = recent[recent[mercado_col].astype(str) == name]
        if len(rg) >= max(10, int(MIN_MARKET_BETS * 0.2)):
            recent_stake = float(rg["_stake"].sum()) or float(len(rg))
            recent_roi = float(rg["_profit"].sum()) / recent_stake if recent_stake else 0.0
            if full_roi > 0 and recent_roi < full_roi * (1.0 - ROI_DEGRADATION_LIMIT):
                motivos.setdefault(name, []).append("ROI_DEGRADATION")
    return alertas


def _evaluate_mercado_calibration(motivos: Dict[str, List[str]]) -> List[str]:
    alertas: List[str] = []
    report = read_json(REPORTS_DIR / "calibration_report.json", None)
    if not report:
        return ["CALIBRATION_REPORT_NOT_FOUND"]
    rows = []
    if isinstance(report, list):
        rows = report
    elif isinstance(report, dict):
        if isinstance(report.get("mercados"), list):
            rows = report["mercados"]
        elif isinstance(report.get("by_mercado"), dict):
            rows = [dict({"mercado": k}, **(v if isinstance(v, dict) else {})) for k, v in report["by_mercado"].items()]
    for item in rows:
        mercado = str(item.get("mercado") or item.get("mercado") or "").strip()
        if not mercado:
            continue
        try:
            ece = item.get("ECE", item.get("ece"))
            cal_err = item.get("calibration_error", item.get("erro_calibracao"))
            bad = (ece is not None and float(ece) > 0.10) or (cal_err is not None and float(cal_err) > 0.10)
        except Exception:
            bad = False
        if bad:
            motivos.setdefault(mercado, []).append("MARKET_CALIBRATION_FAIL")
    return alertas


def _evaluate_exposure(df: pd.DataFrame, cols: Dict[str, str], mercado_motivos: Dict[str, List[str]]) -> Dict[str, Any]:
    mercado_col = cols.get("mercado")
    liga_col = cols.get("liga")
    recent = df.copy()
    if "_date" in recent.columns and recent["_date"].notna().any():
        max_date = recent["_date"].max()
        recent = recent[recent["_date"] >= (max_date - timedelta(days=RECENT_DAYS))].copy()
    recent_total = int(len(recent))
    mercados_bloqueados: List[str] = []
    blocked_ligas: List[str] = []
    alertas: List[str] = []

    if recent_total and mercado_col and mercado_col in recent.columns:
        shares = recent[mercado_col].astype(str).value_counts(normalize=True)
        for mercado, share in shares.items():
            if float(share) > MAX_MARKET_SHARE:
                mercados_bloqueados.append(str(mercado))
                mercado_motivos.setdefault(str(mercado), []).append("MERCADO_SUPEREXPOSTO")
    if recent_total and liga_col and liga_col in recent.columns:
        shares = recent[liga_col].astype(str).value_counts(normalize=True)
        for liga, share in shares.items():
            if float(share) > MAX_LEAGUE_SHARE:
                blocked_ligas.append(str(liga))
    today_count = 0
    if "_date" in df.columns and df["_date"].notna().any():
        max_day = df["_date"].max().date()
        today_count = int((df["_date"].dt.date == max_day).sum())
    daily_limit_reached = today_count >= MAX_DAILY_RECOMMENDATIONS
    payload = {
        "timestamp": now_iso(),
        "status": "BLOQUEADA" if (mercados_bloqueados or blocked_ligas or daily_limit_reached) else "OK",
        "recent_days": RECENT_DAYS,
        "recent_records": recent_total,
        "total_records": int(len(df)),
        "max_mercado_share": MAX_MARKET_SHARE,
        "max_liga_share": MAX_LEAGUE_SHARE,
        "max_daily_recommendations": MAX_DAILY_RECOMMENDATIONS,
        "mercados_bloqueados": mercados_bloqueados,
        "blocked_ligas": blocked_ligas,
        "daily_recommendations_count": today_count,
        "daily_limit_reached": daily_limit_reached,
        "motivos": {
            "mercados": {m: ["MERCADO_SUPEREXPOSTO"] for m in mercados_bloqueados},
            "ligas": {l: ["LIGA_SUPEREXPOSTA"] for l in blocked_ligas},
            "daily": ["LIMITE_DIARIO_ATINGIDO"] if daily_limit_reached else [],
        },
        "alertas": alertas,
    }
    write_json(EXPOSURE_GUARD_PATH, payload)
    return payload


def evaluate_mercado_governance() -> Dict[str, Any]:
    env = validate_environment()
    df_raw, alertas = _read_history()
    performance = get_mercado_performance()
    blocked, motivos, perf_alertas = _evaluate_mercado_performance(performance)
    alertas.extend(perf_alertas)

    df, cols, prep_alertas = _prepare_history(df_raw) if not df_raw.empty else (pd.DataFrame(), {}, [])
    alertas.extend(prep_alertas)
    if not df.empty:
        alertas.extend(_evaluate_roi_degradation(df, cols, motivos))
    alertas.extend(_evaluate_mercado_calibration(motivos))
    exposure = _evaluate_exposure(df, cols, motivos) if not df.empty else {
        "status": "ALERTA", "mercados_bloqueados": [], "blocked_ligas": [], "daily_limit_reached": False, "alertas": ["EXPOSURE_SKIPPED_NO_HISTORY"]
    }
    write_json(EXPOSURE_GUARD_PATH, exposure)
    alertas.extend(exposure.get("alertas", []))

    blocked = sorted(set(blocked) | set(motivos.keys()) | set(exposure.get("mercados_bloqueados", [])))
    perf_mercados = sorted((performance.get("mercados") or {}).keys())
    hist_mercados = sorted(df[cols["mercado"]].dropna().astype(str).unique().tolist()) if cols.get("mercado") in df.columns else []
    all_mercados = sorted(set(perf_mercados) | set(hist_mercados))
    allowed = [m for m in all_mercados if m not in blocked]

    result = {
        "timestamp": now_iso(),
        "status": "BLOQUEADA" if blocked or exposure.get("daily_limit_reached") else "OK",
        "environment_status": env.get("status", "DESCONHECIDA"),
        "performance_source": performance.get("source", "empty"),
        "performance_provider_status": performance.get("status", "DESCONHECIDA"),
        "roi_inconsistencies": performance.get("roi_inconsistencies", []),
        "exposure_status": exposure.get("status", "DESCONHECIDA"),
        "mercados_bloqueados": blocked,
        "mercados_permitidos": allowed,
        "mercado_motivos": motivos,
        "blocked_ligas": exposure.get("blocked_ligas", []),
        "daily_limit_reached": exposure.get("daily_limit_reached", False),
        "alertas": sorted(set([w for w in alertas if w])),
        "config": {
            "MIN_MARKET_WINRATE": MIN_MARKET_WINRATE,
            "MIN_MARKET_BETS": MIN_MARKET_BETS,
            "MAX_MARKET_SHARE": MAX_MARKET_SHARE,
            "MAX_LEAGUE_SHARE": MAX_LEAGUE_SHARE,
            "MAX_DAILY_RECOMMENDATIONS": MAX_DAILY_RECOMMENDATIONS,
        },
    }
    write_json(MARKET_GOVERNANCE_PATH, result)
    return result


def main():
    payload = evaluate_mercado_governance()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()

