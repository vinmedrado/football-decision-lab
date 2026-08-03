#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance source hierarchy for operational governance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from core.project_paths import BANCA_DIR, REPORTS_DIR, now_iso, read_json, write_json
    from core.result_normalizer import normalize_result
except Exception:  # pragma: no cover - isolated execution fallback
    import sys
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from core.project_paths import BANCA_DIR, REPORTS_DIR, now_iso, read_json, write_json
    from core.result_normalizer import normalize_result

PERFORMANCE_BY_MARKET_PATH = REPORTS_DIR / "performance_por_mercado.json"
HISTORICO_PATH = BANCA_DIR / "historico_apostas.csv"
DIAGNOSTICS_PATH = REPORTS_DIR / "diagnostico_fonte_performance.json"
INCONSISTENCIA_ROI_LIMIT = 0.05


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    return None


def _read_performance_json() -> Dict[str, Dict[str, Any]]:
    raw = read_json(PERFORMANCE_BY_MARKET_PATH, None)
    if not raw:
        return {}
    rows = raw if isinstance(raw, list) else raw.get("mercados", raw.get("by_mercado", []))
    if isinstance(rows, dict):
        rows = [dict({"mercado": k}, **(v if isinstance(v, dict) else {})) for k, v in rows.items()]
    out: Dict[str, Dict[str, Any]] = {}
    for item in rows if isinstance(rows, list) else []:
        mercado = str(item.get("mercado") or item.get("mercado") or "").strip()
        if not mercado:
            continue
        bets = int(float(item.get("apostas", item.get("bets", item.get("total", 0))) or 0))
        roi = float(item.get("roi", item.get("ROI", 0.0)) or 0.0)
        winrate = float(item.get("win_rate", item.get("winrate", item.get("taxa_acerto", 0.0))) or 0.0)
        profit = float(item.get("lucro_liquido", item.get("profit", item.get("lucro", 0.0))) or 0.0)
        stake = float(item.get("stake", item.get("valor_apostado", bets)) or bets or 0.0)
        out[mercado] = {
            "mercado": mercado,
            "bets": bets,
            "roi": roi,
            "winrate": winrate,
            "profit": profit,
            "stake": stake,
            "source": "performance_por_mercado.json",
            "raw": item,
            "data_scope": item.get("data_scope", "NAO_INFORMADO"),
            "origins": item.get("origins", []),
        }
    return out


def _read_history_performance() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    alertas: List[str] = []
    if not HISTORICO_PATH.exists():
        return {}, ["HISTORICO_APOSTAS_NOT_FOUND"]
    try:
        df = pd.read_csv(HISTORICO_PATH)
    except Exception as exc:
        return {}, [f"HISTORICO_READ_ERRO:{exc}"]
    if df.empty:
        return {}, ["HISTORICO_EMPTY"]

    mercado_col = _find_col(df, ["mercado", "mercado"])
    result_col = _find_col(df, ["resultado", "result", "status"])
    profit_col = _find_col(df, ["lucro", "profit", "pnl", "resultado_financeiro"])
    # O histórico canônico da banca usa ``valor_apostado``. Sem esse alias,
    # cada linha caía no fallback de stake=1 e o ROI era inflado em 5x para
    # apostas de R$ 5,00.
    stake_col = _find_col(
        df,
        ["valor_apostado", "stake", "valor_aposta", "bet_amount", "stake_valor", "unidade"],
    )
    odd_col = _find_col(df, ["odd", "odds"])
    origin_col = _find_col(df, ["origem", "origin", "source"])
    if not stake_col:
        alertas.append("HISTORICO_MISSING_STAKE_COLUMN")
    if not mercado_col or not result_col:
        return {}, ["HISTORICO_MISSING_MARKET_OR_RESULT"]

    data = df.copy()
    data["_result_norm"] = data[result_col].map(normalize_result)
    closed = data[data["_result_norm"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return {}, ["MARKET_GOVERNANCE_NO_CLOSED_BETS"]

    if profit_col:
        closed["_profit"] = pd.to_numeric(closed[profit_col], errors="coerce").fillna(0.0)
    else:
        stake = pd.to_numeric(closed[stake_col], errors="coerce").fillna(1.0) if stake_col else 1.0
        odd = pd.to_numeric(closed[odd_col], errors="coerce").fillna(2.0) if odd_col else 2.0
        closed["_profit"] = 0.0
        closed.loc[closed["_result_norm"] == "WIN", "_profit"] = stake * (odd - 1.0)
        closed.loc[closed["_result_norm"] == "LOSS", "_profit"] = -stake
    closed["_stake"] = pd.to_numeric(closed[stake_col], errors="coerce").fillna(1.0) if stake_col else 1.0
    closed["_is_win"] = closed["_result_norm"] == "WIN"
    if origin_col:
        closed["_origin"] = closed[origin_col].fillna("nao_informada").astype(str).str.strip().str.lower()
    else:
        closed["_origin"] = "nao_informada"

    out: Dict[str, Dict[str, Any]] = {}
    for mercado, group in closed.groupby(mercado_col):
        bets = int(len(group))
        profit = float(group["_profit"].sum())
        stake_sum = float(group["_stake"].sum()) or float(bets)
        origins = sorted(set(group["_origin"].dropna().astype(str)))
        flags_simulated = [
            any(token in origin for token in ("simulad", "backfill", "paper"))
            for origin in origins
        ]
        data_scope = (
            "SIMULATED" if flags_simulated and all(flags_simulated)
            else "MIXED" if any(flags_simulated)
            else "REAL_OR_UNDECLARED"
        )
        out[str(mercado)] = {
            "mercado": str(mercado),
            "bets": bets,
            "roi": profit / stake_sum if stake_sum else 0.0,
            "winrate": float(group["_is_win"].mean()) if bets else 0.0,
            "profit": profit,
            "stake": stake_sum,
            "source": "historico_apostas.csv",
            "data_scope": data_scope,
            "origins": origins,
        }
    return out, alertas


def get_mercado_performance() -> Dict[str, Any]:
    primary = _read_performance_json()
    historical, hist_alertas = _read_history_performance()
    alertas: List[str] = list(hist_alertas)
    inconsistencies: List[Dict[str, Any]] = []

    if primary:
        source = "performance_por_mercado.json"
        metrics = primary
        for mercado, item in primary.items():
            if mercado in historical:
                diff = abs(float(item.get("roi", 0.0)) - float(historical[mercado].get("roi", 0.0)))
                if diff > INCONSISTENCIA_ROI_LIMIT:
                    inconsistencies.append({
                        "mercado": mercado,
                        "performance_roi": item.get("roi"),
                        "historical_roi": historical[mercado].get("roi"),
                        "difference": round(diff, 6),
                        "warning": "INCONSISTENCIA_ROI",
                    })
        if inconsistencies:
            alertas.append("INCONSISTENCIA_ROI")
    elif historical:
        source = "historico_apostas.csv"
        metrics = historical
    else:
        source = "empty"
        metrics = {}

    payload = {
        "timestamp": now_iso(),
        "status": "OK" if metrics else "ALERTA",
        "source": source,
        "mercados_found": len(metrics),
        "mercados": metrics,
        "roi_inconsistencies": inconsistencies,
        "alertas": alertas,
    }
    write_json(DIAGNOSTICS_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(get_mercado_performance(), ensure_ascii=False, indent=2))
