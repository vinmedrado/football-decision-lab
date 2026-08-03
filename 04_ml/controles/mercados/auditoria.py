#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoria Operacional de Mercados & Overfitting Detection.

Analytical/simulation-only audit. It compares historical/backtest indicators
against settled production history and writes reports. It does not alter models,
predict, guards, bankroll, settlement or any operational lock.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import REPORTS_DIR, ROOT_DIR, now_iso, read_json, write_json  # noqa: E402
from core.result_normalizer import normalize_result  # noqa: E402

ML_DIR = ROOT_DIR / "04_ml"
BANCA_DIR = ML_DIR / "banca"
BACKTEST_RESULTS_DIR = ROOT_DIR / "03_backtest" / "results"

HISTORICO_PATHS = [
    BANCA_DIR / "historico_apostas.csv",
    ML_DIR / "historico_apostas.csv",
]
BACKTEST_PATHS = [
    BACKTEST_RESULTS_DIR / "resumo.csv",
    REPORTS_DIR / "backtest_segmentado_mercado_liga.csv",
]
PERFORMANCE_BY_MARKET_PATH = REPORTS_DIR / "performance_por_mercado.json"
MARKET_GOVERNANCE_PATH = REPORTS_DIR / "relatorio_governanca_mercados.json"

MARKET_AUDIT_PATH = REPORTS_DIR / "auditoria_operacional_mercados.json"
REALITY_SUMMARY_PATH = REPORTS_DIR / "resumo_lacuna_operacional.json"
DC12_DEEP_DIVE_PATH = REPORTS_DIR / "analise_detalhada_dc12.json"
LEAGUE_AUDIT_PATH = REPORTS_DIR / "auditoria_operacional_ligas.json"
ODD_RANGE_AUDIT_PATH = REPORTS_DIR / "auditoria_faixas_odds.json"
CONFIDENCE_RANGE_AUDIT_PATH = REPORTS_DIR / "auditoria_faixas_confianca.json"

MIN_REAL_SAMPLE = 100
ROI_ALERTA_ABS = 0.02
OVERFIT_BACKTEST_ROI = 0.10


def _read_csv_first(paths: List[Path]) -> Tuple[Optional[pd.DataFrame], Optional[Path], str]:
    for path in paths:
        if not path.exists():
            continue
        for kwargs in ({"encoding": "utf-8-sig"}, {"encoding": "latin-1"}, {}):
            try:
                return pd.read_csv(path, low_memory=False, **kwargs), path, "OK"
            except Exception as exc:
                last = str(exc)
        return None, path, last
    return None, None, "NOT_FOUND"


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace("R$", "").replace(".", "").replace(",", ".") if "," in value else value.replace("%", "").replace("R$", "")
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _roi_from_profit_stake(profit: float, stake: float) -> float:
    return profit / stake if stake else 0.0


def _normalise_roi(value: Any) -> float:
    roi = _to_float(value, 0.0)
    # Reports in this project can use either decimal ROI (0.10) or percent (10.0).
    if abs(roi) > 2:
        return roi / 100.0
    return roi


def _load_performance_by_mercado() -> Dict[str, Dict[str, Any]]:
    data = read_json(PERFORMANCE_BY_MARKET_PATH, [])
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            mercado = str(item.get("mercado") or item.get("mercado") or "").strip()
            if mercado and mercado.lower() not in {"nan", "none", "null"}:
                result[mercado] = item
    elif isinstance(data, dict):
        for mercado, item in data.items():
            if isinstance(item, dict):
                result[str(mercado)] = item
    return result


def _settled_history(df: pd.DataFrame) -> pd.DataFrame:
    result_col = _find_col(df, ["resultado", "result", "status"])
    if result_col is None:
        df = df.copy()
        df["_normalized_result"] = "PENDING"
        return df.iloc[0:0]
    out = df.copy()
    out["_normalized_result"] = out[result_col].map(normalize_result)
    return out[out["_normalized_result"].isin(["WIN", "LOSS"])].copy()


def _production_by_mercado(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty:
        return {}
    mercado_col = _find_col(df, ["mercado", "mercado", "event"])
    odd_col = _find_col(df, ["odd", "odds", "cotacao"])
    stake_col = _find_col(df, ["valor_apostado", "stake", "valor"])
    profit_col = _find_col(df, ["lucro", "profit", "resultado_financeiro"])
    bank_col = _find_col(df, ["banca_apos", "bankroll_after", "banca"])

    if mercado_col is None:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for mercado, g in df.groupby(mercado_col, dropna=False):
        mercado = str(mercado).strip()
        if not mercado or mercado.lower() in {"nan", "none", "null"}:
            continue
        bets = int(len(g))
        wins = int((g["_normalized_result"] == "WIN").sum())
        winrate = wins / bets if bets else 0.0
        odd_mean = _to_float(g[odd_col].mean(), 0.0) if odd_col else 0.0
        stake = float(pd.to_numeric(g[stake_col], errors="coerce").fillna(0).sum()) if stake_col else float(bets)
        profit = float(pd.to_numeric(g[profit_col], errors="coerce").fillna(0).sum()) if profit_col else 0.0
        roi = _roi_from_profit_stake(profit, stake)
        drawdown = 0.0
        if bank_col:
            curve = pd.to_numeric(g[bank_col], errors="coerce").dropna().tolist()
            peak = None
            max_dd = 0.0
            for val in curve:
                peak = val if peak is None else max(peak, val)
                if peak and peak > 0:
                    max_dd = max(max_dd, (peak - val) / peak)
            drawdown = max_dd
        result[mercado] = {
            "mercado": mercado,
            "roi_real": roi,
            "taxa_acerto_real": winrate,
            "total_apostas_real": bets,
            "odd_media_real": odd_mean,
            "lucro_real": profit,
            "stake_real": stake,
            "drawdown_real": drawdown,
        }
    return result


def _load_backtest_metrics() -> Dict[str, Dict[str, Any]]:
    df, path, status = _read_csv_first(BACKTEST_PATHS)
    metrics: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return metrics

    mercado_col = _find_col(df, ["mercado", "mercado", "event"])
    if mercado_col is None:
        return metrics
    roi_col = _find_col(df, ["roi_backtest", "roi_bt", "roi_bt_medio", "roi", "roi_real"])
    wr_col = _find_col(df, ["winrate_backtest", "winrate", "taxa_acerto"])
    bets_col = _find_col(df, ["total_apostas_backtest", "apostas", "bets", "total"])
    odd_col = _find_col(df, ["odd_media_backtest", "odd_media", "avg_odd"])

    for mercado, g in df.groupby(mercado_col, dropna=False):
        mercado = str(mercado).strip()
        if not mercado or mercado.lower() in {"nan", "none", "null"}:
            continue
        metrics[mercado] = {
            "mercado": mercado,
            "roi_backtest": _normalise_roi(g[roi_col].mean()) if roi_col else 0.0,
            "winrate_backtest": _normalise_roi(g[wr_col].mean()) if wr_col else 0.0,
            "total_apostas_backtest": int(pd.to_numeric(g[bets_col], errors="coerce").fillna(0).sum()) if bets_col else int(len(g)),
            "odd_media_backtest": _to_float(g[odd_col].mean(), 0.0) if odd_col else 0.0,
            "source": str(path.relative_to(ROOT_DIR)) if path else "unknown",
        }
    return metrics


def _history_scope(df: pd.DataFrame) -> Tuple[str, list[str]]:
    origin_col = _find_col(df, ["origem", "origin", "source"])
    if origin_col is None:
        return "NAO_INFORMADO", []
    origins = sorted(
        df[origin_col].fillna("nao_informada").astype(str).str.strip().str.lower().unique().tolist()
    )
    flags_simulated = [
        any(token in origin for token in ("simulad", "backfill", "paper"))
        for origin in origins
    ]
    scope = (
        "SIMULATED" if flags_simulated and all(flags_simulated)
        else "MIXED" if any(flags_simulated)
        else "REAL_OR_UNDECLARED"
    )
    return scope, origins


def _status_and_action(
    bt: Dict[str, Any], prod: Dict[str, Any], data_scope: str
) -> Tuple[str, str]:
    rb = float(bt.get("roi_backtest", 0) or 0)
    rr = float(prod.get("roi_real", 0) or 0)
    n = int(prod.get("total_apostas_real", 0) or 0)

    if n < MIN_REAL_SAMPLE:
        return "AMOSTRA_HISTORICA_INSUFICIENTE", "OBSERVACAO"
    if data_scope in {"SIMULATED", "MIXED"}:
        if rb > OVERFIT_BACKTEST_ROI and rr < 0:
            return "SUSPEITA_DE_SOBREAJUSTE_NA_SIMULACAO", "REVISAR_MODELO"
        if rr < 0:
            return "REPROVADA_NA_SIMULACAO", "MANTER_BLOQUEADO"
        if abs(rr) <= ROI_ALERTA_ABS:
            return "ALERTA_NA_SIMULACAO", "OBSERVACAO"
        return "POSITIVA_NA_SIMULACAO_SEM_APROVACAO_REAL", "PAPER_FORWARD"
    if rb > OVERFIT_BACKTEST_ROI and rr < 0:
        return "SUSPEITA_DE_SOBREAJUSTE", "APOSENTAR_MERCADO"
    if rr < 0:
        return "REPROVADA_NA_OPERACAO", "APOSENTAR_MERCADO"
    if abs(rr) <= ROI_ALERTA_ABS:
        return "ALERTA_OPERACAO", "OBSERVACAO"
    return "APROVADA_NA_OPERACAO", "KEEP_IN_SIMULATION_ANALYSIS"


def _pontuacao_lacuna_operacional(bt: Dict[str, Any], prod: Dict[str, Any]) -> float:
    roi_gap = abs(float(bt.get("roi_backtest", 0) or 0) - float(prod.get("roi_real", 0) or 0))
    wr_gap = abs(float(bt.get("winrate_backtest", 0) or 0) - float(prod.get("taxa_acerto_real", 0) or 0))
    odd_gap = abs(float(bt.get("odd_media_backtest", 0) or 0) - float(prod.get("odd_media_real", 0) or 0))
    score = min(100.0, roi_gap * 70 + wr_gap * 20 + min(odd_gap, 2.0) * 5)
    return round(score, 4)


def _bucket_odd(odd: float) -> str:
    if odd <= 1.30:
        return "1.00-1.30"
    if odd <= 1.60:
        return "1.31-1.60"
    if odd <= 2.00:
        return "1.61-2.00"
    if odd <= 2.50:
        return "2.01-2.50"
    return "2.51+"


def _bucket_conf(conf: float) -> str:
    if conf < 0.60:
        return "50-60%"
    if conf < 0.70:
        return "60-70%"
    if conf < 0.80:
        return "70-80%"
    if conf < 0.90:
        return "80-90%"
    return "90-100%"


def _group_financial(df: pd.DataFrame, group_col: str, extra_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty or group_col not in df.columns:
        return []
    odd_col = _find_col(df, ["odd", "odds"])
    stake_col = _find_col(df, ["valor_apostado", "stake", "valor"])
    profit_col = _find_col(df, ["lucro", "profit"])
    mercado_col = _find_col(df, ["mercado", "mercado", "event"])
    out = []
    for key, g in df.groupby(group_col, dropna=False):
        bets = int(len(g))
        wins = int((g["_normalized_result"] == "WIN").sum())
        stake = float(pd.to_numeric(g[stake_col], errors="coerce").fillna(0).sum()) if stake_col else float(bets)
        profit = float(pd.to_numeric(g[profit_col], errors="coerce").fillna(0).sum()) if profit_col else 0.0
        row = {
            str(group_col): str(key),
            "apostas": bets,
            "roi": _roi_from_profit_stake(profit, stake),
            "lucro": profit,
            "winrate": wins / bets if bets else 0.0,
            "odd_media": _to_float(g[odd_col].mean(), 0.0) if odd_col else 0.0,
        }
        if mercado_col:
            row["mercados_utilizados"] = sorted({str(x) for x in g[mercado_col].dropna().unique()})
        out.append(row)
    out.sort(key=lambda x: (x.get("roi", 0), x.get("apostas", 0)), reverse=True)
    return out


def _add_liga_status(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows:
        roi = float(row.get("roi", 0) or 0)
        n = int(row.get("apostas", 0) or 0)
        if n < MIN_REAL_SAMPLE:
            row["status"] = "INSUFFICIENT_SAMPLE"
        elif roi < 0:
            row["status"] = "LEAGUE_FAIL"
        elif roi <= ROI_ALERTA_ABS:
            row["status"] = "LEAGUE_ALERTA"
        else:
            row["status"] = "LEAGUE_OK"
    return rows


def _dc12_deep_dive(hist: pd.DataFrame) -> Dict[str, Any]:
    mercado_col = _find_col(hist, ["mercado", "mercado", "event"])
    if hist is None or hist.empty or mercado_col is None:
        return {"timestamp": now_iso(), "status": "NO_DATA"}
    dc = hist[hist[mercado_col].astype(str) == "DC_12"].copy()
    if dc.empty:
        return {"timestamp": now_iso(), "status": "NO_DC_12_DATA"}

    odd_col = _find_col(dc, ["odd", "odds"])
    conf_col = _find_col(dc, ["probabilidade_calibrada", "confianca", "prob_modelo", "probabilidade", "prob"])
    data_col = _find_col(dc, ["data", "date"])
    liga_col = _find_col(dc, ["liga", "liga"])
    bank_col = _find_col(dc, ["banca_apos", "bankroll_after"])
    profit_col = _find_col(dc, ["lucro", "profit"])

    if odd_col:
        dc["odd_range"] = pd.to_numeric(dc[odd_col], errors="coerce").fillna(0).map(_bucket_odd)
    if conf_col:
        dc["faixa_confianca"] = pd.to_numeric(dc[conf_col], errors="coerce").fillna(0).map(_bucket_conf)
    if data_col:
        dt = pd.to_datetime(dc[data_col], errors="coerce")
        dc["month"] = dt.dt.to_period("M").astype(str)

    # Losing streak and cumulative drawdown from profit curve.
    max_losing_streak = 0
    current = 0
    for res in dc["_normalized_result"].tolist():
        if res == "LOSS":
            current += 1
            max_losing_streak = max(max_losing_streak, current)
        else:
            current = 0
    cumulative_drawdown = 0.0
    if profit_col:
        equity = pd.to_numeric(dc[profit_col], errors="coerce").fillna(0).cumsum()
        peak = equity.cummax()
        dd = peak - equity
        cumulative_drawdown = float(dd.max()) if len(dd) else 0.0
    if bank_col:
        curve = pd.to_numeric(dc[bank_col], errors="coerce").dropna()
        if len(curve):
            pct_dd = ((curve.cummax() - curve) / curve.cummax().replace(0, pd.NA)).fillna(0).max()
        else:
            pct_dd = 0.0
    else:
        pct_dd = 0.0

    return {
        "timestamp": now_iso(),
        "status": "OK",
        "scope": "simulation_analysis_only",
        "mercado": "DC_12",
        "overall": _production_by_mercado(dc).get("DC_12", {}),
        "roi_by_liga": _add_liga_status(_group_financial(dc, liga_col)) if liga_col else [],
        "roi_by_odd_range": _group_financial(dc, "odd_range") if "odd_range" in dc.columns else [],
        "roi_by_faixa_confianca": _group_financial(dc, "faixa_confianca") if "faixa_confianca" in dc.columns else [],
        "roi_by_month": _group_financial(dc, "month") if "month" in dc.columns else [],
        "roi_by_derived_mercado": _group_financial(dc, mercado_col) if mercado_col else [],
        "max_losing_streak": max_losing_streak,
        "drawdown_acumulado_valor": cumulative_drawdown,
        "drawdown_acumulado_pct": float(pct_dd),
    }


def run_audit() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    hist, hist_path, hist_status = _read_csv_first(HISTORICO_PATHS)
    if hist is None:
        payload = {
            "timestamp": now_iso(),
            "status": "NO_HISTORY",
            "scope": "simulation_analysis_only",
            "message": "Histórico de apostas não encontrado para auditoria analítica.",
        }
        write_json(MARKET_AUDIT_PATH, payload)
        return payload

    settled = _settled_history(hist)
    data_scope, origins = _history_scope(hist)
    backtest = _load_backtest_metrics()
    prod = _production_by_mercado(settled)
    provider = _load_performance_by_mercado()
    governance = read_json(MARKET_GOVERNANCE_PATH, {})

    all_mercados = sorted(set(backtest) | set(prod) | set(provider))
    mercado_rows = []
    for mercado in all_mercados:
        bt = backtest.get(mercado, {"mercado": mercado, "roi_backtest": 0.0, "winrate_backtest": 0.0, "total_apostas_backtest": 0, "odd_media_backtest": 0.0})
        pr = prod.get(mercado, {"mercado": mercado, "roi_real": 0.0, "taxa_acerto_real": 0.0, "total_apostas_real": 0, "odd_media_real": 0.0, "lucro_real": 0.0, "drawdown_real": 0.0})
        # Use performance_by_mercado as additional operational source when present.
        provider_row = provider.get(mercado, {})
        provider_roi = provider_row.get("roi") or provider_row.get("roi_real") or provider_row.get("ROI")
        if provider_roi is not None:
            pr["roi_real_provider"] = _normalise_roi(provider_roi)
        status, action = _status_and_action(bt, pr, data_scope)
        row = {
            "mercado": mercado,
            "backtest": bt,
            "production": pr,
            "comparison": {
                "roi_gap_signed": float(pr.get("roi_real", 0) or 0) - float(bt.get("roi_backtest", 0) or 0),
                "roi_gap_abs": abs(float(pr.get("roi_real", 0) or 0) - float(bt.get("roi_backtest", 0) or 0)),
                "roi_gap": abs(float(pr.get("roi_real", 0) or 0) - float(bt.get("roi_backtest", 0) or 0)),
                "winrate_gap": float(bt.get("winrate_backtest", 0) or 0) - float(pr.get("taxa_acerto_real", 0) or 0),
                "odd_gap": float(bt.get("odd_media_backtest", 0) or 0) - float(pr.get("odd_media_real", 0) or 0),
                "pontuacao_lacuna_operacional": _pontuacao_lacuna_operacional(bt, pr),
            },
            "status_historico": status,
            "acao_recomendada": action,
            "simulation_only": True,
        }
        mercado_rows.append(row)

    mercado_rows.sort(key=lambda r: (r["comparison"]["pontuacao_lacuna_operacional"], abs(r["production"].get("roi_real", 0))), reverse=True)

    # League audit.
    liga_col = _find_col(settled, ["liga", "liga"])
    liga_rows = _add_liga_status(_group_financial(settled, liga_col)) if liga_col else []

    # Odd range audit.
    odd_col = _find_col(settled, ["odd", "odds"])
    odd_rows = []
    if odd_col:
        tmp = settled.copy()
        tmp["odd_range"] = pd.to_numeric(tmp[odd_col], errors="coerce").fillna(0).map(_bucket_odd)
        odd_rows = _group_financial(tmp, "odd_range")

    # Confidence audit.
    conf_col = _find_col(settled, ["probabilidade_calibrada", "confianca", "prob_modelo", "probabilidade", "prob"])
    confianca_rows = []
    if conf_col:
        tmp = settled.copy()
        tmp["_conf"] = pd.to_numeric(tmp[conf_col], errors="coerce").fillna(0)
        tmp["faixa_confianca"] = tmp["_conf"].map(_bucket_conf)
        confianca_rows = _group_financial(tmp, "faixa_confianca")
        for row in confianca_rows:
            bucket_df = tmp[tmp["faixa_confianca"] == row["faixa_confianca"]]
            predicted = float(bucket_df["_conf"].mean()) if len(bucket_df) else 0.0
            actual = float((bucket_df["_normalized_result"] == "WIN").mean()) if len(bucket_df) else 0.0
            row["predicted_mean"] = predicted
            row["actual_rate"] = actual
            row["calibration_gap"] = predicted - actual
            row["probability_source"] = conf_col

    summary = {
        "timestamp": now_iso(),
        "status": "OK",
        "scope": "simulation_analysis_only" if data_scope == "SIMULATED" else "historical_analysis",
        "data_scope": data_scope,
        "origins": origins,
        "history_source": str(hist_path.relative_to(ROOT_DIR)) if hist_path else None,
        "settled_records": int(len(settled)),
        "mercados_checked": len(mercado_rows),
        "melhores_mercados_observados": sorted([
            {"mercado": r["mercado"], "roi_observado": r["production"].get("roi_real", 0), "apostas": r["production"].get("total_apostas_real", 0)}
            for r in mercado_rows
        ], key=lambda x: x["roi_observado"], reverse=True)[:5],
        "piores_mercados_observados": sorted([
            {"mercado": r["mercado"], "roi_observado": r["production"].get("roi_real", 0), "apostas": r["production"].get("total_apostas_real", 0)}
            for r in mercado_rows
        ], key=lambda x: x["roi_observado"])[:5],
        "mercados_suspeitos_overfitting": [
            r["mercado"] for r in mercado_rows
            if "SUSPEITA_DE_SOBREAJUSTE" in r["status_historico"]
        ],
        "mercados_sem_amostra_suficiente": [
            r["mercado"] for r in mercado_rows
            if r["status_historico"] == "AMOSTRA_HISTORICA_INSUFICIENTE"
        ],
        "acao_recomendadas": {r["mercado"]: r["acao_recomendada"] for r in mercado_rows},
        "safety_note": "Auditoria analítica; não altera predict, guards, banca ou desbloqueios.",
    }

    audit_payload = {
        "timestamp": now_iso(),
        "status": "OK",
        "scope": "simulation_analysis_only" if data_scope == "SIMULATED" else "historical_analysis",
        "data_scope": data_scope,
        "origins": origins,
        "min_real_sample": MIN_REAL_SAMPLE,
        "sources": {
            "history": str(hist_path.relative_to(ROOT_DIR)) if hist_path else None,
            "backtest_candidates": [str(p.relative_to(ROOT_DIR)) for p in BACKTEST_PATHS if p.exists()],
            "performance_by_mercado": str(PERFORMANCE_BY_MARKET_PATH.relative_to(ROOT_DIR)) if PERFORMANCE_BY_MARKET_PATH.exists() else None,
            "mercado_governance": str(MARKET_GOVERNANCE_PATH.relative_to(ROOT_DIR)) if MARKET_GOVERNANCE_PATH.exists() else None,
        },
        "mercado_governance_status": governance.get("status") if isinstance(governance, dict) else None,
        "mercados": mercado_rows,
    }

    write_json(MARKET_AUDIT_PATH, audit_payload)
    write_json(REALITY_SUMMARY_PATH, summary)
    write_json(DC12_DEEP_DIVE_PATH, _dc12_deep_dive(settled))
    write_json(LEAGUE_AUDIT_PATH, {"timestamp": now_iso(), "status": "OK", "scope": "simulation_analysis_only", "ligas": liga_rows})
    write_json(ODD_RANGE_AUDIT_PATH, {"timestamp": now_iso(), "status": "OK", "scope": "simulation_analysis_only", "odd_ranges": odd_rows})
    write_json(CONFIDENCE_RANGE_AUDIT_PATH, {"timestamp": now_iso(), "status": "OK", "scope": "simulation_analysis_only", "faixa_confiancas": confianca_rows})
    return audit_payload


def _fmt_pct(x: Any) -> str:
    return f"{float(x or 0)*100:.2f}%"


def main() -> int:
    payload = run_audit()
    print("\nAuditoria Operacional de Mercados & Overfitting Detection")
    print(f"Status: {payload.get('status')}")
    print("Modo: análise/simulação — nenhum guard, predict, banca ou mercado foi alterado.")
    if payload.get("status") != "OK":
        print(payload.get("message", "Sem dados suficientes."))
        return 0

    mercados = payload.get("mercados", [])
    by_roi = sorted(mercados, key=lambda r: r.get("production", {}).get("roi_real", 0), reverse=True)
    print(f"\nEscopo do histórico: {payload.get('data_scope', 'NAO_INFORMADO')}")
    print("Top 5 mercados observados:")
    for r in by_roi[:5]:
        print(f"  {r['mercado']}: ROI observado {_fmt_pct(r['production'].get('roi_real'))} | apostas {r['production'].get('total_apostas_real')} | status {r['status_historico']}")
    print("\nBottom 5 mercados observados:")
    for r in by_roi[-5:]:
        print(f"  {r['mercado']}: ROI observado {_fmt_pct(r['production'].get('roi_real'))} | apostas {r['production'].get('total_apostas_real')} | status {r['status_historico']}")
    suspects = [r for r in mercados if "SUSPEITA_DE_SOBREAJUSTE" in r.get("status_historico", "")]
    print("\nOverfit suspects:")
    if suspects:
        for r in suspects:
            print(f"  {r['mercado']}: BT {_fmt_pct(r['backtest'].get('roi_backtest'))} vs observado {_fmt_pct(r['production'].get('roi_real'))} | ação analítica {r['acao_recomendada']}")
    else:
        print("  nenhum")
    print("\nArquivos gerados:")
    for path in [MARKET_AUDIT_PATH, REALITY_SUMMARY_PATH, DC12_DEEP_DIVE_PATH, LEAGUE_AUDIT_PATH, ODD_RANGE_AUDIT_PATH, CONFIDENCE_RANGE_AUDIT_PATH]:
        print(f"  - {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
