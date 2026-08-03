#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 19 — Governança de Ligas Engine.

Analytical/simulation-only liga governance layer.

This script identifies which ligas support or degrade real production performance
without changing ML, predict, calibration, bankroll, settlement, guards, or any
real-betting/recommendation flags.

Generated artifacts:
- reports/relatorio_governanca_ligas.json
- reports/classificacao_ligas.json
- reports/ciclo_vida_ligas.json
- reports/analise_detalhada_piores_ligas.json
- reports/relatorio_penalidade_liga_mercado.json
- reports/liga_painel_governanca.html
"""
from __future__ import annotations

import csv
import html
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import REPORTS_DIR, ROOT_DIR, now_iso, read_json, write_json  # noqa: E402
from core.result_normalizer import normalize_result  # noqa: E402

HISTORY_PATHS = [
    ROOT_DIR / "04_ml" / "banca" / "historico_apostas.csv",
    ROOT_DIR / "04_ml" / "historico_apostas.csv",
]
MARKET_REALITY_AUDIT_PATH = REPORTS_DIR / "auditoria_operacional_mercados.json"
MARKET_LIFECYCLE_STATE_PATH = REPORTS_DIR / "ciclo_vida_mercados.json"
PERFORMANCE_BY_MARKET_PATH = REPORTS_DIR / "performance_por_mercado.json"

LEAGUE_GOVERNANCE_REPORT_PATH = REPORTS_DIR / "relatorio_governanca_ligas.json"
LEAGUE_RANKING_PATH = REPORTS_DIR / "classificacao_ligas.json"
LEAGUE_LIFECYCLE_STATE_PATH = REPORTS_DIR / "ciclo_vida_ligas.json"
LEAGUE_DEEP_DIVE_PATH = REPORTS_DIR / "analise_detalhada_piores_ligas.json"
LEAGUE_MARKET_PENALTY_PATH = REPORTS_DIR / "relatorio_penalidade_liga_mercado.json"
LEAGUE_DASHBOARD_PATH = REPORTS_DIR / "liga_painel_governanca.html"
GOVERNANCE_DASHBOARD_PATH = REPORTS_DIR / "painel_governanca.html"

MIN_LEAGUE_SAMPLE = 100
ROI_ALERTA_ABS = 0.02

SAFE_FLAGS = {
    "modo_simulacao": True,
    "apostas_reais_habilitadas": False,
    "recomendacoes_habilitadas": False,
    "modo_seguro": True,
    "scope": "simulation_analysis_only",
}

ODD_BUCKETS = [
    (1.00, 1.30, "1.00-1.30"),
    (1.31, 1.60, "1.31-1.60"),
    (1.61, 2.00, "1.61-2.00"),
    (2.01, 2.50, "2.01-2.50"),
    (2.51, float("inf"), "2.51+"),
]
CONF_BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip().replace("R$", "").replace("%", "")
            if "," in text and "." in text:
                text = text.replace(".", "").replace(",", ".")
            elif "," in text:
                text = text.replace(",", ".")
            value = text
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _find_history_path() -> Path | None:
    for path in HISTORY_PATHS:
        if path.exists():
            return path
    return None


def _read_history_rows() -> Tuple[List[Dict[str, Any]], str]:
    path = _find_history_path()
    if path is None:
        return [], "NOT_FOUND"
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                return list(csv.DictReader(fh)), str(path.relative_to(ROOT_DIR))
        except Exception:
            continue
    return [], str(path.relative_to(ROOT_DIR))


def _bucket(value: float, buckets: Iterable[Tuple[float, float, str]]) -> str:
    for low, high, label in buckets:
        if low <= value <= high:
            return label
    return "unknown"


def _profit_from_row(row: Dict[str, Any], normalized_result: str) -> Tuple[float, float]:
    stake = _as_float(row.get("valor_apostado") or row.get("stake") or row.get("valor"), 1.0)
    profit_raw = row.get("lucro") or row.get("profit") or row.get("pnl")
    if profit_raw not in (None, ""):
        return _as_float(profit_raw), max(stake, 0.0)
    odd = _as_float(row.get("odd") or row.get("odds") or row.get("cotacao"), 0.0)
    if normalized_result == "WIN":
        return stake * max(odd - 1.0, 0.0), max(stake, 0.0)
    if normalized_result == "LOSS":
        return -stake, max(stake, 0.0)
    return 0.0, max(stake, 0.0)


def _empty_agg() -> Dict[str, Any]:
    return {
        "bets": 0,
        "wins": 0,
        "losses": 0,
        "stake": 0.0,
        "profit": 0.0,
        "odds_sum": 0.0,
        "confianca_sum": 0.0,
        "mercados": defaultdict(int),
        "monthly": defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "profit": 0.0}),
        "odd_ranges": defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "profit": 0.0}),
        "faixa_confiancas": defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "profit": 0.0, "predicted_sum": 0.0}),
        "equity": [],
    }


def _max_drawdown(profits: List[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for profit in profits:
        cumulative += profit
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_dd = max(max_dd, drawdown)
    total_stake = sum(abs(p) for p in profits) or 1.0
    return max_dd / total_stake


def _status(roi: float, bets: int) -> str:
    if bets < MIN_LEAGUE_SAMPLE:
        if roi > 0:
            return "LEAGUE_OBSERVACAO"
        return "LEAGUE_DESCONHECIDA"
    if roi > 0:
        return "LEAGUE_ATIVA"
    if roi < 0:
        return "LEAGUE_FAIL"
    if abs(roi) <= ROI_ALERTA_ABS:
        return "LEAGUE_ALERTA"
    return "LEAGUE_DESCONHECIDA"


def _stability_score(monthly_rows: Dict[str, Dict[str, Any]]) -> float:
    rois: List[float] = []
    positive = 0
    for item in monthly_rows.values():
        stake = _as_float(item.get("stake"))
        roi = _as_float(item.get("profit")) / stake if stake else 0.0
        rois.append(roi)
        if roi > 0:
            positive += 1
    if not rois:
        return 0.0
    mean = sum(rois) / len(rois)
    variance = sum((x - mean) ** 2 for x in rois) / len(rois)
    stdev = math.sqrt(variance)
    positive_ratio = positive / len(rois)
    return round(max(0.0, min(100.0, positive_ratio * 70.0 + (1.0 - min(stdev, 1.0)) * 30.0)), 2)


def _serialize_range_stats(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label, item in sorted(data.items()):
        bets = int(item.get("bets") or 0)
        stake = _as_float(item.get("stake"))
        profit = _as_float(item.get("profit"))
        wins = int(item.get("wins") or 0)
        predicted_sum = _as_float(item.get("predicted_sum"))
        row = {
            "range": label,
            "bets": bets,
            "roi": round(profit / stake, 6) if stake else 0.0,
            "profit": round(profit, 2),
            "winrate": round(wins / bets, 6) if bets else 0.0,
        }
        if predicted_sum:
            predicted_mean = predicted_sum / bets if bets else 0.0
            actual_rate = wins / bets if bets else 0.0
            row["predicted_mean"] = round(predicted_mean, 6)
            row["actual_rate"] = round(actual_rate, 6)
            row["calibration_gap"] = round(abs(predicted_mean - actual_rate), 6)
        rows.append(row)
    return rows


def _build_liga_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggs: Dict[str, Dict[str, Any]] = defaultdict(_empty_agg)

    for row in rows:
        liga = str(row.get("liga") or row.get("liga") or row.get("Liga") or "").strip() or "DESCONHECIDA_LEAGUE"
        mercado = str(row.get("mercado") or row.get("mercado") or row.get("event") or "").strip() or "DESCONHECIDA_MARKET"
        normalized = normalize_result(row.get("resultado") or row.get("result") or row.get("status"))
        if normalized not in {"WIN", "LOSS"}:
            continue
        profit, stake = _profit_from_row(row, normalized)
        odd = _as_float(row.get("odd") or row.get("odds") or row.get("cotacao"), 0.0)
        confianca = _as_float(row.get("probabilidade_calibrada") or row.get("prob_calibrada") or row.get("confianca") or row.get("prob_modelo"), 0.0)
        if confianca > 1.0:
            confianca /= 100.0
        month = str(row.get("data") or row.get("date") or "")[:7] or "unknown"

        agg = aggs[liga]
        agg["bets"] += 1
        agg["wins"] += 1 if normalized == "WIN" else 0
        agg["losses"] += 1 if normalized == "LOSS" else 0
        agg["stake"] += stake
        agg["profit"] += profit
        agg["odds_sum"] += odd
        agg["confianca_sum"] += confianca
        agg["mercados"][mercado] += 1
        agg["equity"].append(profit)

        m = agg["monthly"][month]
        m["bets"] += 1
        m["wins"] += 1 if normalized == "WIN" else 0
        m["stake"] += stake
        m["profit"] += profit

        odd_bucket = _bucket(odd, ODD_BUCKETS)
        o = agg["odd_ranges"][odd_bucket]
        o["bets"] += 1
        o["wins"] += 1 if normalized == "WIN" else 0
        o["stake"] += stake
        o["profit"] += profit

        conf_bucket = _bucket(confianca, CONF_BUCKETS)
        c = agg["faixa_confiancas"][conf_bucket]
        c["bets"] += 1
        c["wins"] += 1 if normalized == "WIN" else 0
        c["stake"] += stake
        c["profit"] += profit
        c["predicted_sum"] += confianca

    metrics: Dict[str, Dict[str, Any]] = {}
    for liga, agg in aggs.items():
        bets = int(agg["bets"])
        stake = _as_float(agg["stake"])
        profit = _as_float(agg["profit"])
        wins = int(agg["wins"])
        roi = profit / stake if stake else 0.0
        monthly = {
            month: {
                "bets": int(item["bets"]),
                "roi": round(_as_float(item["profit"]) / _as_float(item["stake"]), 6) if _as_float(item["stake"]) else 0.0,
                "profit": round(_as_float(item["profit"]), 2),
                "winrate": round(_as_float(item["wins"]) / _as_float(item["bets"]), 6) if _as_float(item["bets"]) else 0.0,
            }
            for month, item in sorted(agg["monthly"].items())
        }
        stability = _stability_score(agg["monthly"])
        metrics[liga] = {
            "liga": liga,
            "total_apostas": bets,
            "roi_real": round(roi, 6),
            "lucro_real": round(profit, 2),
            "stake_real": round(stake, 2),
            "taxa_acerto_real": round(wins / bets, 6) if bets else 0.0,
            "odd_media": round(_as_float(agg["odds_sum"]) / bets, 6) if bets else 0.0,
            "confianca_media": round(_as_float(agg["confianca_sum"]) / bets, 6) if bets else 0.0,
            "drawdown": round(_max_drawdown(agg["equity"]), 6),
            "stability_score": stability,
            "mercados_utilizados": dict(sorted(agg["mercados"].items(), key=lambda x: x[1], reverse=True)),
            "status": _status(roi, bets),
            "monthly_roi": monthly,
            "odd_distribution": _serialize_range_stats(agg["odd_ranges"]),
            "confianca_distribution": _serialize_range_stats(agg["faixa_confiancas"]),
        }
    return metrics


def _build_penalty_report(liga_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    lifecycle = read_json(MARKET_LIFECYCLE_STATE_PATH, {})
    mercado_states = lifecycle.get("mercados", {}) if isinstance(lifecycle, dict) else {}
    fail_ligas = {liga for liga, row in liga_metrics.items() if row.get("status") == "LEAGUE_FAIL"}
    mercado_usage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for liga, row in liga_metrics.items():
        for mercado, count in (row.get("mercados_utilizados") or {}).items():
            if liga in fail_ligas:
                mercado_usage[str(mercado)].append({
                    "liga": liga,
                    "liga_status": row.get("status"),
                    "liga_roi": row.get("roi_real"),
                    "bets_in_liga": count,
                    "penalty_motivo": "LEAGUE_FAIL_EXPOSURE",
                })

    mercados = {}
    for mercado, exposures in mercado_usage.items():
        mercados[mercado] = {
            "mercado": mercado,
            "current_status_ciclo_vida": (mercado_states.get(mercado, {}) or {}).get("status_ciclo_vida", "DESCONHECIDA") if isinstance(mercado_states, dict) else "DESCONHECIDA",
            "liga_fail_exposures": exposures,
            "diagnostic_penalty": min(30, len(exposures) * 5),
            "simulation_only": True,
            "applied_to_predict": False,
            "applied_to_banca": False,
        }
    return {
        "generated_at": now_iso(),
        **SAFE_FLAGS,
        "description": "Penalidade diagnóstica por exposição a ligas FAIL. Não altera predict, banca, lifecycle ou guards.",
        "fail_ligas": sorted(fail_ligas),
        "mercados": mercados,
    }


def _build_dashboard(report: Dict[str, Any], classificacao: Dict[str, Any], deep_dive: Dict[str, Any]) -> None:
    top = classificacao.get("top_by_profit", [])[:10]
    bottom = classificacao.get("bottom_by_profit", [])[:10]
    counts = report.get("summary", {}).get("status_counts", {})

    def table_rows(rows: List[Dict[str, Any]]) -> str:
        out = []
        for r in rows:
            out.append(
                "<tr>"
                f"<td>{html.escape(str(r.get('liga')))}</td>"
                f"<td>{r.get('status')}</td>"
                f"<td>{r.get('total_apostas')}</td>"
                f"<td>{r.get('roi_real')}</td>"
                f"<td>{r.get('lucro_real')}</td>"
                f"<td>{r.get('taxa_acerto_real')}</td>"
                f"<td>{r.get('stability_score')}</td>"
                "</tr>"
            )
        return "\n".join(out)

    dashboard = f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <title>Football Lab — Governança de Ligas</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #111827; color: #e5e7eb; }}
    h1, h2 {{ color: #f9fafb; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .card {{ background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 14px; min-width: 150px; }}
    .ok {{ color: #34d399; }} .warn {{ color: #fbbf24; }} .fail {{ color: #f87171; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1f2937; }}
    th, td {{ border: 1px solid #374151; padding: 8px; text-align: left; }}
    th {{ background: #111827; }}
    code {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <h1>Governança de Ligas</h1>
  <p>Modo: <b>simulação/auditoria</b>. Apostas reais: <b>false</b>. Recomendações: <b>false</b>.</p>
  <div class="cards">
    <div class="card"><b>Total ligas</b><br>{report.get('summary', {}).get('total_ligas', 0)}</div>
    <div class="card ok"><b>ATIVA</b><br>{counts.get('LEAGUE_ATIVA', 0)}</div>
    <div class="card warn"><b>OBSERVACAO</b><br>{counts.get('LEAGUE_OBSERVACAO', 0)}</div>
    <div class="card fail"><b>FAIL</b><br>{counts.get('LEAGUE_FAIL', 0)}</div>
    <div class="card"><b>DESCONHECIDA</b><br>{counts.get('LEAGUE_DESCONHECIDA', 0)}</div>
  </div>

  <h2>Top ligas por lucro</h2>
  <table><thead><tr><th>Liga</th><th>Status</th><th>Apostas</th><th>ROI</th><th>Lucro</th><th>Winrate</th><th>Stability</th></tr></thead><tbody>{table_rows(top)}</tbody></table>

  <h2>Bottom ligas por lucro</h2>
  <table><thead><tr><th>Liga</th><th>Status</th><th>Apostas</th><th>ROI</th><th>Lucro</th><th>Winrate</th><th>Stability</th></tr></thead><tbody>{table_rows(bottom)}</tbody></table>

  <h2>Deep Dive</h2>
  <p>Arquivo detalhado: <code>analise_detalhada_piores_ligas.json</code></p>
  <p>Gerado em: {html.escape(str(report.get('generated_at')))}</p>
</body>
</html>"""
    LEAGUE_DASHBOARD_PATH.write_text(dashboard, encoding="utf-8")

    # Keep existing executive dashboard intact; append a lightweight link/section once.
    if GOVERNANCE_DASHBOARD_PATH.exists():
        try:
            text = GOVERNANCE_DASHBOARD_PATH.read_text(encoding="utf-8", errors="ignore")
            marker = "<!-- FASE19_LEAGUE_GOVERNANCE_SECTION -->"
            section = f"""
{marker}
<section>
  <h2>LEAGUE GOVERNANCE</h2>
  <p>Status por liga gerado em modo simulação/auditoria.</p>
  <ul>
    <li>Total ligas: {report.get('summary', {}).get('total_ligas', 0)}</li>
    <li>ATIVA: {counts.get('LEAGUE_ATIVA', 0)}</li>
    <li>OBSERVACAO: {counts.get('LEAGUE_OBSERVACAO', 0)}</li>
    <li>FAIL: {counts.get('LEAGUE_FAIL', 0)}</li>
  </ul>
  <p>Painel específico: <code>liga_painel_governanca.html</code></p>
</section>
"""
            if marker not in text:
                if "</body>" in text:
                    text = text.replace("</body>", section + "\n</body>")
                else:
                    text += section
                GOVERNANCE_DASHBOARD_PATH.write_text(text, encoding="utf-8")
        except Exception:
            pass


def run_liga_governance() -> Dict[str, Any]:
    rows, source = _read_history_rows()
    now = now_iso()
    liga_metrics = _build_liga_metrics(rows)

    ligas = sorted(liga_metrics.values(), key=lambda x: (x.get("lucro_real", 0), x.get("roi_real", 0)), reverse=True)
    status_counts = defaultdict(int)
    for item in ligas:
        status_counts[item.get("status", "LEAGUE_DESCONHECIDA")] += 1

    report = {
        "generated_at": now,
        "phase": "FASE_19_LEAGUE_GOVERNANCE_ENGINE",
        **SAFE_FLAGS,
        "status": "OK" if ligas else "NO_DATA",
        "sources": {
            "historico_apostas": source,
            "mercado_reality_audit": str(MARKET_REALITY_AUDIT_PATH.relative_to(ROOT_DIR)) if MARKET_REALITY_AUDIT_PATH.exists() else "NOT_FOUND",
            "mercado_lifecycle_state": str(MARKET_LIFECYCLE_STATE_PATH.relative_to(ROOT_DIR)) if MARKET_LIFECYCLE_STATE_PATH.exists() else "NOT_FOUND",
            "performance_by_mercado": str(PERFORMANCE_BY_MARKET_PATH.relative_to(ROOT_DIR)) if PERFORMANCE_BY_MARKET_PATH.exists() else "NOT_FOUND",
        },
        "policy": {
            "min_liga_sample": MIN_LEAGUE_SAMPLE,
            "does_not_modify_predict": True,
            "does_not_modify_guards": True,
            "does_not_enable_betting": True,
        },
        "summary": {
            "total_ligas": len(ligas),
            "status_counts": dict(status_counts),
            "total_closed_bets": sum(int(x.get("total_apostas", 0)) for x in ligas),
            "total_profit": round(sum(_as_float(x.get("lucro_real")) for x in ligas), 2),
        },
        "ligas": ligas,
    }

    classificacao = {
        "generated_at": now,
        **SAFE_FLAGS,
        "top_by_roi": sorted(ligas, key=lambda x: (x.get("roi_real", 0), x.get("total_apostas", 0)), reverse=True)[:20],
        "bottom_by_roi": sorted(ligas, key=lambda x: (x.get("roi_real", 0), x.get("total_apostas", 0)))[:20],
        "top_by_profit": sorted(ligas, key=lambda x: x.get("lucro_real", 0), reverse=True)[:20],
        "bottom_by_profit": sorted(ligas, key=lambda x: x.get("lucro_real", 0))[:20],
        "top_by_stability": sorted(ligas, key=lambda x: (x.get("stability_score", 0), x.get("total_apostas", 0)), reverse=True)[:20],
    }

    lifecycle = {
        "generated_at": now,
        "source": "liga_governance_engine",
        **SAFE_FLAGS,
        "min_liga_sample": MIN_LEAGUE_SAMPLE,
        "ligas": {
            item["liga"]: {
                "status": item["status"],
                "roi_real": item["roi_real"],
                "lucro_real": item["lucro_real"],
                "total_apostas": item["total_apostas"],
                "taxa_acerto_real": item["taxa_acerto_real"],
                "drawdown": item["drawdown"],
                "stability_score": item["stability_score"],
                "mercados_used": item["mercados_utilizados"],
            }
            for item in ligas
        },
    }

    bottom5 = sorted(ligas, key=lambda x: (x.get("lucro_real", 0), x.get("roi_real", 0)))[:5]
    deep_dive = {
        "generated_at": now,
        **SAFE_FLAGS,
        "description": "Deep dive das 5 piores ligas por lucro/ROI real.",
        "ligas": {
            item["liga"]: {
                "status": item["status"],
                "total_apostas": item["total_apostas"],
                "roi_real": item["roi_real"],
                "lucro_real": item["lucro_real"],
                "drawdown": item["drawdown"],
                "mercados_utilizados": item["mercados_utilizados"],
                "monthly_roi": item["monthly_roi"],
                "odd_distribution": item["odd_distribution"],
                "confianca_distribution": item["confianca_distribution"],
            }
            for item in bottom5
        },
    }

    penalty_report = _build_penalty_report(liga_metrics)

    write_json(LEAGUE_GOVERNANCE_REPORT_PATH, report)
    write_json(LEAGUE_RANKING_PATH, classificacao)
    write_json(LEAGUE_LIFECYCLE_STATE_PATH, lifecycle)
    write_json(LEAGUE_DEEP_DIVE_PATH, deep_dive)
    write_json(LEAGUE_MARKET_PENALTY_PATH, penalty_report)
    _build_dashboard(report, classificacao, deep_dive)

    return report


def main() -> None:
    report = run_liga_governance()
    classificacao = read_json(LEAGUE_RANKING_PATH, {})
    top = classificacao.get("top_by_profit", [])[:10] if isinstance(classificacao, dict) else []
    bottom = classificacao.get("bottom_by_profit", [])[:10] if isinstance(classificacao, dict) else []
    counts = report.get("summary", {}).get("status_counts", {})

    print("\n" + "=" * 72)
    print("FASE 19 — Governança de Ligas Engine")
    print("=" * 72)
    print("Modo: SIMULAÇÃO/AUDITORIA")
    print("Apostas reais habilitadas: false")
    print("Recomendações habilitadas: false")
    print(f"Status: {report.get('status')}")
    print(f"Ligas avaliadas: {report.get('summary', {}).get('total_ligas', 0)}")
    print("\nResumo por status:")
    for key in ["LEAGUE_ATIVA", "LEAGUE_OBSERVACAO", "LEAGUE_FAIL", "LEAGUE_DESCONHECIDA", "LEAGUE_ALERTA"]:
        print(f"  - {key}: {counts.get(key, 0)}")

    print("\nTop 10 ligas por lucro:")
    for item in top:
        print(f"  - {item.get('liga')} | {item.get('status')} | ROI={item.get('roi_real')} | lucro={item.get('lucro_real')} | bets={item.get('total_apostas')}")

    print("\nBottom 10 ligas por lucro:")
    for item in bottom:
        print(f"  - {item.get('liga')} | {item.get('status')} | ROI={item.get('roi_real')} | lucro={item.get('lucro_real')} | bets={item.get('total_apostas')}")

    print("\nArquivos gerados:")
    for path in [
        LEAGUE_GOVERNANCE_REPORT_PATH,
        LEAGUE_RANKING_PATH,
        LEAGUE_LIFECYCLE_STATE_PATH,
        LEAGUE_DEEP_DIVE_PATH,
        LEAGUE_MARKET_PENALTY_PATH,
        LEAGUE_DASHBOARD_PATH,
    ]:
        print(f"  - {path.relative_to(ROOT_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
