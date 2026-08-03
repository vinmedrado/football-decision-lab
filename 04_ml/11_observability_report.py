#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 5 — Observabilidade histórica do Football Lab.

Gera relatórios locais de evolução diária, curva de banca, drawdown e ROI acumulado.
Não executa apostas, não recomenda entradas e não altera histórico/banca.
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
BANCA_DIR = BASE_DIR / "banca"
REPORTS_DIR = BASE_DIR / "reports"
HIST_FILE = BANCA_DIR / "historico_apostas.csv"
ESTADO_FILE = BANCA_DIR / "banca_estado.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def load_history() -> pd.DataFrame:
    if not HIST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_FILE)
    for col in ["odd", "valor_apostado", "lucro", "banca_apos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["data", "mercado", "liga", "resultado"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    return df


def money(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v:+.2%}"


def build_daily(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    if final.empty:
        return []
    final["data_sort"] = pd.to_datetime(final["data"], errors="coerce", dayfirst=False)
    daily = final.groupby("data", dropna=False).agg(
        apostas=("resultado", "size"),
        ganhos=("resultado", lambda s: int((s.str.lower() == "ganhou").sum())),
        perdas=("resultado", lambda s: int((s.str.lower() == "perdeu").sum())),
        stake=("valor_apostado", "sum"),
        lucro=("lucro", "sum"),
        banca_fechamento=("banca_apos", "last"),
        data_sort=("data_sort", "min"),
    ).reset_index().sort_values("data_sort")
    daily["lucro_acumulado"] = daily["lucro"].cumsum()
    daily["stake_acumulada"] = daily["stake"].cumsum()
    daily["roi_dia"] = daily.apply(lambda r: (r["lucro"] / r["stake"]) if r["stake"] and r["stake"] > 0 else 0.0, axis=1)
    daily["roi_acumulado"] = daily.apply(lambda r: (r["lucro_acumulado"] / r["stake_acumulada"]) if r["stake_acumulada"] and r["stake_acumulada"] > 0 else 0.0, axis=1)
    peaks = daily["banca_fechamento"].cummax()
    daily["drawdown"] = daily.apply(lambda r: ((peaks.loc[r.name] - r["banca_fechamento"]) / peaks.loc[r.name]) if peaks.loc[r.name] > 0 else 0.0, axis=1)
    out = []
    for _, r in daily.iterrows():
        out.append({
            "data": str(r["data"]),
            "apostas": int(r["apostas"]),
            "ganhos": int(r["ganhos"]),
            "perdas": int(r["perdas"]),
            "stake": round(float(r["stake"] or 0), 2),
            "lucro": round(float(r["lucro"] or 0), 2),
            "banca_fechamento": round(float(r["banca_fechamento"] or 0), 2),
            "lucro_acumulado": round(float(r["lucro_acumulado"] or 0), 2),
            "roi_dia": round(float(r["roi_dia"] or 0), 6),
            "roi_acumulado": round(float(r["roi_acumulado"] or 0), 6),
            "drawdown": round(float(r["drawdown"] or 0), 6),
        })
    return out


def build_summary(estado: dict[str, Any], daily: list[dict[str, Any]], df: pd.DataFrame) -> dict[str, Any]:
    banca_atual = float(estado.get("banca_atual", 0) or 0)
    banca_inicial = float(estado.get("banca_inicial", banca_atual) or banca_atual or 0)
    banca_pico = float(estado.get("banca_pico", banca_atual) or banca_atual or 0)
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])] if not df.empty else pd.DataFrame()
    stake_total = float(final["valor_apostado"].sum()) if not final.empty and "valor_apostado" in final.columns else 0.0
    lucro_total = float(final["lucro"].sum()) if not final.empty and "lucro" in final.columns else 0.0
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "historical_analytics_only",
        "banca_atual": round(banca_atual, 2),
        "banca_inicial": round(banca_inicial, 2),
        "banca_pico": round(banca_pico, 2),
        "variacao_banca": round((banca_atual / banca_inicial - 1) if banca_inicial > 0 else 0.0, 6),
        "drawdown_atual": round(((banca_pico - banca_atual) / banca_pico) if banca_pico > 0 else 0.0, 6),
        "apostas_finalizadas": int(len(final)),
        "stake_total": round(stake_total, 2),
        "lucro_total": round(lucro_total, 2),
        "roi_total": round((lucro_total / stake_total) if stake_total > 0 else 0.0, 6),
        "dias_com_registro": len(daily),
        "maior_drawdown_historico": max([float(d.get("drawdown", 0)) for d in daily], default=0.0),
    }


def write_html(summary: dict[str, Any], daily: list[dict[str, Any]]) -> None:
    rows = []
    for d in daily[-90:]:
        cls_l = "good" if d["lucro"] >= 0 else "bad"
        cls_dd = "bad" if d["drawdown"] >= 0.2 else "warn" if d["drawdown"] >= 0.1 else ""
        rows.append(
            f"<tr><td>{d['data']}</td><td>{d['apostas']}</td><td class='{cls_l}'>{money(d['lucro'])}</td>"
            f"<td>{money(d['banca_fechamento'])}</td><td>{pct(d['roi_acumulado'])}</td><td class='{cls_dd}'>{d['drawdown']:.2%}</td></tr>"
        )
    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Football Lab — Observabilidade</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f6f7fb;color:#152033}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card,section{{background:white;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 2px 10px #0001}}.card b{{display:block;font-size:22px;margin-top:6px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e8eaf0;text-align:left}}th{{background:#eef1f7}}.good{{color:#087a35;font-weight:700}}.bad{{color:#b00020;font-weight:700}}.warn{{color:#995b00;font-weight:700}}</style></head><body>
<h1>Football Lab — Observabilidade Histórica</h1><p>Gerado em {summary['generated_at']}. Uso analítico local.</p>
<div class="grid"><div class="card">Banca atual<b>{money(summary['banca_atual'])}</b></div><div class="card">ROI total<b class="{'good' if summary['roi_total'] >= 0 else 'bad'}">{pct(summary['roi_total'])}</b></div><div class="card">Drawdown atual<b>{summary['drawdown_atual']:.2%}</b></div><div class="card">Maior drawdown<b>{summary['maior_drawdown_historico']:.2%}</b></div><div class="card">Apostas finalizadas<b>{summary['apostas_finalizadas']}</b></div></div>
<section><h2>Evolução diária</h2><table><thead><tr><th>Data</th><th>Registros</th><th>P&L</th><th>Banca</th><th>ROI acumulado</th><th>Drawdown</th></tr></thead><tbody>{''.join(rows) if rows else '<tr><td colspan="6">Sem dados.</td></tr>'}</tbody></table></section>
</body></html>"""
    (REPORTS_DIR / "painel_observabilidade.html").write_text(html, encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_history()
    estado = read_json(ESTADO_FILE)
    daily = build_daily(df)
    summary = build_summary(estado, daily, df)
    payload = {"summary": summary, "daily": daily}
    (REPORTS_DIR / "observabilidade_banca_diaria.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(summary, daily)
    print(json.dumps({"status": "OK", "files": ["04_ml/reports/observabilidade_banca_diaria.json", "04_ml/reports/painel_observabilidade.html"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
