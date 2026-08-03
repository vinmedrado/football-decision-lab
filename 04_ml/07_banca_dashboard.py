#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera dashboard HTML local da banca.

Saída padrão: 04_ml/reports/painel_banca.html
Usa somente arquivos locais da banca, sem chamadas externas.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
BANCA_DIR = BASE_DIR / "banca"
REPORTS_DIR = BASE_DIR / "reports"
HISTORICO_FILE = BANCA_DIR / "historico_apostas.csv"
ESTADO_FILE = BANCA_DIR / "banca_estado.json"
OUTPUT_FILE = REPORTS_DIR / "painel_banca.html"
DRIFT_SUMMARY_FILE = REPORTS_DIR / "resumo_deriva.json"
GUARD_STATE_FILE = REPORTS_DIR / "estado_guard_calibracao.json"
ALERTA_DRAWDOWN = 0.20


def money(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v:+.2%}"


def load_data() -> tuple[dict, pd.DataFrame]:
    if ESTADO_FILE.exists():
        estado = json.loads(ESTADO_FILE.read_text(encoding="utf-8"))
    else:
        estado = {
            "banca_inicial": 300.0,
            "banca_atual": 300.0,
            "banca_pico": 300.0,
            "lucro_total": 0.0,
            "roi_total": 0.0,
            "total_apostas": 0,
            "total_ganhos": 0,
            "total_perdas": 0,
        }

    if HISTORICO_FILE.exists():
        df = pd.read_csv(HISTORICO_FILE)
    else:
        df = pd.DataFrame(columns=[
            "data", "jogo", "liga", "mercado", "event", "odd",
            "valor_apostado", "resultado", "lucro", "banca_apos",
        ])

    for col in ["odd", "valor_apostado", "lucro", "banca_apos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["data", "liga", "mercado", "resultado"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    return estado, df


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    if final.empty or by not in final.columns:
        return pd.DataFrame(columns=[by, "apostas", "stake", "lucro", "roi", "winrate", "odd_media"])
    grouped = final.groupby(by, dropna=False)
    out = grouped.agg(
        apostas=("resultado", "size"),
        stake=("valor_apostado", "sum"),
        lucro=("lucro", "sum"),
        odd_media=("odd", "mean"),
    ).reset_index()
    wins = grouped["resultado"].apply(lambda s: (s.str.lower() == "ganhou").mean()).reset_index(name="winrate")
    out = out.merge(wins, on=by, how="left")
    out["roi"] = out.apply(lambda r: (r["lucro"] / r["stake"]) if r["stake"] > 0 else 0.0, axis=1)
    return out.sort_values(["lucro", "roi"], ascending=[False, False])


def html_table(df: pd.DataFrame, label_col: str) -> str:
    if df.empty:
        return "<p>Sem dados finalizados ainda.</p>"
    rows = []
    for _, r in df.iterrows():
        roi_class = "good" if r["roi"] >= 0 else "bad"
        lucro_class = "good" if r["lucro"] >= 0 else "bad"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r[label_col]))}</td>"
            f"<td>{int(r['apostas'])}</td>"
            f"<td>{money(float(r['stake']))}</td>"
            f"<td class='{lucro_class}'>{money(float(r['lucro']))}</td>"
            f"<td class='{roi_class}'>{pct(float(r['roi']))}</td>"
            f"<td>{float(r['winrate']):.1%}</td>"
            f"<td>{float(r['odd_media']):.2f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        f"<th>{html.escape(label_col)}</th><th>Apostas</th><th>Stake</th><th>Lucro</th>"
        "<th>ROI</th><th>Winrate</th><th>Odd média</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def build_equity_table(df: pd.DataFrame) -> str:
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    if final.empty:
        return "<p>Sem curva de banca ainda.</p>"
    daily = final.groupby("data", dropna=False).agg(
        apostas=("resultado", "size"),
        lucro=("lucro", "sum"),
        banca=("banca_apos", "last"),
    ).reset_index().tail(30)
    rows = []
    for _, r in daily.iterrows():
        cls = "good" if r["lucro"] >= 0 else "bad"
        rows.append(
            f"<tr><td>{html.escape(str(r['data']))}</td><td>{int(r['apostas'])}</td>"
            f"<td class='{cls}'>{money(float(r['lucro']))}</td><td>{money(float(r['banca']))}</td></tr>"
        )
    return "<table><thead><tr><th>Data</th><th>Apostas</th><th>P&L</th><th>Banca após</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    estado, df = load_data()
    drift_summary = load_json_safe(DRIFT_SUMMARY_FILE)
    guard_state = load_json_safe(GUARD_STATE_FILE)

    banca_atual = float(estado.get("banca_atual", 0) or 0)
    banca_inicial = float(estado.get("banca_inicial", banca_atual) or banca_atual or 0)
    banca_pico = float(estado.get("banca_pico", banca_atual) or banca_atual or 0)
    drawdown = ((banca_pico - banca_atual) / banca_pico) if banca_pico > 0 else 0.0
    variacao = ((banca_atual / banca_inicial) - 1) if banca_inicial > 0 else 0.0
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])]
    pendentes = df[df["resultado"].str.lower().eq("pendente")]
    drift_overall = drift_summary.get("overall") or {}
    guard_status = str(guard_state.get("status", "SEM_GUARD"))
    guard_blocked = bool(guard_state.get("blocked", False))

    alert = ""
    if drawdown >= ALERTA_DRAWDOWN:
        alert = f"<div class='alert'>ALERTA: drawdown atual de {drawdown:.1%}, acima do limite de {ALERTA_DRAWDOWN:.0%}.</div>"

    html_doc = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Football Lab — Painel da Banca</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1f2937; }}
.header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }}
.cardgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
.card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 5px rgba(0,0,0,.08); }}
.card small {{ color: #6b7280; display:block; margin-bottom: 6px; }}
.card strong {{ font-size: 22px; }}
.good {{ color: #047857; font-weight: 700; }}
.bad {{ color: #b91c1c; font-weight: 700; }}
.alert {{ background:#fee2e2; color:#991b1b; border:1px solid #fecaca; padding:14px; border-radius:10px; margin:16px 0; font-weight:700; }}
section {{ background: white; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 5px rgba(0,0,0,.08); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; }}
th {{ background: #f3f4f6; }}
.footer {{ color:#6b7280; font-size:12px; margin-top:20px; }}
</style>
</head>
<body>
<div class="header"><h1>Football Lab — Painel da Banca</h1><span>Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></div>
{alert}
<div class="cardgrid">
  <div class="card"><small>Banca atual</small><strong>{money(banca_atual)}</strong></div>
  <div class="card"><small>Variação total</small><strong class="{'good' if variacao >= 0 else 'bad'}">{pct(variacao)}</strong></div>
  <div class="card"><small>Drawdown atual</small><strong class="{'bad' if drawdown >= ALERTA_DRAWDOWN else ''}">{drawdown:.2%}</strong></div>
  <div class="card"><small>Apostas finalizadas</small><strong>{len(final)}</strong></div>
  <div class="card"><small>Apostas pendentes</small><strong>{len(pendentes)}</strong></div>
  <div class="card"><small>Calibração</small><strong class="{'bad' if str(drift_overall.get('alert_level', '')).upper() == 'CRITICO' else ''}">{html.escape(str(drift_overall.get('alert_level', 'N/A')))}</strong></div>
  <div class="card"><small>Guard operacional</small><strong class="{'bad' if guard_blocked else 'good'}">{html.escape(guard_status)}</strong></div>
</div>
<section><h2>Calibração / Drift</h2><p><strong>Registros avaliados:</strong> {html.escape(str(drift_summary.get("evaluated_records", "N/A")))} | <strong>Erro calibração:</strong> {html.escape(str(drift_overall.get("calibration_error", "N/A")))} | <strong>Brier:</strong> {html.escape(str(drift_overall.get("brier_score", "N/A")))}</p><p><strong>Motivos do guard:</strong> {html.escape("; ".join(map(str, guard_state.get("motivos", []))) or "Nenhum bloqueio")}</p></section>
<section><h2>ROI por mercado</h2>{html_table(summarize(df, 'mercado'), 'mercado')}</section>
<section><h2>ROI por liga</h2>{html_table(summarize(df, 'liga'), 'liga')}</section>
<section><h2>P&L por dia — últimos 30 dias com aposta finalizada</h2>{build_equity_table(df)}</section>
<div class="footer">Fonte: 04_ml/banca/historico_apostas.csv e 04_ml/banca/banca_estado.json</div>
</body>
</html>"""
    OUTPUT_FILE.write_text(html_doc, encoding="utf-8")
    print(f"✅ Painel gerado: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
