#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoria operacional segura do Football Lab.

Gera relatórios de qualidade e performance histórica local. Não recomenda
entradas, não altera banca, não muda histórico e não executa aposta.
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
PROJECT_ROOT = BASE_DIR.parent
BANCA_DIR = BASE_DIR / "banca"
REPORTS_DIR = BASE_DIR / "reports"
ALERTS_DIR = BASE_DIR / "alerts"
HISTORICO_FILE = BANCA_DIR / "historico_apostas.csv"
ESTADO_FILE = BANCA_DIR / "banca_estado.json"
DASHBOARD_FILE = REPORTS_DIR / "painel_performance.html"

for _path in (PROJECT_ROOT, BASE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from utils.data_quality import write_quality_report  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def load_history() -> pd.DataFrame:
    if not HISTORICO_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORICO_FILE)
    for col in ["odd", "valor_apostado", "lucro", "banca_apos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["data", "liga", "mercado", "resultado"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    return df


def history_scope(df: pd.DataFrame) -> tuple[str, list[str]]:
    if df.empty or "origem" not in df.columns:
        return "NAO_INFORMADO", []
    origins = sorted(
        df["origem"].fillna("nao_informada").astype(str).str.strip().str.lower().unique().tolist()
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


def summarize(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    if df.empty or by not in df.columns:
        return []
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    if final.empty:
        return []
    grouped = final.groupby(by, dropna=False)
    out = grouped.agg(
        apostas=("resultado", "size"),
        ganhos=("resultado", lambda s: int((s.str.lower() == "ganhou").sum())),
        perdas=("resultado", lambda s: int((s.str.lower() == "perdeu").sum())),
        stake=("valor_apostado", "sum"),
        lucro_liquido=("lucro", "sum"),
        odd_media=("odd", "mean"),
    ).reset_index()
    out["win_rate"] = out.apply(lambda r: (r["ganhos"] / r["apostas"]) if r["apostas"] else 0.0, axis=1)
    out["roi"] = out.apply(lambda r: (r["lucro_liquido"] / r["stake"]) if r["stake"] > 0 else 0.0, axis=1)
    out = out.sort_values(["roi", "lucro_liquido"], ascending=[False, False])
    records = []
    for _, r in out.iterrows():
        group = final[final[by].astype(str) == str(r[by])]
        data_scope, origins = history_scope(group)
        records.append({
            by: str(r[by]),
            "apostas": int(r["apostas"]),
            "ganhos": int(r["ganhos"]),
            "perdas": int(r["perdas"]),
            "win_rate": round(float(r["win_rate"]), 6),
            "roi": round(float(r["roi"]), 6),
            "lucro_liquido": round(float(r["lucro_liquido"]), 2),
            "stake": round(float(r["stake"]), 2),
            "odd_media": round(float(r["odd_media"]), 4),
            "data_scope": data_scope,
            "origins": origins,
        })
    return records


def build_degradation_alert(perf_mercado: list[dict[str, Any]]) -> dict[str, Any]:
    degraded = [
        m for m in perf_mercado
        if int(m.get("apostas", 0)) >= 100 and float(m.get("roi", 0.0)) < -0.03
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ALERT" if degraded else "OK",
        "rule": "ROI < -3% com mínimo de 100 registros finalizados",
        "degraded_mercados": degraded,
    }


def money(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v:+.2%}"


def rows_html(records: list[dict[str, Any]], key: str) -> str:
    if not records:
        return "<p>Sem dados disponíveis.</p>"
    rows = []
    for r in records:
        roi = float(r.get("roi", 0.0))
        lucro = float(r.get("lucro_liquido", 0.0))
        rows.append(
            "<tr>"
            f"<td>{str(r.get(key, ''))}</td>"
            f"<td>{int(r.get('apostas', 0))}</td>"
            f"<td>{pct(float(r.get('win_rate', 0.0)))}</td>"
            f"<td class='{'good' if roi >= 0 else 'bad'}'>{pct(roi)}</td>"
            f"<td class='{'good' if lucro >= 0 else 'bad'}'>{money(lucro)}</td>"
            f"<td>{float(r.get('odd_media', 0.0)):.2f}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Grupo</th><th>Registros</th><th>Win rate</th><th>ROI</th><th>P&L</th><th>Odd média</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def write_dashboard(estado: dict[str, Any], perf_mercado: list[dict[str, Any]], perf_liga: list[dict[str, Any]], alert: dict[str, Any], quality: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    banca_atual = float(estado.get("banca_atual", 0) or 0)
    banca_pico = float(estado.get("banca_pico", banca_atual) or banca_atual or 0)
    drawdown = ((banca_pico - banca_atual) / banca_pico) if banca_pico > 0 else 0.0
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Football Lab — Painel de Performance</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f6f7fb;color:#162033}} .card{{background:white;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 10px #0001}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}} .metric{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 10px #0001}} .metric b{{display:block;font-size:22px;margin-top:6px}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{padding:10px;border-bottom:1px solid #e8eaf0;text-align:left}} th{{background:#eef1f7}} .good{{color:#0a7a32;font-weight:700}} .bad{{color:#b00020;font-weight:700}} .warn{{color:#a15c00;font-weight:700}} code{{background:#eef1f7;padding:2px 6px;border-radius:6px}}
</style></head><body>
<h1>Football Lab — Painel de Performance</h1>
<p>Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Relatório analítico local, sem recomendação operacional.</p>
<div class="grid">
<div class="metric">Banca atual<b>{money(banca_atual)}</b></div>
<div class="metric">Drawdown<b class="{'bad' if drawdown > 0.2 else 'warn' if drawdown > 0.1 else 'good'}">{pct(-drawdown)}</b></div>
<div class="metric">Qualidade dos dados<b>{quality.get('status','SEM_DADOS')}</b></div>
<div class="metric">Alertas de mercado<b class="{'bad' if alert.get('status') == 'ALERT' else 'good'}">{alert.get('status','OK')}</b></div>
</div>
<div class="card"><h2>Alertas ativos</h2><pre>{json.dumps(alert, ensure_ascii=False, indent=2)}</pre></div>
<div class="card"><h2>Performance por mercado</h2>{rows_html(perf_mercado, 'mercado')}</div>
<div class="card"><h2>Performance por liga</h2>{rows_html(perf_liga[:30], 'liga')}</div>
<div class="card"><h2>Qualidade dos dados</h2><pre>{json.dumps(quality, ensure_ascii=False, indent=2)}</pre></div>
</body></html>"""
    DASHBOARD_FILE.write_text(html, encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_history()
    estado = load_json(ESTADO_FILE)
    data_scope, origins = history_scope(df)
    quality = write_quality_report(REPORTS_DIR / "relatorio_qualidade_dados.json")
    perf_mercado = summarize(df, "mercado")
    perf_liga = summarize(df, "liga")
    alert = build_degradation_alert(perf_mercado)

    (REPORTS_DIR / "performance_por_mercado.json").write_text(json.dumps(perf_mercado, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "performance_por_liga.json").write_text(json.dumps(perf_liga, ensure_ascii=False, indent=2), encoding="utf-8")
    (ALERTS_DIR / "mercado_degradation.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ALERTA" if quality.get("status") != "OK" or alert.get("status") == "ALERT" else "OK",
        "data_scope": data_scope,
        "origins": origins,
        "data_quality": quality,
        "mercado_degradation": alert,
        "files_generated": [
            "04_ml/reports/relatorio_qualidade_dados.json",
            "04_ml/reports/performance_por_mercado.json",
            "04_ml/reports/performance_por_liga.json",
            "04_ml/alerts/mercado_degradation.json",
            "04_ml/reports/painel_performance.html",
        ],
    }
    (REPORTS_DIR / "auditoria_operacional.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dashboard(estado, perf_mercado, perf_liga, alert, quality)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
