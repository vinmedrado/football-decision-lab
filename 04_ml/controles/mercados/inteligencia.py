#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inteligência histórica por segmentos de mercado.

Rankings descritivos por mercado, liga, faixa de odd, dia e combinações.
Não recomenda apostas e não altera nenhum arquivo operacional da banca.
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

BASE_DIR = Path(__file__).resolve().parents[2]
BANCA_DIR = BASE_DIR / "banca"
REPORTS_DIR = BASE_DIR / "reports"
HIST_FILE = BANCA_DIR / "historico_apostas.csv"
MIN_SAMPLE_DEFAULT = 20


def load_history() -> pd.DataFrame:
    if not HIST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_FILE)
    for col in ["odd", "valor_apostado", "lucro"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["data", "mercado", "liga", "resultado"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    return df


def odd_bucket(odd: float) -> str:
    if odd <= 0:
        return "odd_invalida"
    if odd < 1.40:
        return "1.01-1.39"
    if odd < 1.60:
        return "1.40-1.59"
    if odd < 1.80:
        return "1.60-1.79"
    if odd < 2.10:
        return "1.80-2.09"
    if odd < 2.50:
        return "2.10-2.49"
    return "2.50+"


def summarize(df: pd.DataFrame, group_cols: list[str], min_sample: int = MIN_SAMPLE_DEFAULT) -> list[dict[str, Any]]:
    if df.empty:
        return []
    final = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    if final.empty or any(c not in final.columns for c in group_cols):
        return []
    grouped = final.groupby(group_cols, dropna=False)
    out = grouped.agg(
        registros=("resultado", "size"),
        ganhos=("resultado", lambda s: int((s.str.lower() == "ganhou").sum())),
        perdas=("resultado", lambda s: int((s.str.lower() == "perdeu").sum())),
        stake=("valor_apostado", "sum"),
        lucro=("lucro", "sum"),
        odd_media=("odd", "mean"),
        odd_min=("odd", "min"),
        odd_max=("odd", "max"),
    ).reset_index()
    out["win_rate"] = out.apply(lambda r: r["ganhos"] / r["registros"] if r["registros"] else 0.0, axis=1)
    out["roi"] = out.apply(lambda r: r["lucro"] / r["stake"] if r["stake"] > 0 else 0.0, axis=1)
    out = out[out["registros"] >= min_sample].copy()
    out = out.sort_values(["roi", "lucro", "registros"], ascending=[False, False, False])
    records = []
    for _, r in out.iterrows():
        rec = {c: str(r[c]) for c in group_cols}
        rec.update({
            "registros": int(r["registros"]),
            "ganhos": int(r["ganhos"]),
            "perdas": int(r["perdas"]),
            "win_rate": round(float(r["win_rate"]), 6),
            "roi": round(float(r["roi"]), 6),
            "lucro": round(float(r["lucro"]), 2),
            "stake": round(float(r["stake"]), 2),
            "odd_media": round(float(r["odd_media"]), 4),
            "odd_min": round(float(r["odd_min"]), 4),
            "odd_max": round(float(r["odd_max"]), 4),
        })
        records.append(rec)
    return records


def top_bottom(records: list[dict[str, Any]], n: int = 10) -> dict[str, Any]:
    return {
        "top": records[:n],
        "bottom": sorted(records, key=lambda r: (float(r.get("roi", 0)), float(r.get("lucro", 0))))[:n],
    }


def write_html(payload: dict[str, Any]) -> None:
    def table(records: list[dict[str, Any]], label_cols: list[str]) -> str:
        if not records:
            return "<p>Sem amostra suficiente.</p>"
        rows = []
        for r in records[:30]:
            label = " / ".join(str(r.get(c, "")) for c in label_cols)
            roi = float(r.get("roi", 0))
            lucro = float(r.get("lucro", 0))
            rows.append(f"<tr><td>{label}</td><td>{r.get('registros',0)}</td><td>{float(r.get('win_rate',0)):.1%}</td><td class='{'good' if roi >= 0 else 'bad'}'>{roi:+.2%}</td><td class='{'good' if lucro >= 0 else 'bad'}'>R$ {lucro:.2f}</td><td>{float(r.get('odd_media',0)):.2f}</td></tr>")
        return "<table><thead><tr><th>Segmento</th><th>Registros</th><th>Win rate</th><th>ROI</th><th>P&L</th><th>Odd média</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Football Lab — Inteligência de Segmentos</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f6f7fb;color:#152033}}section{{background:white;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 2px 10px #0001}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e8eaf0;text-align:left}}th{{background:#eef1f7}}.good{{color:#087a35;font-weight:700}}.bad{{color:#b00020;font-weight:700}}</style></head><body>
<h1>Football Lab — Inteligência Histórica de Segmentos</h1><p>Gerado em {payload['generated_at']}. Relatório descritivo, sem recomendação operacional.</p>
<section><h2>Mercados — melhores amostras</h2>{table(payload['classificacaos']['mercado']['top'], ['mercado'])}</section>
<section><h2>Mercados — piores amostras</h2>{table(payload['classificacaos']['mercado']['bottom'], ['mercado'])}</section>
<section><h2>Ligas — melhores amostras</h2>{table(payload['classificacaos']['liga']['top'], ['liga'])}</section>
<section><h2>Faixas de odd</h2>{table(payload['classificacaos']['odd_bucket']['top'], ['odd_bucket'])}</section>
<section><h2>Mercado + Liga</h2>{table(payload['classificacaos']['mercado_liga']['top'], ['mercado','liga'])}</section>
</body></html>"""
    (REPORTS_DIR / "painel_inteligencia_mercados.html").write_text(html, encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_history()
    if not df.empty:
        df["odd_bucket"] = df["odd"].apply(odd_bucket)
        dt = pd.to_datetime(df["data"], errors="coerce")
        df["weekday"] = dt.dt.day_name().fillna("unknown")
    groups = {
        "mercado": summarize(df, ["mercado"], min_sample=10),
        "liga": summarize(df, ["liga"], min_sample=10),
        "odd_bucket": summarize(df, ["odd_bucket"], min_sample=10),
        "weekday": summarize(df, ["weekday"], min_sample=10),
        "mercado_liga": summarize(df, ["mercado", "liga"], min_sample=5),
        "mercado_odd_bucket": summarize(df, ["mercado", "odd_bucket"], min_sample=5),
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "historical_segment_analysis_only",
        "min_sample_default": MIN_SAMPLE_DEFAULT,
        "classificacaos": {k: top_bottom(v, 10) for k, v in groups.items()},
        "full_segments": groups,
    }
    (REPORTS_DIR / "inteligencia_mercados.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(payload)
    print(json.dumps({"status": "OK", "files": ["04_ml/reports/inteligencia_mercados.json", "04_ml/reports/painel_inteligencia_mercados.html"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
