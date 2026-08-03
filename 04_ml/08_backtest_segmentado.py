#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relatório operacional segmentado por mercado/liga/odd.

Este script usa o histórico real da banca como proxy operacional. Não retreina modelo.
Saídas:
- 04_ml/reports/backtest_segmentado_mercado_liga.csv
- 04_ml/reports/resumo_backtest_segmentado.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
HISTORICO_FILE = BASE_DIR / "banca" / "historico_apostas.csv"
REPORTS_DIR = BASE_DIR / "reports"
OUT_CSV = REPORTS_DIR / "backtest_segmentado_mercado_liga.csv"
OUT_JSON = REPORTS_DIR / "resumo_backtest_segmentado.json"
MIN_AMOSTRA = 20


def odd_bucket(odd: float) -> str:
    if odd < 1.35:
        return "<1.35"
    if odd < 1.50:
        return "1.35-1.49"
    if odd < 1.70:
        return "1.50-1.69"
    if odd < 2.00:
        return "1.70-1.99"
    return ">=2.00"


def load_history() -> pd.DataFrame:
    if not HISTORICO_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORICO_FILE)
    for col in ["odd", "valor_apostado", "lucro", "confianca", "prob_modelo", "roi_bt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["resultado", "mercado", "liga", "data"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    df = df[df["resultado"].str.lower().isin(["ganhou", "perdeu"])].copy()
    df["odd_bucket"] = df["odd"].fillna(0).apply(odd_bucket)
    return df


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        apostas=("resultado", "size"),
        stake=("valor_apostado", "sum"),
        lucro=("lucro", "sum"),
        odd_media=("odd", "mean"),
        confianca_media=("confianca", "mean"),
        roi_bt_medio=("roi_bt", "mean"),
    ).reset_index()
    wins = grouped["resultado"].apply(lambda s: (s.str.lower() == "ganhou").mean()).reset_index(name="winrate")
    out = out.merge(wins, on=group_cols, how="left")
    out["roi_real"] = out.apply(lambda r: (r["lucro"] / r["stake"]) if r["stake"] > 0 else 0.0, axis=1)
    out["aprovado_amostra"] = out["apostas"] >= MIN_AMOSTRA
    out["alerta"] = out.apply(
        lambda r: "pausar_segmento" if r["aprovado_amostra"] and r["roi_real"] < 0 else (
            "amostra_baixa" if not r["aprovado_amostra"] else "ok"
        ),
        axis=1,
    )
    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_history()
    if df.empty:
        print("⚠️ Sem histórico finalizado para segmentar.")
        return 0

    tables = []
    for name, group_cols in {
        "mercado": ["mercado"],
        "liga": ["liga"],
        "mercado_liga": ["mercado", "liga"],
        "mercado_odd_bucket": ["mercado", "odd_bucket"],
    }.items():
        t = summarize(df, group_cols)
        if not t.empty:
            t.insert(0, "segmento", name)
            tables.append(t)

    final = pd.concat(tables, ignore_index=True, sort=False)
    final = final.sort_values(["segmento", "apostas", "roi_real"], ascending=[True, False, False])
    final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    mercado = summarize(df, ["mercado"]).sort_values("apostas", ascending=False)
    resumo = {
        "total_apostas_finalizadas": int(len(df)),
        "min_amostra_segmento": MIN_AMOSTRA,
        "mercados": mercado[["mercado", "apostas", "stake", "lucro", "roi_real", "winrate", "odd_media", "alerta"]].to_dict(orient="records"),
        "arquivos": {"csv": str(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Relatório segmentado gerado: {OUT_CSV}")
    print(f"✅ Resumo JSON gerado: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
