#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIST = PROJECT_ROOT / "04_ml" / "banca" / "historico_apostas.csv"
OUT = PROJECT_ROOT / "04_ml" / "reports" / "contexto_operacional"

MIN_APOSTAS_LIGA = 10
MIN_APOSTAS_TIME = 5
MIN_APOSTAS_CONFRONTO = 3


def atomic_csv(df, path):
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def resumo(df, grupos):
    g = (
        df.groupby(grupos)
        .agg(
            apostas=("mercado", "count"),
            ganhos=("resultado", lambda x: (x == "ganhou").sum()),
            lucro=("lucro", "sum"),
            stake=("valor_apostado", "sum"),
            odd_media=("odd", "mean"),
            prob_media=("prob_modelo", "mean"),
            confianca_media=("confianca", "mean"),
        )
        .reset_index()
    )

    g["winrate"] = g["ganhos"] / g["apostas"]
    g["roi"] = g["lucro"] / g["stake"]
    g["break_even"] = 1 / g["odd_media"]
    g["edge_winrate"] = g["winrate"] - g["break_even"]
    return g.sort_values("lucro", ascending=False)


def classificar(row, min_apostas):
    if row["apostas"] < min_apostas:
        return "SEM_AMOSTRA"
    if row["roi"] >= 0.05:
        return "FORTE"
    if row["roi"] >= 0:
        return "POSITIVO"
    if row["roi"] > -0.05:
        return "NEUTRO"
    return "RUIM"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(HIST, low_memory=False)
    df = df[df["resultado"].isin(["ganhou", "perdeu"])].copy()

    for col in ["lucro", "valor_apostado", "odd", "prob_modelo", "confianca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["mercado"] = df["mercado"].astype(str).str.upper().str.strip()
    df["liga"] = df["liga"].astype(str).str.upper().str.strip()

    partes = df["jogo"].astype(str).str.split(" x ", n=1, expand=True)
    df["home"] = partes[0].fillna("").str.strip()
    df["away"] = partes[1].fillna("").str.strip()

    mercado = resumo(df, ["mercado"])
    mercado["status_contexto"] = mercado.apply(lambda r: classificar(r, 20), axis=1)

    liga_mercado = resumo(df, ["mercado", "liga"])
    liga_mercado["status_contexto"] = liga_mercado.apply(lambda r: classificar(r, MIN_APOSTAS_LIGA), axis=1)

    home_mercado = resumo(df, ["mercado", "home"])
    home_mercado["status_contexto"] = home_mercado.apply(lambda r: classificar(r, MIN_APOSTAS_TIME), axis=1)

    away_mercado = resumo(df, ["mercado", "away"])
    away_mercado["status_contexto"] = away_mercado.apply(lambda r: classificar(r, MIN_APOSTAS_TIME), axis=1)

    confronto_mercado = resumo(df, ["mercado", "home", "away"])
    confronto_mercado["status_contexto"] = confronto_mercado.apply(lambda r: classificar(r, MIN_APOSTAS_CONFRONTO), axis=1)

    atomic_csv(mercado, OUT / "01_mercado.csv")
    atomic_csv(liga_mercado, OUT / "02_liga_mercado.csv")
    atomic_csv(home_mercado, OUT / "03_home_mercado.csv")
    atomic_csv(away_mercado, OUT / "04_away_mercado.csv")
    atomic_csv(confronto_mercado, OUT / "05_confronto_mercado.csv")

    print("=" * 60)
    print("RELATÓRIO CONTEXTUAL OPERACIONAL GERADO")
    print("=" * 60)
    print(f"Mercados            : {len(mercado)}")
    print(f"Liga x mercado      : {len(liga_mercado)}")
    print(f"Home x mercado      : {len(home_mercado)}")
    print(f"Away x mercado      : {len(away_mercado)}")
    print(f"Confronto x mercado : {len(confronto_mercado)}")
    print(f"Pasta salva         : {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
