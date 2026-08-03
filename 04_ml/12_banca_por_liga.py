#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quebra o histórico de apostas REAL (já liquidado) por liga, não só por mercado.

O painel do 04_banca.py mostra só o agregado por mercado (ex: R_FT_H | -7.6%),
o que esconde exatamente o problema que motivou a elegibilidade por liga lá no
backtest: um mercado pode estar negativo no total e ainda assim ter ligas boas
escondidas dentro dele.

Esse script lê o mesmo historico_apostas.csv que o 04_banca.py já usa e mostra
o ROI e winrate REAL, liga por liga, dentro de cada mercado.

Uso:
    python 04_ml/12_banca_por_liga.py
    python 04_ml/12_banca_por_liga.py --min-apostas 10
"""
import argparse
from pathlib import Path

import pandas as pd

HISTORICO = Path("04_ml/banca/historico_apostas.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-apostas", type=int, default=5,
                         help="Só mostra ligas com pelo menos N apostas liquidadas (padrão: 5)")
    args = parser.parse_args()

    if not HISTORICO.exists():
        print(f"Histórico não encontrado: {HISTORICO}")
        return

    df = pd.read_csv(HISTORICO, low_memory=False)
    df["resultado"] = df["resultado"].astype(str).str.strip().str.lower()
    df = df[df["resultado"].isin(["ganhou", "perdeu"])].copy()

    if df.empty:
        print("Nenhuma aposta liquidada ainda (só pendentes).")
        return

    df["valor_apostado"] = pd.to_numeric(df["valor_apostado"], errors="coerce").fillna(0)
    df["lucro"] = pd.to_numeric(df["lucro"], errors="coerce").fillna(0)
    df["liga"] = df["liga"].astype(str).str.strip().str.upper()

    resumo = df.groupby(["mercado", "liga"]).agg(
        apostas=("resultado", "size"),
        ganhos=("resultado", lambda s: (s == "ganhou").sum()),
        stake=("valor_apostado", "sum"),
        lucro=("lucro", "sum"),
    ).reset_index()

    resumo["winrate"] = (resumo["ganhos"] / resumo["apostas"]).round(4)
    resumo["roi"] = (resumo["lucro"] / resumo["stake"].replace(0, float("nan"))).round(4)

    resumo = resumo[resumo["apostas"] >= args.min_apostas].sort_values("roi", ascending=False)

    print("=" * 90)
    print(f"DESEMPENHO REAL POR LIGA (min. {args.min_apostas} apostas liquidadas)")
    print("=" * 90)
    print(
        resumo[["mercado", "liga", "apostas", "ganhos", "winrate", "lucro", "roi"]]
        .to_string(index=False, formatters={
            "winrate": "{:.1%}".format,
            "roi": "{:+.1%}".format,
            "lucro": "R$ {:+.2f}".format,
        })
    )
    print("=" * 90)

    positivas = resumo[resumo["roi"] > 0]
    negativas = resumo[resumo["roi"] <= 0]
    print(f"\nLigas com ROI positivo : {len(positivas)} de {len(resumo)}")
    print(f"Ligas com ROI negativo : {len(negativas)} de {len(resumo)}")
    if not positivas.empty:
        print(f"\nMelhor liga: {positivas.iloc[0]['liga']} "
              f"({positivas.iloc[0]['apostas']} apostas, ROI {positivas.iloc[0]['roi']:+.1%})")
    if not negativas.empty:
        pior = negativas.sort_values('roi').iloc[0]
        print(f"Pior liga : {pior['liga']} "
              f"({pior['apostas']} apostas, ROI {pior['roi']:+.1%})")


if __name__ == "__main__":
    main()
