#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Football Lab — Revisor simples de aliases de times.

Lê data/times_alias_revisao.csv e aprova/edita/rejeita sugestões,
atualizando data/dicionario_times_alias.csv.

Uso:
    python 04_ml/08_revisar_alias_times.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
ALIAS_FILE = ROOT_DIR / "data" / "dicionario_times_alias.csv"
REVISAO_FILE = ROOT_DIR / "data" / "times_alias_revisao.csv"

COLS_ALIAS = [
    "alias",
    "time_padronizado",
    "score",
    "fonte",
    "data_exemplo",
    "jogo_historico",
    "jogo_base",
]


def carregar_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def salvar_alias(df: pd.DataFrame) -> None:
    df = df[COLS_ALIAS].drop_duplicates(subset=["alias"], keep="first")
    df.to_csv(ALIAS_FILE, index=False, encoding="utf-8-sig")


def main() -> int:
    alias = carregar_csv(ALIAS_FILE, COLS_ALIAS)
    rev = carregar_csv(REVISAO_FILE, [])

    if rev.empty:
        print("Nenhum item para revisão.")
        return 0

    i = 0
    while i < len(rev):
        row = rev.iloc[i]
        print("\n" + "=" * 80)
        print(f"Item {i+1}/{len(rev)}")
        print(f"Data        : {row.get('data', '')}")
        print(f"Liga        : {row.get('liga', '')}")
        print(f"Histórico   : {row.get('jogo_historico', '')}")
        print(f"Sugestão    : {row.get('jogo_base_sugerido', '')}")
        print(f"Home        : {row.get('home_hist', '')} -> {row.get('home_base_sugerido', '')} ({row.get('score_home', '')})")
        print(f"Away        : {row.get('away_hist', '')} -> {row.get('away_base_sugerido', '')} ({row.get('score_away', '')})")
        print("=" * 80)
        print("a = aprovar | e = editar | r = rejeitar | p = pular | q = sair")
        op = input("Opção: ").strip().lower()

        if op == "q":
            break
        if op == "p" or not op:
            i += 1
            continue
        if op == "r":
            rev = rev.drop(rev.index[i]).reset_index(drop=True)
            continue
        if op == "a":
            novos = [
                {
                    "alias": row.get("home_hist", ""),
                    "time_padronizado": row.get("home_base_sugerido", ""),
                    "score": row.get("score_home", 0),
                    "fonte": "revisao_manual",
                    "data_exemplo": row.get("data", ""),
                    "jogo_historico": row.get("jogo_historico", ""),
                    "jogo_base": row.get("jogo_base_sugerido", ""),
                },
                {
                    "alias": row.get("away_hist", ""),
                    "time_padronizado": row.get("away_base_sugerido", ""),
                    "score": row.get("score_away", 0),
                    "fonte": "revisao_manual",
                    "data_exemplo": row.get("data", ""),
                    "jogo_historico": row.get("jogo_historico", ""),
                    "jogo_base": row.get("jogo_base_sugerido", ""),
                },
            ]
            alias = pd.concat([alias, pd.DataFrame(novos)], ignore_index=True)
            rev = rev.drop(rev.index[i]).reset_index(drop=True)
            salvar_alias(alias)
            rev.to_csv(REVISAO_FILE, index=False, encoding="utf-8-sig")
            print("Aprovado.")
            continue
        if op == "e":
            home_pad = input(f"Padronizado para HOME [{row.get('home_base_sugerido','')}]: ").strip() or str(row.get("home_base_sugerido", ""))
            away_pad = input(f"Padronizado para AWAY [{row.get('away_base_sugerido','')}]: ").strip() or str(row.get("away_base_sugerido", ""))
            novos = [
                {
                    "alias": row.get("home_hist", ""),
                    "time_padronizado": home_pad,
                    "score": 100,
                    "fonte": "revisao_manual_editada",
                    "data_exemplo": row.get("data", ""),
                    "jogo_historico": row.get("jogo_historico", ""),
                    "jogo_base": row.get("jogo_base_sugerido", ""),
                },
                {
                    "alias": row.get("away_hist", ""),
                    "time_padronizado": away_pad,
                    "score": 100,
                    "fonte": "revisao_manual_editada",
                    "data_exemplo": row.get("data", ""),
                    "jogo_historico": row.get("jogo_historico", ""),
                    "jogo_base": row.get("jogo_base_sugerido", ""),
                },
            ]
            alias = pd.concat([alias, pd.DataFrame(novos)], ignore_index=True)
            rev = rev.drop(rev.index[i]).reset_index(drop=True)
            salvar_alias(alias)
            rev.to_csv(REVISAO_FILE, index=False, encoding="utf-8-sig")
            print("Editado e aprovado.")
            continue

        print("Opção inválida.")

    salvar_alias(alias)
    rev.to_csv(REVISAO_FILE, index=False, encoding="utf-8-sig")
    print("\nRevisão salva.")
    print(f"Aliases: {ALIAS_FILE}")
    print(f"Pendentes: {REVISAO_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
