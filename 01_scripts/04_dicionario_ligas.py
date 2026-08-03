from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CAMINHO_CATALOGO = Path(os.getenv("FUTPYTHON_CATALOG_PATH", "data/catalog/ligas_catalog.csv"))
CAMINHO_SAIDA = Path("data/dicionario_ligas.csv")


def gerar_dicionario_ligas():
    if not CAMINHO_CATALOGO.exists():
        raise FileNotFoundError(
            f"Catálogo não encontrado: {CAMINHO_CATALOGO}. Rode 00_build_ligas_catalog.py primeiro."
        )

    df_ligas = pd.read_csv(CAMINHO_CATALOGO, dtype=str).fillna("")
    df_dic = df_ligas[["liga", "country"]].copy()
    df_dic = df_dic.rename(columns={"liga": "name"})
    df_dic = df_dic.drop_duplicates(subset=["name", "country"]).reset_index(drop=True)

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df_dic.to_csv(CAMINHO_SAIDA, index=False)
    print(f"Dicionário de ligas salvo em: {CAMINHO_SAIDA}")
    print(f"Total de ligas: {len(df_dic)}")


if __name__ == "__main__":
    gerar_dicionario_ligas()
