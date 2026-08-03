from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

# ==============================
# Configurações
# ==============================
CAMINHO_BASE = Path("data/base_unificada.csv")
CAMINHO_DIC_LIGAS = Path("data/dicionario_ligas.csv")
CAMINHO_SAIDA = Path("data/base_ligas.csv")

LIMITE_FUZZY = 90


def remover_colunas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas duplicadas preservando o primeiro valor não vazio por linha."""
    if not df.columns.duplicated().any():
        return df

    resultado = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        partes = df.loc[:, df.columns == col]
        if partes.shape[1] == 1:
            resultado[col] = partes.iloc[:, 0]
            continue

        partes = partes.copy()
        partes = partes.replace(r"^\s*$", pd.NA, regex=True)
        resultado[col] = partes.bfill(axis=1).iloc[:, 0]

    return resultado


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CAMINHO_BASE.exists():
        raise FileNotFoundError(f"Base unificada não encontrada: {CAMINHO_BASE}")
    if not CAMINHO_DIC_LIGAS.exists():
        raise FileNotFoundError(f"Dicionário de ligas não encontrado: {CAMINHO_DIC_LIGAS}")

    df = pd.read_csv(CAMINHO_BASE, dtype=str).fillna("")
    df_dic_ligas = pd.read_csv(CAMINHO_DIC_LIGAS, dtype=str).fillna("")
    return df, df_dic_ligas


def encontrar_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    mapa = {c.lower().strip(): c for c in df.columns}
    for candidato in candidatos:
        col = mapa.get(candidato.lower().strip())
        if col:
            return col
    return None


def padronizar_liga(valor: str, ligas_oficiais, limite: int = LIMITE_FUZZY) -> str:
    if pd.isna(valor) or str(valor).strip() == "":
        return valor

    resultado = process.extractOne(str(valor), ligas_oficiais, scorer=fuzz.ratio)
    if not resultado:
        return valor

    match, score, _ = resultado
    if score >= limite:
        return match
    return valor


def aplicar_padronizacao(df: pd.DataFrame, ligas_oficiais) -> pd.DataFrame:
    coluna_liga = encontrar_coluna(df, ["League", "liga", "Liga", "liga", "Competition", "competition"])
    if not coluna_liga:
        raise KeyError(
            "Não encontrei coluna de liga. Colunas disponíveis: " + ", ".join(df.columns)
        )

    if coluna_liga != "League":
        df = df.rename(columns={coluna_liga: "League"})

    df["League_padronizada"] = df["League"].apply(
        lambda x: padronizar_liga(x, ligas_oficiais)
    )
    return df


def adicionar_metadados(df: pd.DataFrame, df_dic_ligas: pd.DataFrame) -> pd.DataFrame:
    coluna_nome_dic = encontrar_coluna(df_dic_ligas, ["name", "liga", "Liga"])
    coluna_pais_dic = encontrar_coluna(df_dic_ligas, ["country", "Country", "País", "Pais"])

    if not coluna_nome_dic:
        raise KeyError("Dicionário de ligas não possui coluna de nome/name/liga.")
    if not coluna_pais_dic:
        raise KeyError("Dicionário de ligas não possui coluna de país/country.")

    dic = df_dic_ligas[[coluna_nome_dic, coluna_pais_dic]].copy()
    dic = dic.rename(columns={coluna_nome_dic: "name", coluna_pais_dic: "country_dic"})
    dic = dic.drop_duplicates(subset=["name"])

    # Se a base já tiver Country/country, preserva como country_original e depois prioriza o catálogo.
    coluna_pais_base = encontrar_coluna(df, ["country", "Country", "País", "Pais"])
    if coluna_pais_base:
        df = df.rename(columns={coluna_pais_base: "country_original"})

    df = df.merge(
        dic,
        left_on="League_padronizada",
        right_on="name",
        how="left",
    )

    if "country_dic" in df.columns:
        df["country"] = df["country_dic"]
    else:
        df["country"] = ""

    if "country_original" in df.columns:
        df["country"] = df["country"].where(df["country"].astype(str).str.strip() != "", df["country_original"])

    df = df.drop(columns=[c for c in ["name", "country_dic", "country_original"] if c in df.columns])
    df = remover_colunas_duplicadas(df)
    return df


def main() -> None:
    df, df_dic_ligas = carregar_dados()

    coluna_nome_dic = encontrar_coluna(df_dic_ligas, ["name", "liga", "Liga"])
    ligas_oficiais = df_dic_ligas[coluna_nome_dic].dropna().astype(str).unique()

    print("Padronizando ligas...")
    df = aplicar_padronizacao(df, ligas_oficiais)

    print("Adicionando metadados do dicionário de ligas...")
    df = adicionar_metadados(df, df_dic_ligas)

    colunas_inicio = [c for c in ["League", "League_padronizada", "country"] if c in df.columns]
    outras_colunas = [c for c in df.columns if c not in colunas_inicio]
    df = df[colunas_inicio + outras_colunas]

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False)

    print("Processo concluído.")
    print(f"Arquivo salvo em: {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
