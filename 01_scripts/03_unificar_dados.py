from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CAMINHO_DADOS = Path(os.getenv("FUTPYTHON_RAW_DIR", "data/raw/futpython/leagues"))
CAMINHO_CATALOGO = Path(os.getenv("FUTPYTHON_CATALOG_PATH", "data/catalog/ligas_catalog.csv"))
CAMINHO_SAIDA = Path("data/base_unificada.csv")
EXTENSOES = {".csv", ".xlsx", ".xls"}


def ler_arquivo(arquivo: Path) -> pd.DataFrame:
    if arquivo.suffix.lower() == ".csv":
        return pd.read_csv(arquivo)
    return pd.read_excel(arquivo)


def carregar_catalogo() -> pd.DataFrame:
    if CAMINHO_CATALOGO.exists():
        return pd.read_csv(CAMINHO_CATALOGO, dtype=str).fillna("")
    return pd.DataFrame(columns=["country", "liga", "season", "liga_id"])


def enriquecer_metadados(df: pd.DataFrame, arquivo: Path, catalogo: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["arquivo_origem"] = arquivo.name

    # Caso a API já traga League/Home/Away, preserva. Caso contrário, tenta usar metadados pelo nome/meta.
    meta_path = arquivo.with_suffix(".meta.txt")
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

    if "League" not in df.columns and meta.get("liga"):
        df["League"] = meta["liga"]
    if "country" not in df.columns and meta.get("country"):
        df["country"] = meta["country"]
    if "Season" not in df.columns and meta.get("season"):
        df["Season"] = meta["season"]
    if "liga_id" not in df.columns and meta.get("liga_id"):
        df["liga_id"] = meta["liga_id"]

    df["liga_arquivo"] = meta.get("liga", arquivo.stem)
    return df


def carregar_dados(caminho_base: Path):
    dataframes = []
    catalogo = carregar_catalogo()

    if not caminho_base.exists():
        print(f"Pasta não encontrada: {caminho_base}")
        return dataframes

    for arquivo in caminho_base.rglob("*"):
        if arquivo.suffix.lower() in EXTENSOES:
            try:
                df = ler_arquivo(arquivo)
                df = enriquecer_metadados(df, arquivo, catalogo)
                dataframes.append(df)
            except Exception as erro:
                print(f"Erro ao processar {arquivo.name}: {erro}")

    return dataframes


def unificar_dataframes(lista_dfs):
    if not lista_dfs:
        return pd.DataFrame()
    return pd.concat(lista_dfs, ignore_index=True, sort=False)


def normalizar_chaves_merge(df: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    """
    Evita erro de merge quando uma mesma chave vem como int em uma base
    e texto em outra. Isso acontece bastante com Season/temporada.
    """
    df = df.copy()
    for coluna in chaves:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("string").fillna("").str.strip()
    return df


def salvar_dataframe(df: pd.DataFrame, caminho_saida: Path):
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("Nenhum dado para salvar.")
        return

    if caminho_saida.exists():
        df_existente = pd.read_csv(caminho_saida, low_memory=False)
        if "Id_Jogo" in df.columns and "Id_Jogo" in df_existente.columns:
            ids_existentes = set(df_existente["Id_Jogo"].astype(str).str.strip())
            df_novos = df[~df["Id_Jogo"].astype(str).str.strip().isin(ids_existentes)]
        else:
            chaves = [
                c
                for c in ["Date", "League", "Home", "Away", "Season"]
                if c in df.columns and c in df_existente.columns
            ]
            if chaves:
                df_merge = normalizar_chaves_merge(df, chaves)
                existente_merge = normalizar_chaves_merge(df_existente[chaves].drop_duplicates(), chaves)
                df_tmp = df_merge.merge(existente_merge, on=chaves, how="left", indicator=True)
                df_novos = df_tmp[df_tmp["_merge"] == "left_only"].drop(columns=["_merge"])
            else:
                df_novos = df
        df_final = pd.concat([df_existente, df_novos], ignore_index=True, sort=False)
        print(f"Jogos novos adicionados: {len(df_novos)}")
    else:
        df_final = df

    df_final.to_csv(caminho_saida, index=False)


def main():
    dataframes = carregar_dados(CAMINHO_DADOS)
    print(f"Arquivos processados: {len(dataframes)}")
    df_final = unificar_dataframes(dataframes)
    print(f"Linhas totais: {df_final.shape[0]}")
    print(f"Colunas totais: {df_final.shape[1]}")
    salvar_dataframe(df_final, CAMINHO_SAIDA)
    print(f"\nArquivo salvo em: {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
