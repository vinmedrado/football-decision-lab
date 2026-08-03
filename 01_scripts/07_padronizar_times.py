from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

# ==============================
# Configurações
# ==============================
CAMINHO_BASE = Path("data/base_ligas.csv")
CAMINHO_DICIONARIO = Path("data/dicionario_times.csv")
CAMINHO_SAIDA = Path("data/base_times_padronizados.csv")
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


def encontrar_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    mapa = {c.lower().strip(): c for c in df.columns}
    for candidato in candidatos:
        col = mapa.get(candidato.lower().strip())
        if col:
            return col
    return None

def normalizar_nome_coluna(nome: str) -> str:
    return str(nome).lower().strip().replace("_", " ").replace("-", " ")


def coalescer_coluna(df: pd.DataFrame, destino: str, candidatos: list[str]) -> pd.DataFrame:
    """Cria uma coluna canônica a partir de aliases, evitando colunas duplicadas.

    Ex.: Country/country/Pais -> country. Se houver mais de uma coluna equivalente,
    usa o primeiro valor não vazio por linha.
    """
    candidatos_norm = {normalizar_nome_coluna(c) for c in candidatos + [destino]}
    colunas_match = [c for c in df.columns if normalizar_nome_coluna(c) in candidatos_norm]

    if not colunas_match:
        df[destino] = ""
        return df

    partes = df.loc[:, colunas_match].copy()
    if isinstance(partes, pd.Series):
        serie = partes
    else:
        partes = partes.replace(r"^\s*$", pd.NA, regex=True)
        serie = partes.bfill(axis=1).iloc[:, 0]

    df = df.drop(columns=colunas_match)
    df[destino] = serie.fillna("").astype(str).str.strip()
    return df


def garantir_colunas_base(df: pd.DataFrame) -> pd.DataFrame:
    df = remover_colunas_duplicadas(df.copy())

    df = coalescer_coluna(df, "country", ["country", "Country", "País", "Pais", "nation"])
    df = coalescer_coluna(df, "League_padronizada", ["League_padronizada", "League", "liga", "Liga", "competition"])
    df = coalescer_coluna(df, "Home", ["Home", "Home Team", "home_team", "Mandante", "Time Mandante"])
    df = coalescer_coluna(df, "Away", ["Away", "Away Team", "away_team", "Visitante", "Time Visitante"])

    df = remover_colunas_duplicadas(df)
    for col in ["country", "League_padronizada", "Home", "Away"]:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].bfill(axis=1).iloc[:, 0]
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CAMINHO_BASE.exists():
        raise FileNotFoundError(f"Base de ligas não encontrada: {CAMINHO_BASE}")

    df_base = pd.read_csv(CAMINHO_BASE, dtype=str).fillna("")
    df_base = garantir_colunas_base(df_base)

    if CAMINHO_DICIONARIO.exists():
        df_dic = pd.read_csv(CAMINHO_DICIONARIO, dtype=str).fillna("")
        df_dic = remover_colunas_duplicadas(df_dic)
    else:
        df_dic = pd.DataFrame(columns=["League_padronizada", "country", "Time_padronizado"])

    for col in ["League_padronizada", "country", "Time_padronizado"]:
        if col not in df_dic.columns:
            df_dic[col] = ""
        df_dic[col] = df_dic[col].fillna("").astype(str).str.strip()

    return df_base, df_dic


def padronizar_time(time: str, liga: str, pais: str, df_dic: pd.DataFrame, limite: int = LIMITE_FUZZY):
    if pd.isna(time) or str(time).strip() == "":
        return time, False

    df_filtrado = df_dic[(df_dic["League_padronizada"] == liga) & (df_dic["country"] == pais)]
    times_ref = df_filtrado["Time_padronizado"].dropna().astype(str).tolist()

    if not times_ref:
        return time, True

    resultado = process.extractOne(str(time), times_ref, scorer=fuzz.ratio)
    if not resultado:
        return time, True

    match, score, _ = resultado
    if score >= limite:
        return match, False
    return time, True


def aplicar_padronizacao(df_base: pd.DataFrame, df_dic: pd.DataFrame) -> pd.DataFrame:
    def processar_linha(row):
        liga = row.get("League_padronizada", "")
        pais = row.get("country", "")

        home, novo_home = padronizar_time(row.get("Home", ""), liga, pais, df_dic)
        away, novo_away = padronizar_time(row.get("Away", ""), liga, pais, df_dic)

        row["Home_padronizado"] = home
        row["Away_padronizado"] = away
        row["Home_novo"] = novo_home
        row["Away_novo"] = novo_away
        return row

    df_base = df_base.apply(processar_linha, axis=1)
    return df_base


def atualizar_dicionario(df_base: pd.DataFrame, df_dic: pd.DataFrame) -> pd.DataFrame:
    novos_home = df_base[df_base["Home_novo"]][["League_padronizada", "country", "Home_padronizado"]]
    novos_away = df_base[df_base["Away_novo"]][["League_padronizada", "country", "Away_padronizado"]]

    if not novos_home.empty:
        novos_home = novos_home.rename(columns={"Home_padronizado": "Time_padronizado"})
        df_dic = pd.concat([df_dic, novos_home], ignore_index=True)

    if not novos_away.empty:
        novos_away = novos_away.rename(columns={"Away_padronizado": "Time_padronizado"})
        df_dic = pd.concat([df_dic, novos_away], ignore_index=True)

    df_dic = df_dic.drop_duplicates(subset=["League_padronizada", "country", "Time_padronizado"])
    return df_dic


def main() -> None:
    print("Carregando dados...")
    df_base, df_dic = carregar_dados()

    print("Padronizando times...")
    df_base = aplicar_padronizacao(df_base, df_dic)

    print("Atualizando dicionário de times...")
    df_dic = atualizar_dicionario(df_base, df_dic)

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df_base.to_csv(CAMINHO_SAIDA, index=False)
    df_dic.to_csv(CAMINHO_DICIONARIO, index=False)

    print(f"Padronização concluída. Arquivo salvo em: {CAMINHO_SAIDA}")
    print(f"Times novos Home: {df_base['Home_novo'].sum()}, Away: {df_base['Away_novo'].sum()}")
    print(f"Dicionário de times atualizado: {len(df_dic)} registros")


if __name__ == "__main__":
    main()
