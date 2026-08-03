from __future__ import annotations

from pathlib import Path

import pandas as pd

# ==============================
# Configurações
# ==============================
CAMINHO_BASE_LIGAS = Path("data/base_ligas.csv")
CAMINHO_SAIDA = Path("data/dicionario_times.csv")


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
    """Normaliza colunas mínimas usadas pelo dicionário de times."""
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


# ==============================
# Função principal
# ==============================
def gerar_dicionario_times() -> None:
    if not CAMINHO_BASE_LIGAS.exists():
        raise FileNotFoundError(f"Base de ligas não encontrada: {CAMINHO_BASE_LIGAS}")

    df = pd.read_csv(CAMINHO_BASE_LIGAS, dtype=str).fillna("")
    df = garantir_colunas_base(df)

    lista_times = []

    grupos = df.groupby(["country", "League_padronizada"], dropna=False)
    for (pais, liga), df_liga in grupos:
        times_home = df_liga["Home"].dropna().astype(str).str.strip()
        times_away = df_liga["Away"].dropna().astype(str).str.strip()

        times_unicos = sorted(set(times_home) | set(times_away))
        for time in times_unicos:
            if not time:
                continue
            lista_times.append(
                {
                    "country": pais,
                    "League_padronizada": liga,
                    "Time_original": time,
                    "Time_padronizado": time,
                    "Novo": False,
                }
            )

    df_dicionario = pd.DataFrame(
        lista_times,
        columns=["country", "League_padronizada", "Time_original", "Time_padronizado", "Novo"],
    )

    if not df_dicionario.empty:
        df_dicionario = df_dicionario.sort_values(
            ["country", "League_padronizada", "Time_original"]
        ).reset_index(drop=True)

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df_dicionario.to_csv(CAMINHO_SAIDA, index=False)
    print(f"Dicionário de times salvo em: {CAMINHO_SAIDA}")
    print(f"Total de times: {len(df_dicionario)}")


if __name__ == "__main__":
    gerar_dicionario_times()
