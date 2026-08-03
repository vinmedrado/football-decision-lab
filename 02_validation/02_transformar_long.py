import pandas as pd
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

INPUT_FILE = ROOT_DIR / "data" / "base_oficial.csv"
FALLBACK_FILE = ROOT_DIR / "data" / "base_times_padronizados.csv"
MARKET_FILE = BASE_DIR / "mercados_map.json"
OUTPUT_DIR = ROOT_DIR / "data" / "eventos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ler_base():
    caminho = INPUT_FILE if INPUT_FILE.exists() else FALLBACK_FILE
    if not caminho.exists():
        raise FileNotFoundError(
            "Nenhuma base encontrada. Esperado data/base_oficial.csv ou data/base_times_padronizados.csv"
        )
    print(f"Base usada para eventos: {caminho}")
    df = pd.read_csv(caminho, encoding="utf-8-sig", low_memory=False, dtype={"Game_ID": "string"})
    df.columns = df.columns.astype(str).str.strip()
    # Remove colunas duplicadas mantendo a primeira ocorrência.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _pegar_primeira_coluna(df, possiveis):
    mapa = {str(c).lower().strip(): c for c in df.columns}
    for col in possiveis:
        real = mapa.get(str(col).lower().strip())
        if real is not None:
            valor = df.loc[:, real]
            if isinstance(valor, pd.DataFrame):
                valor = valor.bfill(axis=1).iloc[:, 0]
            return valor
    return None


def garantir_colunas_minimas(df):
    """Cria aliases canônicos esperados pelo runner do backtest.

    A FutPython pode vir com nomes diferentes dos arquivos antigos. Esta função
    centraliza a ponte para que o backtest continue usando: Country,
    League_std, Home, Away, Date, Season, Round e Game_ID.
    """
    # Remove/combina duplicadas que podem aparecer depois de merges.
    if df.columns.duplicated().any():
        novo = pd.DataFrame(index=df.index)
        for col in dict.fromkeys(df.columns):
            bloco = df.loc[:, df.columns == col]
            if bloco.shape[1] == 1:
                novo[col] = bloco.iloc[:, 0]
            else:
                novo[col] = bloco.replace(r"^\s*$", pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
        df = novo

    aliases = {
        "Country": ["Country", "country", "Pais", "País", "pais"],
        "League_std": [
            "League_std", "liga_std", "Liga_std", "liga_std",
            "League_padronizada", "liga_padronizada", "Liga_padronizada", "liga_padronizada",
            "League", "liga", "Liga", "liga", "Competition", "competition"
        ],
        "Home": ["Home", "home", "Mandante", "mandante", "Home_new", "Home_std", "Home Team", "home_team"],
        "Away": ["Away", "away", "Visitante", "visitante", "Away_new", "Away_std", "Away Team", "away_team"],
        "Date": ["Date", "date", "Data", "data"],
        "Season": ["Season", "season", "Temporada", "temporada"],
        "Round": ["Round", "round", "Rodada", "rodada"],
    }

    for destino, possiveis in aliases.items():
        valor = _pegar_primeira_coluna(df, possiveis)
        if valor is not None:
            df[destino] = valor.fillna("").astype(str).str.strip()

    if "League_std" not in df.columns or df["League_std"].astype(str).str.strip().eq("").all():
        raise KeyError(
            "Não consegui montar League_std. Colunas disponíveis: " + ", ".join(map(str, df.columns))
        )

    if "Game_ID" not in df.columns or df["Game_ID"].astype(str).str.strip().eq("").all():
        cols = [c for c in ["Date", "Country", "League_std", "Home", "Away"] if c in df.columns]
        if cols:
            df["Game_ID"] = df[cols].fillna("").astype(str).agg("__".join, axis=1)
        else:
            df["Game_ID"] = range(1, len(df) + 1)
    else:
        df["Game_ID"] = df["Game_ID"].fillna("").astype(str).str.strip()

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce", format="mixed", dayfirst=True)
        df["Date"] = dt.dt.strftime("%Y-%m-%d").fillna(df["Date"].astype(str).str.strip())

    return df

def main():
    df = garantir_colunas_minimas(ler_base())

    with open(MARKET_FILE, "r", encoding="utf-8") as f:
        mercado_map = json.load(f)

    metadata_cols_preferidas = [
        "League_std", "Country", "Num", "Game_ID", "Season", "Date", "Round",
        "Home", "Away", "PPG_H_Pre", "PPG_A_Pre", "PPG_H", "PPG_A",
        "XG_H_Pre", "XG_A_Pre", "XG_T_Pre", "File_Origin", "League_File",
        "Home_std", "Away_std", "Home_new", "Away_new"
    ]
    metadata_cols = [c for c in metadata_cols_preferidas if c in df.columns]
    obrigatorias = ["Game_ID", "Date", "League_std", "Home", "Away"]
    faltando = [c for c in obrigatorias if c not in metadata_cols]
    if faltando:
        raise KeyError(f"Colunas obrigatórias ausentes para backtest: {faltando}")

    colunas_nao_encontradas = []
    colunas_processadas = []

    for i, (col, info) in enumerate(mercado_map.items(), start=1):
        tipo = info.get("type")
        event_name = info.get("event", col)

        if tipo != "quantidade":
            continue

        if col not in df.columns:
            colunas_nao_encontradas.append(col)
            print(f"[AVISO] Coluna não encontrada: {col}")
            continue

        try:
            df_out = df[metadata_cols + [col]].copy()
            df_out.rename(columns={col: "value"}, inplace=True)
            df_out["value"] = pd.to_numeric(df_out["value"], errors="coerce").fillna(0).astype("float64")
            df_out["event"] = event_name

            output_path = OUTPUT_DIR / f"{event_name}.csv"
            df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

            colunas_processadas.append(col)
            print(f"Mercado salvo: {output_path} ({len(df_out)} linhas) [{i}/{len(mercado_map)}]")

        except Exception as e:
            colunas_nao_encontradas.append(col)
            print(f"[ERRO] Falha ao processar {col}: {e}")

    print("\nTransformação para eventos concluída.")
    print(f"Colunas processadas com sucesso: {len(colunas_processadas)}")
    print(f"Colunas ausentes/erro: {len(colunas_nao_encontradas)}")
    if colunas_nao_encontradas:
        print("Lista de colunas não processadas:", colunas_nao_encontradas)


if __name__ == "__main__":
    main()
