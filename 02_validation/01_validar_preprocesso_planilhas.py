# validation/01_validar_preprocesso_planilhas.py
import pandas as pd
import json
import os
import ast

# --- Configurações ---
BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "..", "data", "base_times_padronizados.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "base_oficial.csv")
SCHEMA_FILE = os.path.join(BASE_DIR, "..", "data", "schema.json")

# --- Carrega schema ---
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    schema = json.load(f)


# --- Aliases FutPython -> nomes esperados pelo schema legado ---
FUTPYTHON_ALIASES = {
    "Id_Jogo": ["Match_ID", "Game_ID"],
    "League_padronizada": ["League_std"],
    "country": ["Country", "Pais", "País"],
    "Rodada": ["Round"],
    "Goals_H_FT": ["Home_Score"],
    "Goals_A_FT": ["Away_Score"],
    "Odd_H_HT": ["Odd_1_HT"],
    "Odd_D_HT": ["Odd_X_HT"],
    "Odd_A_HT": ["Odd_2_HT"],
    "Odd_H_FT": ["Odd_1_FT"],
    "Odd_D_FT": ["Odd_X_FT"],
    "Odd_A_FT": ["Odd_2_FT"],
    "Odd_Over05_HT": ["Over_HT_0_5"],
    "Odd_Under05_HT": ["Under_HT_0_5"],
    "Odd_Over15_HT": ["Over_HT_1_5"],
    "Odd_Under15_HT": ["Under_HT_1_5"],
    "Odd_Over25_HT": ["Over_HT_2_5"],
    "Odd_Under25_HT": ["Under_HT_2_5"],
    "Odd_Over05_FT": ["Over_FT_0_5"],
    "Odd_Under05_FT": ["Under_FT_0_5"],
    "Odd_Over15_FT": ["Over_FT_1_5"],
    "Odd_Under15_FT": ["Under_FT_1_5"],
    "Odd_Over25_FT": ["Over_FT_2_5"],
    "Odd_Under25_FT": ["Under_FT_2_5"],
    "Odd_BTTS_Yes": ["BTTS_Yes"],
    "Odd_BTTS_No": ["BTTS_No"],
    "Odd_DC_1X": ["DC_1X"],
    "Odd_DC_12": ["DC_12"],
    "Odd_DC_X2": ["DC_X2"],
    "ShotsOnTarget_H": ["Shots_On_Target_Home_FT"],
    "ShotsOnTarget_A": ["Shots_On_Target_Away_FT"],
    "ShotsOffTarget_H": ["Shots_Off_Target_Home_FT"],
    "ShotsOffTarget_A": ["Shots_Off_Target_Away_FT"],
    "Shots_H": ["Total_Shots_Home_FT"],
    "Shots_A": ["Total_Shots_Away_FT"],
    "Corners_H_FT": ["Corners_Home_FT"],
    "Corners_A_FT": ["Corners_Away_FT"],
    "Odd_Corners_H": ["Corners_Home_FT"],
    "Odd_Corners_A": ["Corners_Away_FT"],
    "arquivo_origem": ["File_Origin"],
    "liga_arquivo": ["League_File"],
    "Home_padronizado": ["Home_std"],
    "Away_padronizado": ["Away_std"],
    "Home_novo": ["Home_new"],
    "Away_novo": ["Away_new"],
}

def aplicar_aliases_futpython(df):
    lower_cols = {str(c).lower().strip(): c for c in df.columns}
    for destino, possiveis in FUTPYTHON_ALIASES.items():
        if destino in df.columns:
            continue
        for candidato in possiveis:
            real = lower_cols.get(str(candidato).lower().strip())
            if real is not None:
                df[destino] = df[real]
                break
    if "TotalGoals_FT" not in df.columns and {"Goals_H_FT", "Goals_A_FT"}.issubset(df.columns):
        df["TotalGoals_FT"] = pd.to_numeric(df["Goals_H_FT"], errors="coerce").fillna(0) + pd.to_numeric(df["Goals_A_FT"], errors="coerce").fillna(0)
    if "TotalCorners_FT" not in df.columns and {"Corners_H_FT", "Corners_A_FT"}.issubset(df.columns):
        df["TotalCorners_FT"] = pd.to_numeric(df["Corners_H_FT"], errors="coerce").fillna(0) + pd.to_numeric(df["Corners_A_FT"], errors="coerce").fillna(0)
    return df

# --- Lê a planilha ---
if INPUT_FILE.endswith(".csv"):
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.replace("Â", "", regex=False).str.strip()
elif INPUT_FILE.endswith(".xlsx"):
    df = pd.read_excel(INPUT_FILE)
    df.columns = df.columns.str.replace("Â", "", regex=False).str.strip()
else:
    raise ValueError("Formato de arquivo não suportado!")

df = aplicar_aliases_futpython(df)

# --- Verifica colunas obrigatórias ---
missing_cols = [col for col in schema if col not in df.columns]
if missing_cols:
    print(f"[ERRO] Colunas obrigatórias faltando: {missing_cols}")

# --- Renomeia colunas ---
rename_map = {col: schema[col]["rename"] for col in schema if col in df.columns}
df.rename(columns=rename_map, inplace=True)

# --- Funções auxiliares ---
def to_bool(value):
    if pd.isna(value):
        return pd.NA

    value_str = str(value).strip().lower()

    true_values = {"true", "1", "sim", "yes", "y", "verdadeiro"}
    false_values = {"false", "0", "nao", "não", "no", "n", "falso"}

    if value_str in true_values:
        return True
    if value_str in false_values:
        return False

    return pd.NA


def parse_goal_minutes(col):
    parsed = []

    for v in col:
        if pd.isna(v) or str(v).strip() in ("[]", "", "nan", "None"):
            parsed.append([])
            continue

        try:
            parsed_list = ast.literal_eval(str(v).strip())

            if not isinstance(parsed_list, list):
                parsed.append([])
                continue

            int_list = []
            for x in parsed_list:
                x_str = str(x).strip()

                if "+" in x_str:
                    parts = x_str.split("+")
                    total = sum(int(p.strip()) for p in parts if p.strip().isdigit())
                    int_list.append(total)
                else:
                    int_list.append(int(float(x_str)))

            parsed.append(int_list)

        except Exception as e:
            print(f"[AVISO] Erro ao converter minutos '{v}': {e}")
            parsed.append([])

    return parsed


def convert_int_column(col_data, col_name):
    numeric = pd.to_numeric(col_data, errors="coerce")

    # Caso especial: se vier decimal indevido, mantém como float para não quebrar
    non_null = numeric.dropna()
    if not non_null.empty and (non_null % 1 != 0).any():
        print(f"[AVISO] Coluna {col_name} tem valores não inteiros; convertida para float")
        return numeric.astype("float64")

    return numeric.astype("Int64")


def convert_round_column(col_data):
    # Extrai apenas o número da rodada se vier algo como "Rodada 10", "10ª Rodada", etc.
    extracted = col_data.astype(str).str.extract(r"(\d+)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    return numeric.astype("Int64")


def convert_game_id_column(col_data):
    numeric = pd.to_numeric(col_data, errors="coerce")
    non_null = numeric.dropna()

    if not non_null.empty and (non_null % 1 != 0).any():
        print("[AVISO] Game_ID contém valores decimais; convertendo para float")
        return numeric.astype("float64")

    return numeric.astype("Int64")


# --- Valida tipos ---
for original_col, props in schema.items():
    renamed_col = props["rename"]

    if renamed_col not in df.columns:
        continue

    dtype = props["type"]
    col_data = df[renamed_col]

    try:
        if dtype == "int":
            if renamed_col == "Round":
                df[renamed_col] = convert_round_column(col_data)
            elif renamed_col == "Game_ID":
                df[renamed_col] = convert_game_id_column(col_data)
            else:
                df[renamed_col] = convert_int_column(col_data, renamed_col)

        elif dtype == "float":
            df[renamed_col] = pd.to_numeric(col_data, errors="coerce").astype("float64")

        elif dtype == "str":
            # preserva nulos reais em vez de transformar tudo em "nan"
            df[renamed_col] = col_data.where(col_data.isna(), col_data.astype(str).str.strip())

        elif dtype == "date":
            df[renamed_col] = pd.to_datetime(col_data, errors="coerce").dt.date

        elif dtype == "bool":
            df[renamed_col] = col_data.apply(to_bool).astype("boolean")

        else:
            print(f"[AVISO] Tipo não tratado no schema para coluna {renamed_col}: {dtype}")

    except Exception as e:
        print(f"[ERRO] Falha ao converter coluna {renamed_col} para {dtype}: {e}")

# --- Converte colunas de minutos de gols ---
for col_min in ["G_H_Min", "G_A_Min"]:
    if col_min in df.columns:
        df[col_min] = parse_goal_minutes(df[col_min])

# --- Relatório de valores nulos pós-conversão ---
null_summary = df.isnull().sum()
null_summary = null_summary[null_summary > 0]

if not null_summary.empty:
    print("\n[AVISO] Colunas com valores nulos após conversão:")
    print(null_summary.sort_values(ascending=False))

# --- Salva CSV oficial ---
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n[OK] CSV oficial padronizado salvo em: {OUTPUT_FILE}")