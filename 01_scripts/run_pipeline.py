from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="DataFrame is highly fragmented.*"
)

SCRIPTS_BASE = [
    ("00_build_leagues_catalog.py", "0- Catálogo FutPython"),
    ("01_validar_arquivos.py", "1- Validação de Arquivos"),
    ("02_normalize_futpython_to_base.py", "2- Normalização FutPython"),
    ("02_validar_colunas.py", "3- Validação de Colunas"),
    ("03_unificar_dados.py", "4- Unificação de Dados"),
    ("04_dicionario_ligas.py", "5- Dicionário de Ligas"),
    # IMPORTANTE:
    # base_ligas.csv é gerada pela padronização de ligas.
    # O dicionário de times depende desse arquivo.
    # Ordem correta:
    # dicionário de ligas -> padronização de ligas -> dicionário de times -> padronização de times
    ("06_padronizar_ligas.py", "6- Padronização de Ligas"),
    ("05_dicionario_times.py", "7- Dicionário de Times"),
    ("07_padronizar_times.py", "8- Padronização de Times"),
]

LOG_FILE = Path("pipeline.log")
OUTPUT_LOG: list[tuple[str, bool, str]] = []


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env

COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"


def log(msg: str, ok: bool = True, nome_etapa: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if ok else "ERRO"
    cor = COLOR_GREEN if ok else COLOR_RED
    linha = f"[{timestamp}] [{status}] {nome_etapa or 'Geral'} -> {msg}"
    print(f"{cor}{linha}{COLOR_RESET}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")
    OUTPUT_LOG.append((nome_etapa or "Geral", ok, msg))


def log_info(msg: str, nome_etapa: str | None = None) -> None:
    log(f"{COLOR_YELLOW}{msg}{COLOR_RESET}", ok=True, nome_etapa=nome_etapa)


def log_error(msg: str, nome_etapa: str | None = None) -> None:
    log(msg, ok=False, nome_etapa=nome_etapa)



def _stream_subprocess(cmd: list[str]) -> int:
    """Executa subprocesso em streaming para o terminal não ficar cego."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subprocess_env(),
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def rodar_script(script_file: str, nome_exibicao: str, extra_args: list[str] | None = None) -> bool:
    path = Path("01_scripts") / script_file
    if not path.exists():
        log_error(f"Script não encontrado: {path}", nome_exibicao)
        return False

    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd.extend(extra_args)

    log_info("Iniciando etapa...", nome_exibicao)
    start_time = datetime.now()
    returncode = _stream_subprocess(cmd)
    duracao = (datetime.now() - start_time).total_seconds()

    if returncode != 0:
        log_error(f"Erro ao executar {nome_exibicao} (duração: {duracao:.2f}s)", nome_exibicao)
        return False

    log_info(f"{nome_exibicao} concluído com sucesso (duração: {duracao:.2f}s)", nome_exibicao)
    return True


def rodar_comando(cmd: list[str], nome_exibicao: str) -> bool:
    log_info("Iniciando etapa...", nome_exibicao)
    start_time = datetime.now()
    returncode = _stream_subprocess(cmd)
    duracao = (datetime.now() - start_time).total_seconds()
    if returncode != 0:
        log_error(f"Erro ao executar {nome_exibicao} (duração: {duracao:.2f}s)", nome_exibicao)
        return False
    log_info(f"{nome_exibicao} concluído com sucesso (duração: {duracao:.2f}s)", nome_exibicao)
    return True


def build_fetch_args(args: argparse.Namespace) -> list[str]:
    fetch_args: list[str] = []
    for attr, flag in [("country", "--country"), ("liga", "--liga"), ("season", "--season")]:
        value = getattr(args, attr, None)
        if value:
            fetch_args.extend([flag, str(value)])

    # --fetch-all significa usar todo o catálogo filtrado. Não envia --limit.
    if getattr(args, "limit", None) and not getattr(args, "fetch_all", False):
        fetch_args.extend(["--limit", str(args.limit)])

    if getattr(args, "dry_run", False):
        fetch_args.append("--dry-run")
    if getattr(args, "force", False):
        fetch_args.append("--force")
    if getattr(args, "incremental", False):
        fetch_args.append("--incremental")
    if getattr(args, "sleep", None) is not None:
        fetch_args.extend(["--sleep", str(args.sleep)])
    if getattr(args, "timeout", None) is not None:
        fetch_args.extend(["--timeout", str(args.timeout)])
    return fetch_args


def _coalescer_colunas_duplicadas(df):
    import pandas as pd
    if not df.columns.duplicated().any():
        return df.copy()
    novo = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        bloco = df.loc[:, df.columns == col]
        if bloco.shape[1] == 1:
            novo[col] = bloco.iloc[:, 0]
        else:
            novo[col] = bloco.replace(r"^\s*$", pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
    return novo


def _contar_gols_ht(valor) -> float:
    """Conta gols no 1º tempo a partir da coluna de minutos.

    Aceita formatos comuns vindos da API, por exemplo:
    - "12, 45+2, 78"
    - "12;45+2;78"
    - "[12, 45+2, 78]"
    - "12' 45+2' 78'"

    Regra: qualquer gol com minuto base <= 45 conta como HT.
    Assim, 45+1, 45+2 etc. também entram como gol do primeiro tempo.
    """
    if valor is None:
        return 0.0

    texto = str(valor).strip()
    if not texto or texto.lower() in {"nan", "none", "-", "[]", "{}"}:
        return 0.0

    count = 0

    # Captura minutos mesmo quando vêm com colchetes, aspas ou acréscimos.
    # Ex.: "45+2" captura minuto_base=45.
    for match in re.finditer(r"(?<!\d)(\d{1,3})(?:\s*\+\s*\d{1,2})?", texto):
        try:
            minuto_base = int(match.group(1))
        except Exception:
            continue

        # Ignora valores impossíveis e conta acréscimos do 1º tempo como 45.
        if 1 <= minuto_base <= 45:
            count += 1

    return float(count)


def _tem_minutos_validos(serie) -> bool:
    """Indica se a coluna de minutos tem informação útil para recalcular HT."""
    try:
        return serie.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "[]": "", "{}": "", "-": ""}).ne("").any()
    except Exception:
        return False


def _converter_tipo(serie, tipo: str):
    import pandas as pd
    tipo = (tipo or "str").lower()
    if tipo in {"int", "integer"}:
        return pd.to_numeric(serie, errors="coerce").fillna(0).round().astype("int64")
    if tipo in {"float", "double", "number"}:
        return pd.to_numeric(serie, errors="coerce")
    if tipo in {"bool", "boolean"}:
        return serie.fillna(False).astype(str).str.lower().isin(["true", "1", "sim", "yes", "y"])
    if tipo in {"date", "datetime"}:
        dt = pd.to_datetime(serie, errors="coerce", dayfirst=True, format="mixed")
        return dt.dt.strftime("%Y-%m-%d").fillna(serie.fillna("").astype(str).str.strip())
    return serie.fillna("").astype(str).str.strip()


def preparar_base_oficial() -> bool:
    """Cria data/base_oficial.csv no contrato esperado pelo backtest e ML.

    Fluxo: base_times_padronizados.csv -> aliases FutPython -> schema.json -> base_oficial.csv.
    Mantém colunas extras para análise, mas garante as colunas curtas usadas pelo projeto legado.
    """
    nome = "8- Base Oficial Backtest"
   
    ROOT_DIR = Path(__file__).resolve().parents[1]
    origem = ROOT_DIR / "data" / "base_times_padronizados.csv"
    destino = ROOT_DIR / "data" / "base_oficial.csv"
    schema_path = ROOT_DIR / "data" / "schema.json"
    if not origem.exists():
        log_error(f"Arquivo não encontrado: {origem}", nome)
        return False

    try:
        import pandas as pd

        df = pd.read_csv(origem, encoding="utf-8-sig", low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        df = _coalescer_colunas_duplicadas(df)

        mapa = {str(c).lower().strip(): c for c in df.columns}

        def pegar(possiveis: list[str]):
            for col in possiveis:
                real = mapa.get(str(col).lower().strip())
                if real is not None:
                    valor = df.loc[:, real]
                    if isinstance(valor, pd.DataFrame):
                        valor = valor.replace(r"^\s*$", pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
                    return valor
            return None

        aliases = {
            "League": ["League", "Liga", "liga", "liga"],
            "League_padronizada": ["League_padronizada", "League_std", "Liga_padronizada", "liga_std", "League"],
            "country": ["country", "Country", "Pais", "País", "pais"],
            "Nº": ["Nº", "Num", "No", "Numero", "Número"],
            "Id_Jogo": ["Id_Jogo", "Game_ID", "Match_ID", "ID_Jogo", "game_id", "fixture_id", "Fixture_ID"],
            "Season": ["Season", "Temporada", "season", "temporada"],
            "Date": ["Date", "Data", "date", "data"],
            "Rodada": ["Rodada", "Round", "round", "rodada"],
            "Home": ["Home_padronizado", "Home_std", "Home", "Mandante", "Home_Team", "Home Team", "home", "home_team"],
            "Away": ["Away_padronizado", "Away_std", "Away", "Visitante", "Away_Team", "Away Team", "away", "away_team"],

            "Goals_H_HT": ["Goals_H_HT", "G_H_HT", "Home_Goals_HT"],
            "Goals_A_HT": ["Goals_A_HT", "G_A_HT", "Away_Goals_HT"],
            "TotalGoals_HT": ["TotalGoals_HT", "TG_HT", "Total_Goals_HT"],
            "Goals_H_FT": ["Goals_H_FT", "G_H_FT", "Home_Score", "Home_Goals_FT", "FTHG", "HG"],
            "Goals_A_FT": ["Goals_A_FT", "G_A_FT", "Away_Score", "Away_Goals_FT", "FTAG", "AG"],
            "TotalGoals_FT": ["TotalGoals_FT", "TG_FT", "Total_Goals_FT"],
            "Goals_H_Minutes": ["Goals_H_Minutes", "G_H_Min", "Min_Goals_Home", "Home_Goals_Minutes"],
            "Goals_A_Minutes": ["Goals_A_Minutes", "G_A_Min", "Min_Goals_Away", "Away_Goals_Minutes"],

            "Odd_H_HT": ["Odd_H_HT", "O_H_HT", "Odd_1_HT"],
            "Odd_D_HT": ["Odd_D_HT", "O_D_HT", "Odd_X_HT"],
            "Odd_A_HT": ["Odd_A_HT", "O_A_HT", "Odd_2_HT"],
            "Odd_Over05_HT": ["Odd_Over05_HT", "O_05_HT", "Over_HT_0_5"],
            "Odd_Under05_HT": ["Odd_Under05_HT", "U_05_HT", "Under_HT_0_5"],
            "Odd_Over15_HT": ["Odd_Over15_HT", "O_15_HT", "Over_HT_1_5"],
            "Odd_Under15_HT": ["Odd_Under15_HT", "U_15_HT", "Under_HT_1_5"],
            "Odd_Over25_HT": ["Odd_Over25_HT", "O_25_HT", "Over_HT_2_5"],
            "Odd_Under25_HT": ["Odd_Under25_HT", "U_25_HT", "Under_HT_2_5"],
            "Odd_H_FT": ["Odd_H_FT", "O_H_FT", "Odd_1_FT"],
            "Odd_D_FT": ["Odd_D_FT", "O_D_FT", "Odd_X_FT"],
            "Odd_A_FT": ["Odd_A_FT", "O_A_FT", "Odd_2_FT"],
            "Odd_Over05_FT": ["Odd_Over05_FT", "O_05_FT", "Over_FT_0_5"],
            "Odd_Under05_FT": ["Odd_Under05_FT", "U_05_FT", "Under_FT_0_5"],
            "Odd_Over15_FT": ["Odd_Over15_FT", "O_15_FT", "Over_FT_1_5"],
            "Odd_Under15_FT": ["Odd_Under15_FT", "U_15_FT", "Under_FT_1_5"],
            "Odd_Over25_FT": ["Odd_Over25_FT", "O_25_FT", "Over_FT_2_5"],
            "Odd_Under25_FT": ["Odd_Under25_FT", "U_25_FT", "Under_FT_2_5"],
            "Odd_BTTS_Yes": ["Odd_BTTS_Yes", "O_BTTS_Y", "BTTS_Yes"],
            "Odd_BTTS_No": ["Odd_BTTS_No", "O_BTTS_N", "BTTS_No"],
            "Odd_DC_1X": ["Odd_DC_1X", "O_DC_1X", "DC_1X"],
            "Odd_DC_12": ["Odd_DC_12", "O_DC_12", "DC_12"],
            "Odd_DC_X2": ["Odd_DC_X2", "O_DC_X2", "DC_X2"],

            "PPG_Home_Pre": ["PPG_Home_Pre", "PPG_H_Pre"],
            "PPG_Away_Pre": ["PPG_Away_Pre", "PPG_A_Pre"],
            "PPG_Home": ["PPG_Home", "PPG_H"],
            "PPG_Away": ["PPG_Away", "PPG_A"],
            "XG_Home_Pre": ["XG_Home_Pre", "XG_H_Pre"],
            "XG_Away_Pre": ["XG_Away_Pre", "XG_A_Pre"],
            "XG_Total_Pre": ["XG_Total_Pre", "XG_T_Pre"],

            "ShotsOnTarget_H": ["ShotsOnTarget_H", "SOT_H", "Shots_On_Target_Home_FT"],
            "ShotsOnTarget_A": ["ShotsOnTarget_A", "SOT_A", "Shots_On_Target_Away_FT"],
            "ShotsOffTarget_H": ["ShotsOffTarget_H", "SOF_H", "Shots_Off_Target_Home_FT"],
            "ShotsOffTarget_A": ["ShotsOffTarget_A", "SOF_A", "Shots_Off_Target_Away_FT"],
            "Shots_H": ["Shots_H", "SH_H", "Total_Shots_Home_FT"],
            "Shots_A": ["Shots_A", "SH_A", "Total_Shots_Away_FT"],
            "Corners_H_FT": ["Corners_H_FT", "C_H_FT", "Corners_Home_FT"],
            "Corners_A_FT": ["Corners_A_FT", "C_A_FT", "Corners_Away_FT"],
            "TotalCorners_FT": ["TotalCorners_FT", "TC_FT"],
            "Odd_Corners_H": ["Odd_Corners_H", "O_C_H"],
            "Odd_Corners_D": ["Odd_Corners_D", "O_C_D"],
            "Odd_Corners_A": ["Odd_Corners_A", "O_C_A"],
            "Odd_Corners_Over75": ["Odd_Corners_Over75", "O_C_O75"],
            "Odd_Corners_Under75": ["Odd_Corners_Under75", "U_C_U75"],
            "Odd_Corners_Over85": ["Odd_Corners_Over85", "O_C_O85"],
            "Odd_Corners_Under85": ["Odd_Corners_Under85", "U_C_U85"],
            "Odd_Corners_Over95": ["Odd_Corners_Over95", "O_C_O95"],
            "Odd_Corners_Under95": ["Odd_Corners_Under95", "U_C_U95"],
            "Odd_Corners_Over105": ["Odd_Corners_Over105", "O_C_O105"],
            "Odd_Corners_Under105": ["Odd_Corners_Under105", "U_C_U105"],
            "Odd_Corners_Over115": ["Odd_Corners_Over115", "O_C_O115"],
            "Odd_Corners_Under115": ["Odd_Corners_Under115", "U_C_U115"],

            "arquivo_origem": ["arquivo_origem", "File_Origin"],
            "liga_arquivo": ["liga_arquivo", "League_File"],
            "Home_padronizado": ["Home_padronizado", "Home_std"],
            "Away_padronizado": ["Away_padronizado", "Away_std"],
            "Home_novo": ["Home_novo", "Home_new"],
            "Away_novo": ["Away_novo", "Away_new"],
        }

        dados_pre_schema: dict[str, object] = {}
        for destino_col, possiveis in aliases.items():
            valor = pegar(possiveis)
            if valor is not None:
                dados_pre_schema[destino_col] = valor

        pre = pd.DataFrame(dados_pre_schema, index=df.index).copy()

        # HT: usa minutos de gols como fonte preferencial quando existem.
        # Antes, o pipeline só recalculava HT se Goals_H_HT/Goals_A_HT estivessem vazios.
        # Se a API trouxesse 0 como default, o sistema aceitava esses zeros e gerava
        # uma base com Under 0.5 HT inflado.
        if "Goals_H_Minutes" in pre.columns and _tem_minutos_validos(pre["Goals_H_Minutes"]):
            pre["Goals_H_HT"] = pre["Goals_H_Minutes"].apply(_contar_gols_ht)
        elif "Goals_H_HT" not in pre.columns:
            pre["Goals_H_HT"] = 0

        if "Goals_A_Minutes" in pre.columns and _tem_minutos_validos(pre["Goals_A_Minutes"]):
            pre["Goals_A_HT"] = pre["Goals_A_Minutes"].apply(_contar_gols_ht)
        elif "Goals_A_HT" not in pre.columns:
            pre["Goals_A_HT"] = 0

        # Recalcula totais sempre que as colunas-base existem, evitando total antigo/stale.
        if "Goals_H_HT" in pre.columns and "Goals_A_HT" in pre.columns:
            pre["TotalGoals_HT"] = pd.to_numeric(pre["Goals_H_HT"], errors="coerce").fillna(0) + pd.to_numeric(pre["Goals_A_HT"], errors="coerce").fillna(0)
        if "Goals_H_FT" in pre.columns and "Goals_A_FT" in pre.columns:
            pre["TotalGoals_FT"] = pd.to_numeric(pre["Goals_H_FT"], errors="coerce").fillna(0) + pd.to_numeric(pre["Goals_A_FT"], errors="coerce").fillna(0)
        if "Corners_H_FT" in pre.columns and "Corners_A_FT" in pre.columns:
            pre["TotalCorners_FT"] = pd.to_numeric(pre["Corners_H_FT"], errors="coerce").fillna(0) + pd.to_numeric(pre["Corners_A_FT"], errors="coerce").fillna(0)

        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        else:
            schema = {}

        canonical_data: dict[str, object] = {}
        for origem_col, cfg in schema.items():
            rename = cfg.get("rename", origem_col) if isinstance(cfg, dict) else origem_col
            tipo = cfg.get("type", "str") if isinstance(cfg, dict) else "str"
            if origem_col in pre.columns:
                canonical_data[rename] = _converter_tipo(pre[origem_col], tipo)

        # Garante colunas essenciais mesmo se schema estiver incompleto.
        fallback_renames = {
            "League": "League", "League_padronizada": "League_std", "country": "Country",
            "Id_Jogo": "Game_ID", "Season": "Season", "Date": "Date", "Rodada": "Round",
            "Home": "Home", "Away": "Away", "Goals_H_HT": "G_H_HT", "Goals_A_HT": "G_A_HT",
            "TotalGoals_HT": "TG_HT", "Goals_H_FT": "G_H_FT", "Goals_A_FT": "G_A_FT",
            "TotalGoals_FT": "TG_FT", "Goals_H_Minutes": "G_H_Min", "Goals_A_Minutes": "G_A_Min",
            "Odd_H_FT": "O_H_FT", "Odd_D_FT": "O_D_FT", "Odd_A_FT": "O_A_FT",
            "Odd_Over25_FT": "O_25_FT", "Odd_Under25_FT": "U_25_FT",
            "ShotsOnTarget_H": "SOT_H", "ShotsOnTarget_A": "SOT_A", "ShotsOffTarget_H": "SOF_H",
            "ShotsOffTarget_A": "SOF_A", "Shots_H": "SH_H", "Shots_A": "SH_A",
            "Corners_H_FT": "C_H_FT", "Corners_A_FT": "C_A_FT", "TotalCorners_FT": "TC_FT",
            "Odd_Corners_H": "O_C_H", "Odd_Corners_D": "O_C_D", "Odd_Corners_A": "O_C_A",
            "Odd_Corners_Over75": "O_C_O75", "Odd_Corners_Over85": "O_C_O85", "Odd_Corners_Over95": "O_C_O95",
            "arquivo_origem": "File_Origin", "liga_arquivo": "League_File",
            "Home_padronizado": "Home_std", "Away_padronizado": "Away_std", "Home_novo": "Home_new", "Away_novo": "Away_new",
        }
        for origem_col, rename in fallback_renames.items():
            if rename not in canonical_data and origem_col in pre.columns:
                canonical_data[rename] = pre[origem_col]

        # Mantém colunas extras originais que podem ajudar em análise futura.
        usados = {str(c).lower().strip() for poss in aliases.values() for c in poss}
        for col in df.columns:
            if col not in canonical_data and str(col).lower().strip() not in usados:
                canonical_data[col] = df[col]

        canonical = pd.DataFrame(canonical_data, index=df.index).copy()
        canonical = _coalescer_colunas_duplicadas(canonical)

        for col in ["League", "League_std", "Country", "Season", "Date", "Round", "Home", "Away"]:
            if col not in canonical.columns:
                canonical[col] = ""
            canonical[col] = canonical[col].fillna("").astype(str).str.strip()

        if "Date" in canonical.columns:
            dt = pd.to_datetime(canonical["Date"], errors="coerce", dayfirst=True, format="mixed")
            canonical["Date"] = dt.dt.strftime("%Y-%m-%d").fillna(canonical["Date"].astype(str).str.strip())

        if "Game_ID" not in canonical.columns or canonical["Game_ID"].fillna("").astype(str).str.strip().eq("").all():
            canonical["Game_ID"] = canonical[["Date", "Country", "League_std", "Home", "Away"]].fillna("").astype(str).agg("__".join, axis=1)
        canonical["Game_ID"] = canonical["Game_ID"].fillna("").astype(str).str.strip()

        for a, b, total in [("G_H_HT", "G_A_HT", "TG_HT"), ("G_H_FT", "G_A_FT", "TG_FT"), ("C_H_FT", "C_A_FT", "TC_FT")]:
            if total not in canonical.columns and a in canonical.columns and b in canonical.columns:
                canonical[total] = pd.to_numeric(canonical[a], errors="coerce").fillna(0) + pd.to_numeric(canonical[b], errors="coerce").fillna(0)

        destino.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(str(destino), index=False, encoding="utf-8-sig")
        log_info(f"Base oficial criada em {destino} com {len(canonical)} linhas e {len(canonical.columns)} colunas", nome)
        log_info("Game_ID disponível para odds/backtest", nome)
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log_error(f"Erro ao preparar base oficial: {repr(exc)}", nome)
        return False


def rodar_transformacao_eventos() -> bool:
    nome = "9- Eventos Backtest"
    path = Path("02_validation") / "02_transformar_long.py"
    if not path.exists():
        log_error(f"Script não encontrado: {path}", nome)
        return False
    log_info("Gerando data/eventos/*.csv para o runner do backtest...", nome)
    start_time = datetime.now()
    resultado = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    duracao = (datetime.now() - start_time).total_seconds()
    if resultado.stdout:
        print(resultado.stdout)
    if resultado.stderr:
        print(resultado.stderr)
    if resultado.returncode != 0:
        log_error(f"Erro ao gerar eventos do backtest (duração: {duracao:.2f}s)", nome)
        return False
    log_info(f"Eventos de backtest gerados com sucesso (duração: {duracao:.2f}s)", nome)
    return True


def rodar_backtest() -> bool:
    return rodar_comando([sys.executable, str(Path("03_backtest") / "runner.py")], "10- Backtest")


def rodar_ml() -> bool:
    etapas = [
        ([sys.executable, str(Path("04_ml") / "01_dataset_builder.py")], "11- ML Dataset Builder"),
        ([sys.executable, str(Path("04_ml") / "02_train_model.py")], "12- ML Train Model"),
    ]
    for cmd, nome in etapas:
        if not rodar_comando(cmd, nome):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline local para treino de ML/backtesting com FutPython.")
    parser.add_argument("--fetch", action="store_true", help="Baixa bases da FutPython antes de processar")
    parser.add_argument("--fetch-all", action="store_true", help="Baixa todas as ligas/temporadas do catálogo filtrado")
    parser.add_argument("--dry-run", action="store_true", help="Mostra URLs da FutPython sem baixar")
    parser.add_argument("--country", help="Filtro por país")
    parser.add_argument("--liga", help="Filtro por liga")
    parser.add_argument("--season", help="Filtro por temporada")
    parser.add_argument("--limit", type=int, help="Limite para teste; ignorado com --fetch-all")
    parser.add_argument("--force", action="store_true", help="Sobrescreve arquivos já baixados")
    parser.add_argument("--incremental", action="store_true", help="Atualiza arquivos existentes via upsert incremental em vez de pular")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pausa entre downloads")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout HTTP em segundos")
    parser.add_argument("--skip-processing", action="store_true", help="Apenas monta catálogo/baixa, sem rodar normalização")
    parser.add_argument("--run-backtest", action="store_true", help="Roda o backtest ao final do pipeline")
    parser.add_argument("--run-ml", action="store_true", help="Roda dataset builder + treino de ML após o pipeline/backtest")
    args = parser.parse_args()

    LOG_FILE.write_text("", encoding="utf-8")

    if not rodar_script("00_build_leagues_catalog.py", "0- Catálogo FutPython"):
        return

    if args.fetch or args.fetch_all or args.dry_run:
        if args.fetch_all and args.limit:
            log_info("--fetch-all informado: --limit será ignorado para baixar tudo do filtro.", "Download FutPython")
        if not rodar_script("01_fetch_futpython_leagues.py", "Download FutPython", build_fetch_args(args)):
            return

    if args.dry_run or args.skip_processing:
        print("\nPipeline finalizado sem processamento dos dados.")
        return

    pipeline_ok = True
    for script_file, nome_exibicao in SCRIPTS_BASE[1:]:
        if not rodar_script(script_file, nome_exibicao):
            log_error("Pipeline interrompido devido a erro.", "Pipeline")
            pipeline_ok = False
            break

    if pipeline_ok and not preparar_base_oficial():
        log_error("Pipeline interrompido ao preparar base oficial.", "Pipeline")
        pipeline_ok = False

    if pipeline_ok and not rodar_transformacao_eventos():
        log_error("Pipeline interrompido ao gerar eventos para backtest.", "Pipeline")
        pipeline_ok = False

    if pipeline_ok and (args.run_backtest or args.run_ml):
        pipeline_ok = rodar_backtest()

    if pipeline_ok and args.run_ml:
        pipeline_ok = rodar_ml()

    print("\nPipeline concluído." if pipeline_ok else "\nPipeline concluído com erro.")
    raise SystemExit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
