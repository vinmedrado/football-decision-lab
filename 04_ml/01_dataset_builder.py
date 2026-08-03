import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
#from controles.mercados.status import get_mercado_lifecycle
from controles.mercados.elegibilidade_liga import (
    ELEGIBILIDADE_CSV_PATH,
    calcular_elegibilidade as _recalcular_elegibilidade_liga,
)
BACKTEST_DIR = Path(os.getenv("ML_BACKTEST_RESULTS_DIR", str(ROOT_DIR / "03_backtest" / "results")))
OUTPUT_DIR = Path(os.getenv("ML_DATASET_OUTPUT_DIR", str(BASE_DIR / "datasets")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESUMO_FILE = BACKTEST_DIR / "resumo.csv"


def env_float(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        print(f"[AVISO] Valor inválido para {name}={value!r}. Usando {default}.")
        return float(default)


def env_int(name: str, default: str) -> int:
    value = os.getenv(name, default)
    try:
        return int(float(str(value).replace(",", ".")))
    except Exception:
        print(f"[AVISO] Valor inválido para {name}={value!r}. Usando {default}.")
        return int(default)


MIN_ROI = env_float("ML_MIN_ROI", "0.03")
MIN_APOSTAS = env_int("ML_MIN_APOSTAS", "1000")
ML_MAX_ROWS = env_int("ML_MAX_ROWS", "0")  # 0 = sem limite
ML_ONLY_MARKET = os.getenv("ML_ONLY_MARKET", "").strip()
TRAIN_SPLIT_PCT = env_float("ML_TRAIN_SPLIT", "0.8")
CALIBRATION_PCT = env_float("ML_CALIBRATION_PCT", "0.2")
MIN_CALIBRATION_SAMPLES = env_int("ML_MIN_CALIBRATION_SAMPLES", "100")
VALIDATION_PCT = env_float("ML_VALIDATION_PCT", "0.2")
MIN_VALIDATION_SAMPLES = env_int("ML_MIN_VALIDATION_SAMPLES", "100")

# Elegibilidade por liga: em vez de aceitar/rejeitar um mercado inteiro pelo
# ROI agregado (ex: DC12 pode ser ótimo em algumas ligas e péssimo em outras),
# um mercado que falha no corte global ainda pode entrar se tiver pelo menos
# uma liga ATIVA com amostra própria suficiente. Dentro do dataset, linhas de
# ligas marcadas BLOQUEADA para aquele mercado são removidas do treino.
# Recalculado a cada execução (não é um filtro fixo no código).
ML_USE_ELEGIBILIDADE_LIGA = os.getenv("ML_USE_ELEGIBILIDADE_LIGA", "1").strip().lower() in {"1", "true", "sim", "s", "yes", "y"}
MIN_APOSTAS_LIGA_PARA_RESGATAR_MERCADO = env_int("ML_MIN_APOSTAS_LIGA_RESGATE", "30")

# Exige que o backtest contextual tenha gerado features contextuais.
# Isso impede o ML de voltar silenciosamente para um modelo global puro.
ML_REQUIRE_CONTEXT_FEATURES = os.getenv("ML_REQUIRE_CONTEXT_FEATURES", "1").strip().lower() in {"1", "true", "sim", "s", "yes", "y"}
MIN_CONTEXT_FEATURES = env_int("ML_MIN_CONTEXT_FEATURES", "5")
CONTEXT_FEATURE_PREFIXES = ("context_", "ctx_")

CATEGORICAL_COLS = ["Home", "Away", "League_std"]

DROP_COLS = {
    "signal", "apostar", "retorno", "odd", "value", "Game_ID",
    "zscore_liga", "zscore_league", "zscore_Home", "zscore_Away", "zscore",
    "z_3", "z_5", "z_7", "z_10",
}

# Colunas com resultado do próprio jogo, estatísticas pós-jogo ou alvos derivados
# não podem entrar como features. O target do mercado atual já é removido
# separadamente, mas outros mercados/eventos do mesmo CSV também precisam sair
# para evitar feature leakage.
LEAKAGE_EXACT_COLS = {
    "FT_Home_Goals", "FT_Away_Goals", "HT_Home_Goals", "HT_Away_Goals",
    "Home_Goals", "Away_Goals", "home_goals", "away_goals",
    "Goals_H", "Goals_A", "Gols_H", "Gols_A",
    "Resultado", "Result", "resultado", "result",
    "Score", "score", "Placar", "placar",
    "Winner", "winner", "Vencedor", "vencedor",
}

LEAKAGE_PREFIXES = (
    "G_H_", "G_A_", "TG_", "BTTS", "TC_", "C_H_", "C_A_",
    "resultado_", "result_", "score_", "placar_",
)

LEAKAGE_KEYWORDS = (
    "final_goal", "goals_final", "gols_final", "fulltime", "full_time",
    "post_match", "pos_jogo", "pós_jogo", "stats_match", "match_stats",
)


# Colunas financeiras/operacionais são geradas por backtest/gestão de banca
# e não representam informação pré-jogo disponível ao modelo. Portanto,
# nunca podem entrar em X_train, X_test ou feature_columns.pkl.
FINANCIAL_OPERATIONAL_KEYWORDS = (
    # Gestão de banca / sizing
    "stake",
    "stake_pct",
    "bank",
    "banca",
    "saldo",
    "drawdown",
    "kelly",

    # Resultado financeiro / pós-aposta
    "retorno",
    "retorno_unitario",
    "lucro",
    "profit",
    "pnl",
    "roi",
    "yield",
    "ev",
    "edge",
    "odd_result",
    "result_bet",
    "green",
    "red",
    "win_bet",
    "loss_bet",
    "aposta",
    "apostar",

    # Artefatos operacionais do backtest/walk-forward
    "wf_",
    "wf_train_games",
    "wf_probability",
    "wf_ready",
    "probabilidade_valida",
    "odd_valida",
)

INF_NAN_WARN_THRESHOLD = 0.05


def read_csv_fast(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def split_by_date_groups(
    df: pd.DataFrame,
    train_fraction: float,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Divide sem permitir que uma data atravesse a fronteira temporal."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction deve estar entre 0 e 1: {train_fraction}")
    if date_col not in df.columns:
        raise ValueError(f"coluna temporal ausente: {date_col}")

    ordered = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    date_counts = ordered.groupby(date_col, sort=True).size()
    if len(date_counts) < 2:
        raise ValueError("split temporal exige pelo menos duas datas distintas")

    cumulative = date_counts.cumsum().iloc[:-1]
    target_rows = len(ordered) * train_fraction
    cutoff_date = (cumulative - target_rows).abs().idxmin()

    train = ordered.loc[ordered[date_col] <= cutoff_date].copy()
    test = ordered.loc[ordered[date_col] > cutoff_date].copy()
    if train.empty or test.empty:
        raise ValueError("split por data resultou em conjunto vazio")
    if set(train[date_col].unique()).intersection(test[date_col].unique()):
        raise AssertionError("leakage temporal: uma data apareceu nos dois conjuntos")
    if train[date_col].max() >= test[date_col].min():
        raise AssertionError("ordem temporal invalida entre os conjuntos")

    return train, test, pd.Timestamp(test[date_col].min())


def deduplicate_matches(
    df: pd.DataFrame,
    mercado: str,
    target_col: str,
    *,
    emit_log: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Remove repeticoes da mesma partida antes de qualquer split temporal."""
    result = df.copy()
    initial_rows = len(result)

    if "Game_ID" in result.columns:
        game_id = result["Game_ID"].astype("string").str.strip()
        valid_id = game_id.notna() & game_id.ne("") & game_id.str.lower().ne("nan")
        if valid_id.any():
            conflict = (
                result.loc[valid_id]
                .assign(_game_id=game_id.loc[valid_id])
                .groupby("_game_id", dropna=False)[target_col]
                .nunique(dropna=False)
                .gt(1)
            )
            if conflict.any():
                ids = conflict[conflict].index.astype(str).tolist()[:10]
                raise ValueError(f"{mercado}: Game_ID duplicado com targets conflitantes: {ids}")
            duplicate_id = valid_id & game_id.duplicated(keep="first")
            result = result.loc[~duplicate_id].copy()

    match_key = [c for c in ["Date", "League_std", "Home", "Away"] if c in result.columns]
    if len(match_key) == 4:
        conflict = result.groupby(match_key, dropna=False)[target_col].nunique(dropna=False).gt(1)
        if conflict.any():
            keys = [tuple(map(str, key if isinstance(key, tuple) else (key,))) for key in conflict[conflict].index[:10]]
            raise ValueError(f"{mercado}: partida duplicada com targets conflitantes: {keys}")
        result = result.drop_duplicates(subset=match_key, keep="first").copy()

    removed = initial_rows - len(result)
    if emit_log and removed:
        print(f"  Deduplicacao temporal: {removed} linhas duplicadas removidas", flush=True)
    return result, removed


def prepare_market_rows(
    df: pd.DataFrame,
    mercado: str,
    target_col: str,
    blocked_leagues: set[str],
    *,
    emit_log: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Aplica filtros existentes e prepara linhas unicas para os splits."""
    result = df.copy()
    if blocked_leagues and "League_std" in result.columns:
        before = len(result)
        result = result[~result["League_std"].astype(str).isin(blocked_leagues)].copy()
        removed = before - len(result)
        if emit_log and removed:
            print(
                f"  Elegibilidade por liga: {removed} linhas removidas "
                f"({len(blocked_leagues)} ligas BLOQUEADA para este mercado)",
                flush=True,
            )

    if "Date" not in result.columns:
        raise ValueError("coluna Date ausente")
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce", format="%Y-%m-%d")
    result = result.dropna(subset=["Date"]).copy()

    if target_col not in result.columns:
        raise ValueError(f"target '{target_col}' nao encontrado")
    result[target_col] = pd.to_numeric(result[target_col], errors="coerce").fillna(0).astype(int)

    if ML_MAX_ROWS and len(result) > ML_MAX_ROWS:
        result = result.sort_values("Date", kind="mergesort").tail(ML_MAX_ROWS).copy()
        if emit_log:
            print(f"  Amostra aplicada: {len(result)} linhas", flush=True)

    result, duplicates_removed = deduplicate_matches(
        result,
        mercado,
        target_col,
        emit_log=emit_log,
    )
    result = result.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return result, duplicates_removed


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def is_leakage_column(column: str, target_column: str) -> bool:
    """Identifica colunas que carregam resultado/estatística do jogo atual.

    A função é propositalmente conservadora: remove alvos e resultados
    conhecidos, mas preserva odds pré-jogo padronizadas começando com O_.
    """
    col = str(column).strip()
    lower = col.lower()

    if col == target_column or col in LEAKAGE_EXACT_COLS:
        return True
    if col.startswith("O_"):
        return False
    if col.startswith(LEAKAGE_PREFIXES):
        return True
    return any(keyword in lower for keyword in LEAKAGE_KEYWORDS)




def is_contextual_pre_match_feature(column: str) -> bool:
    """Features contextuais geradas no backtest com shift(1), logo são pré-jogo.

    Elas podem conter palavras como roi/winrate/edge, mas representam desempenho
    HISTÓRICO anterior do mercado naquele contexto (liga/time), não resultado
    financeiro da aposta atual. Por isso são permitidas no ML.
    """
    lower = str(column).strip().lower()
    return lower.startswith(CONTEXT_FEATURE_PREFIXES)


def listar_context_features(df: pd.DataFrame) -> list[str]:
    """Lista as features contextuais efetivamente presentes no dataset."""
    return [c for c in df.columns if is_contextual_pre_match_feature(c)]


def validar_context_features_obrigatorias(df: pd.DataFrame, mercado: str) -> list[str]:
    """Garante que o ML realmente consuma contexto.

    Se o backtest não gerou colunas contextuais, o dataset builder para o mercado
    em vez de treinar um modelo global disfarçado de contextual.
    """
    context_cols = listar_context_features(df)
    if ML_REQUIRE_CONTEXT_FEATURES and len(context_cols) < MIN_CONTEXT_FEATURES:
        raise ValueError(
            f"{mercado}: features contextuais insuficientes. "
            f"Encontradas={len(context_cols)} | mínimo={MIN_CONTEXT_FEATURES}. "
            "Rode o backtest contextual novamente e confirme colunas context_* nos CSVs *_ml.csv."
        )
    return context_cols


def is_financial_operational_column(column: str) -> bool:
    """Identifica artefatos financeiros/operacionais que não são pré-jogo."""
    lower = str(column).strip().lower()
    if is_contextual_pre_match_feature(lower):
        return False
    return any(keyword in lower for keyword in FINANCIAL_OPERATIONAL_KEYWORDS)


def remove_financial_operational_features(
    df: pd.DataFrame,
    mercado: str,
    removed_records: list[dict],
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features de backtest/banca antes de persistir datasets de ML."""
    removed_cols = [c for c in df.columns if is_financial_operational_column(c)]
    if not removed_cols:
        return df, []

    print(f"  Removendo colunas operacionais/financeiras do ML: {removed_cols}", flush=True)
    for col in removed_cols:
        removed_records.append({
            "mercado": mercado,
            "removed_column": str(col),
            "motivo": "financial_operational_leakage",
        })
    return df.drop(columns=removed_cols), removed_cols

def assert_temporal_feature_safety(df: pd.DataFrame, mercado: str) -> None:
    """Falha cedo se alguma média móvel histórica parecer não defasada.

    O projeto usa features históricas prontas nos CSVs de backtest. Como não
    recalculamos essas features aqui, a proteção viável nesta etapa é impedir
    colunas com nomes que indicam média rolling sem shift/lag explícito.
    """
    suspicious = []
    safe_tokens = ("shift", "lag", "prev", "previous", "antes", "anterior")
    rolling_tokens = ("rolling", "media_movel", "média_móvel", "avg_last", "mean_last")

    for col in df.columns:
        lower = str(col).lower()
        if is_contextual_pre_match_feature(lower):
            continue
        if any(token in lower for token in rolling_tokens) and not any(token in lower for token in safe_tokens):
            suspicious.append(str(col))

    if suspicious:
        raise ValueError(
            f"Possível feature leakage em {mercado}: médias móveis sem indicação de shift(1): "
            f"{suspicious[:20]}"
        )


# ==============================
# LOAD RESUMO
# ==============================
print("=" * 55, flush=True)
print("Carregando resumo...", flush=True)

if not RESUMO_FILE.exists():
    raise FileNotFoundError(f"Resumo não encontrado: {RESUMO_FILE}")

resumo = read_csv_fast(RESUMO_FILE)

if "mercado" not in resumo.columns and "market" in resumo.columns:
    resumo["mercado"] = resumo["market"]

if "event" not in resumo.columns and "mercado" in resumo.columns:
    resumo["event"] = resumo["mercado"]
resumo_original = len(resumo)

if ML_ONLY_MARKET:
    resumo = resumo[resumo["mercado"].astype(str).eq(ML_ONLY_MARKET)].copy()


## Governança Operacional lifecycle filter (safe-by-default).
## APOSENTADA/BLOQUEADA mercados are excluded from dataset generation.
## OBSERVACAO mercados remain available only as experimental analysis datasets.
#lifecycle_removed = []
#lifecycle_experimental = []
#if "mercado" in resumo.columns:
    #keep_mask = []
    #for mercado_value in resumo["mercado"].astype(str):
        #info = get_mercado_lifecycle(mercado_value)
        #status = str(info.get("status_ciclo_vida") or "DESCONHECIDA").upper()
        #if status in {"APOSENTADA", "BLOQUEADA"}:
            #keep_mask.append(False)
            #lifecycle_removed.append((mercado_value, status, info.get("motivo")))
        #else:
            #keep_mask.append(True)
            #if status == "OBSERVACAO":
                #lifecycle_experimental.append((mercado_value, status, info.get("motivo")))
    #resumo = resumo[pd.Series(keep_mask, index=resumo.index)].copy()

# ------------------------------------------------------------------
# Elegibilidade por liga (recalculada agora, a cada execução deste script,
# a partir dos relatórios de contexto mais recentes gerados pelo backtest).
# ------------------------------------------------------------------
elegibilidade_liga = pd.DataFrame()
if ML_USE_ELEGIBILIDADE_LIGA:
    try:
        elegibilidade_liga = _recalcular_elegibilidade_liga()
        REPORTS_DIR_LOCAL = ELEGIBILIDADE_CSV_PATH.parent
        REPORTS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
        elegibilidade_liga.to_csv(ELEGIBILIDADE_CSV_PATH, index=False, encoding="utf-8-sig")
        print(
            f"Elegibilidade por liga recalculada: {len(elegibilidade_liga)} combinações "
            f"mercado x liga (ATIVA={int((elegibilidade_liga['status'] == 'ATIVA').sum()) if not elegibilidade_liga.empty else 0}).",
            flush=True,
        )
    except Exception as exc:
        print(f"[AVISO] Falha ao recalcular elegibilidade por liga: {exc}", flush=True)
        elegibilidade_liga = pd.DataFrame()

ligas_ativas_por_mercado: dict[str, set[str]] = {}
ligas_bloqueadas_por_mercado: dict[str, set[str]] = {}
if not elegibilidade_liga.empty:
    for mercado_key, grupo in elegibilidade_liga.groupby("mercado"):
        ligas_ativas_por_mercado[mercado_key] = set(
            grupo.loc[
                (grupo["status"] == "ATIVA") & (grupo["apostas"] >= MIN_APOSTAS_LIGA_PARA_RESGATAR_MERCADO),
                "liga",
            ]
        )
        ligas_bloqueadas_por_mercado[mercado_key] = set(grupo.loc[grupo["status"] == "BLOQUEADA", "liga"])

resgatados_por_liga: list[str] = []


def _mercado_tem_liga_elegivel(mercado_value: str) -> bool:
    return bool(ligas_ativas_por_mercado.get(mercado_value))


mask_corte_global = (
    (pd.to_numeric(resumo["roi"], errors="coerce") > MIN_ROI) &
    (pd.to_numeric(resumo["apostas"], errors="coerce") > MIN_APOSTAS)
)

if ML_USE_ELEGIBILIDADE_LIGA and ligas_ativas_por_mercado:
    mask_resgate_por_liga = resumo["mercado"].astype(str).map(_mercado_tem_liga_elegivel)
    resgatados_por_liga = sorted(set(resumo.loc[(~mask_corte_global) & mask_resgate_por_liga, "mercado"].astype(str)))
    mask_final = mask_corte_global | mask_resgate_por_liga
else:
    mask_final = mask_corte_global

resumo = resumo[mask_final].reset_index(drop=True)

if resgatados_por_liga:
    print(
        f"Mercados resgatados por liga elegível (falharam no corte global, mas têm liga ATIVA): "
        f"{len(resgatados_por_liga)} -> {', '.join(resgatados_por_liga[:10])}"
        + (" ..." if len(resgatados_por_liga) > 10 else ""),
        flush=True,
    )

print(f"Mercados no resumo     : {resumo_original}", flush=True)
print(f"Após filtro            : {len(resumo)} (ROI > {MIN_ROI}, apostas > {MIN_APOSTAS})", flush=True)
print(f"Descartados pelo filtro: {resumo_original - len(resumo)}", flush=True)
#if lifecycle_removed:
    #print("Mercados removidos por lifecycle:", flush=True)
    #for mercado, status, motivo in lifecycle_removed:
        #print(f"  - {mercado} — {status} — {motivo}", flush=True)
#if lifecycle_experimental:
    #print("Mercados OBSERVACAO mantidos apenas como dataset experimental:", flush=True)
    #for mercado, status, motivo in lifecycle_experimental:
        #print(f"  - {mercado} — {status} — {motivo}", flush=True)
if ML_ONLY_MARKET:
    print(f"Filtro de mercado      : {ML_ONLY_MARKET}", flush=True)
if ML_MAX_ROWS:
    print(f"Limite por mercado     : {ML_MAX_ROWS} linhas", flush=True)

if len(resumo) == 0:
    print("\nNenhum mercado passou no filtro do ML.", flush=True)
    print(f"Filtro atual: ROI > {MIN_ROI}, apostas > {MIN_APOSTAS}", flush=True)
    save_pickle([], OUTPUT_DIR / "mercados_meta.pkl")
    raise SystemExit(0)

# ==============================
# PREPARA ENCODERS GLOBAIS
# ==============================
print("\n" + "=" * 55, flush=True)
print("Preparando encoders globais...", flush=True)

cat_values = {col: [] for col in CATEGORICAL_COLS}
encoder_fit_rows_by_market = {}

for _, row in resumo.iterrows():
    mercado = str(row["mercado"])
    event_name = str(row["event"])
    file_path = BACKTEST_DIR / f"{mercado}_ml.csv"
    if not file_path.exists():
        continue
    try:
        # Lê só o cabeçalho primeiro para reduzir leitura se possível.
        cols = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig").columns.tolist()
        usecols = [c for c in ["Game_ID", "Date", event_name, *CATEGORICAL_COLS] if c in cols]
        if "Date" not in usecols or event_name not in usecols:
            continue
        df_cat = pd.read_csv(file_path, usecols=usecols, low_memory=False, encoding="utf-8-sig")
        df_cat, _ = prepare_market_rows(
            df_cat,
            mercado,
            event_name,
            ligas_bloqueadas_por_mercado.get(mercado, set())
            if ML_USE_ELEGIBILIDADE_LIGA
            else set(),
            emit_log=False,
        )
        train_cat, _, _ = split_by_date_groups(df_cat, TRAIN_SPLIT_PCT)
        validation_rows = max(int(len(train_cat) * VALIDATION_PCT), MIN_VALIDATION_SAMPLES)
        validation_rows = min(validation_rows, len(train_cat) - 1)
        selection_train_fraction = (len(train_cat) - validation_rows) / len(train_cat)
        selection_train_cat, _, _ = split_by_date_groups(train_cat, selection_train_fraction)

        calibration_rows = max(
            int(len(selection_train_cat) * CALIBRATION_PCT),
            MIN_CALIBRATION_SAMPLES,
        )
        calibration_rows = min(calibration_rows, len(selection_train_cat) - 1)
        base_fraction = (len(selection_train_cat) - calibration_rows) / len(selection_train_cat)
        base_cat, _, _ = split_by_date_groups(selection_train_cat, base_fraction)
    except Exception as exc:
        print(f"  AVISO [{mercado}] fora do ajuste dos encoders: {exc}", flush=True)
        continue

    encoder_fit_rows_by_market[mercado] = len(base_cat)
    for col in CATEGORICAL_COLS:
        if col in base_cat.columns:
            cat_values[col].extend(base_cat[col].dropna().astype(str).unique().tolist())

encoders = {}
encoder_maps = {}
for col, values in cat_values.items():
    if values:
        le = LabelEncoder()
        le.fit(pd.Series(values, dtype="string").dropna().astype(str).unique())
        encoders[col] = le
        encoder_maps[col] = {v: i for i, v in enumerate(le.classes_)}
        print(f"  OK [{col}] {len(le.classes_)} categorias únicas", flush=True)

encoders_path = OUTPUT_DIR / "label_encoders.pkl"
save_pickle(encoders, encoders_path)
print(f"  Encoders salvos em: {encoders_path}", flush=True)
print(
    f"  Ajuste restrito ao treino-base: {sum(encoder_fit_rows_by_market.values())} "
    f"linhas em {len(encoder_fit_rows_by_market)} mercados",
    flush=True,
)

# ==============================
# PROCESSA CADA MERCADO
# ==============================
print("\n" + "=" * 55, flush=True)
print("Gerando datasets por mercado...", flush=True)

rejected = []
mercados_ok = []
removed_feature_records = []
context_feature_records = []

for idx, row in resumo.iterrows():
    t0 = time.time()
    mercado = str(row["mercado"])
    event_name = str(row["event"])
    roi_bt = float(row["roi"])
    winrate_bt = float(row.get("winrate", 0))

    file_path = BACKTEST_DIR / f"{mercado}_ml.csv"
    print(f"\n[{idx + 1}/{len(resumo)}] Processando {mercado} -> target {event_name}", flush=True)

    if not file_path.exists():
        rejected.append((mercado, "arquivo não encontrado"))
        print(f"  ERRO arquivo não encontrado: {file_path}", flush=True)
        continue

    df = read_csv_fast(file_path)
    if event_name not in df.columns:
        rejected.append((mercado, f"target '{event_name}' nao encontrado"))
        print(f"  ERRO target ausente: {event_name}", flush=True)
        continue
    try:
        blocked_leagues = (
            ligas_bloqueadas_por_mercado.get(mercado, set())
            if ML_USE_ELEGIBILIDADE_LIGA
            else set()
        )
        df, duplicates_removed = prepare_market_rows(
            df,
            mercado,
            event_name,
            blocked_leagues,
        )
    except ValueError as exc:
        rejected.append((mercado, str(exc)))
        print(f"  ERRO {exc}", flush=True)
        continue
    df = df[[c for c in df.columns if "Game_ID" not in c]].copy()

    if ML_USE_ELEGIBILIDADE_LIGA and "League_std" in df.columns:
        ligas_bloqueadas = ligas_bloqueadas_por_mercado.get(mercado, set())
        if ligas_bloqueadas:
            linhas_antes = len(df)
            df = df[~df["League_std"].astype(str).isin(ligas_bloqueadas)].copy()
            removidas = linhas_antes - len(df)
            if removidas:
                print(
                    f"  Elegibilidade por liga: {removidas} linhas removidas "
                    f"({len(ligas_bloqueadas)} ligas BLOQUEADA para este mercado)",
                    flush=True,
                )
            if df.empty:
                rejected.append((mercado, "todas as ligas bloqueadas pela elegibilidade"))
                print("  ERRO todas as ligas do mercado estão BLOQUEADA -- nada sobrou para treinar", flush=True)
                continue

    if ML_MAX_ROWS and len(df) > ML_MAX_ROWS:
        df = df.sort_values("Date").tail(ML_MAX_ROWS).copy() if "Date" in df.columns else df.tail(ML_MAX_ROWS).copy()
        print(f"  Amostra aplicada: {len(df)} linhas", flush=True)

    if "Date" not in df.columns:
        rejected.append((mercado, "coluna Date ausente"))
        print("  ERRO coluna Date ausente", flush=True)
        continue

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        format="%Y-%m-%d",
    )
    df = df.dropna(subset=["Date"])

    if df.empty:
        rejected.append((mercado, "sem linhas válidas após limpeza de Date"))
        print("  ERRO sem linhas válidas após Date", flush=True)
        continue

    if event_name not in df.columns:
        rejected.append((mercado, f"target '{event_name}' não encontrado"))
        print(f"  ERRO target ausente: {event_name}", flush=True)
        continue

    df[event_name] = pd.to_numeric(df[event_name], errors="coerce").fillna(0).astype(int)
    if df[event_name].nunique() < 2:
        rejected.append((mercado, "target sem variação"))
        print("  ERRO target sem variação", flush=True)
        continue

    target = df[event_name].to_numpy()

    assert_temporal_feature_safety(df, mercado)

    leakage_cols = [c for c in df.columns if is_leakage_column(c, event_name)]
    cols_to_drop = DROP_COLS | set(leakage_cols) | {event_name}
    cols_to_keep = [c for c in df.columns if c not in cols_to_drop]
    if leakage_cols:
        print(
            f"  Removendo possíveis leakage cols: {leakage_cols[:12]}"
            f"{'...' if len(leakage_cols) > 12 else ''}",
            flush=True,
        )
    df_model = df[cols_to_keep].copy()
    df_model["target"] = target

    # Encoding vetorizado. Evita apply linha a linha, que travava com milhares de categorias.
    for col, mapping in encoder_maps.items():
        if col in df_model.columns:
            df_model[col] = (
                df_model[col]
                .astype(str)
                .fillna("DESCONHECIDA")
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )

    # Remove objetos restantes que não são Date. Isso evita conversões lentas e erro no treino.
    object_cols = [c for c in df_model.select_dtypes(include=["object", "string"]).columns if c != "Date"]
    if object_cols:
        print(f"  Removendo colunas texto não modeláveis: {object_cols[:8]}{'...' if len(object_cols) > 8 else ''}", flush=True)
        df_model.drop(columns=object_cols, inplace=True)

    # Converte numéricos de forma segura.
    feature_cols = [c for c in df_model.columns if c not in ["Date", "target"]]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_model[col]):
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    feature_cols = [c for c in df_model.columns if c not in ["Date", "target"]]
    if not feature_cols:
        rejected.append((mercado, "sem features numéricas"))
        print("  ERRO sem features numéricas", flush=True)
        continue

    numeric_part = df_model[feature_cols]
    total_cells = max(df_model.shape[0] * len(feature_cols), 1)
    inf_count = np.isinf(numeric_part.to_numpy(dtype="float64", copy=False)).sum()
    nan_count = numeric_part.isna().sum().sum()

    if inf_count / total_cells > INF_NAN_WARN_THRESHOLD:
        print(f"  AVISO Infs: {inf_count/total_cells:.1%}", flush=True)
    if nan_count / total_cells > INF_NAN_WARN_THRESHOLD:
        print(f"  AVISO NaNs: {nan_count/total_cells:.1%}", flush=True)

    df_model.replace([np.inf, -np.inf], 0, inplace=True)
    df_model.fillna(0, inplace=True)

    # Remove features combinadas muito ruidosas/pesadas, mantendo o contrato atual.
    df_model = df_model[[c for c in df_model.columns if "minus" not in c and "ratio" not in c]]

    # Última barreira antes de salvar X_train/X_test/feature_columns.pkl:
    # artefatos financeiros e operacionais do backtest/banca não são features pré-jogo.
    df_model, financial_operational_cols = remove_financial_operational_features(
        df_model,
        mercado,
        removed_feature_records,
    )

    try:
        context_cols = validar_context_features_obrigatorias(df_model, mercado)
    except ValueError as exc:
        rejected.append((mercado, str(exc)))
        print(f"  ERRO {exc}", flush=True)
        continue

    context_feature_records.extend([
        {"mercado": mercado, "context_feature": str(col)}
        for col in context_cols
    ])
    if context_cols:
        print(
            f"  Features contextuais preservadas: {len(context_cols)} "
            f"({', '.join(context_cols[:8])}{'...' if len(context_cols) > 8 else ''})",
            flush=True,
        )

    try:
        train, test, split_date = split_by_date_groups(df_model, TRAIN_SPLIT_PCT)
    except ValueError as exc:
        rejected.append((mercado, "split resultou em conjunto vazio"))
        print(f"  ERRO split temporal por data invalido: {exc}", flush=True)
        continue

    X_train = train.drop(columns=["target", "Date"])
    y_train = train["target"]
    X_test = test.drop(columns=["target", "Date"])
    y_test = test["target"]
    dates_train = train[["Date"]].copy()
    dates_test = test[["Date"]].copy()

    if set(dates_train["Date"]).intersection(dates_test["Date"]):
        raise AssertionError(f"{mercado}: compartilhamento de datas entre treino e teste")

    forbidden_after_split = [c for c in X_train.columns if is_financial_operational_column(c)]
    if forbidden_after_split:
        rejected.append((mercado, f"colunas operacionais/financeiras permaneceram nas features: {forbidden_after_split}"))
        print(f"  ERRO colunas operacionais/financeiras permaneceram nas features: {forbidden_after_split}", flush=True)
        continue

    mercado_dir = OUTPUT_DIR / mercado
    mercado_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(mercado_dir / "X_train.csv", index=False)
    y_train.to_csv(mercado_dir / "y_train.csv", index=False)
    X_test.to_csv(mercado_dir / "X_test.csv", index=False)
    y_test.to_csv(mercado_dir / "y_test.csv", index=False)
    dates_train.assign(Date=dates_train["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        mercado_dir / "dates_train.csv", index=False
    )
    dates_test.assign(Date=dates_test["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        mercado_dir / "dates_test.csv", index=False
    )
    save_pickle(list(X_train.columns), mercado_dir / "feature_columns.pkl")

    target_dist = pd.Series(target).value_counts(normalize=True).round(3).to_dict()
    elapsed = time.time() - t0
    print(
        f"  OK {mercado}: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"split={split_date.date()}, ROI_bt={roi_bt:+.3f}, target={target_dist}, tempo={elapsed:.1f}s",
        flush=True,
    )

    mercados_ok.append({
        "mercado": mercado,
        "event": event_name,
        "roi_bt": roi_bt,
        "winrate_bt": winrate_bt,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "split_type": "temporal_grouped_by_date",
        "split_date": str(pd.Timestamp(split_date).date()),
        "train_start": str(pd.Timestamp(train["Date"].min()).date()),
        "train_end": str(pd.Timestamp(train["Date"].max()).date()),
        "test_start": str(pd.Timestamp(test["Date"].min()).date()),
        "test_end": str(pd.Timestamp(test["Date"].max()).date()),
        "n_unique_dates_train": int(train["Date"].nunique()),
        "n_unique_dates_test": int(test["Date"].nunique()),
        "duplicates_removed": int(duplicates_removed),
        "encoder_fit_scope": "base_train_only",
        "leakage_cols_removed": leakage_cols,
        "financial_operational_cols_removed": financial_operational_cols,
        "context_features_count": len(context_cols),
        "context_features": context_cols,
    })

removed_features_report = OUTPUT_DIR / "removed_feature_columns.csv"
removed_df = pd.DataFrame(
    removed_feature_records,
    columns=["mercado", "removed_column", "motivo"],
)
removed_df.to_csv(removed_features_report, index=False, encoding="utf-8-sig")
print(f"\nRelatório de features removidas salvo em: {removed_features_report}", flush=True)

context_features_report = OUTPUT_DIR / "context_feature_columns.csv"
pd.DataFrame(
    context_feature_records,
    columns=["mercado", "context_feature"],
).to_csv(context_features_report, index=False, encoding="utf-8-sig")
print(f"Relatório de features contextuais salvo em: {context_features_report}", flush=True)

# ==============================
# RELATÓRIO
# ==============================
print("\n" + "=" * 55, flush=True)
if rejected:
    print(f"Mercados rejeitados ({len(rejected)}):", flush=True)
    for mercado, motivo in rejected:
        print(f"  - {mercado}: {motivo}", flush=True)

meta_mercados_path = OUTPUT_DIR / "mercados_meta.pkl"
save_pickle(mercados_ok, meta_mercados_path)

print(f"\n{len(mercados_ok)} mercados processados com sucesso!", flush=True)
print(f"Meta salva em: {meta_mercados_path}", flush=True)
print("=" * 55, flush=True)

print("\nResumo final de mercados", flush=True)
summary = pd.DataFrame(mercados_ok)
if not summary.empty:
    view = summary[["mercado", "event", "roi_bt", "n_features", "context_features_count", "n_train", "n_test"]].copy()
    view["roi_bt"] = view["roi_bt"].map(lambda x: f"{x:+.2%}")
    view.rename(columns={
        "mercado": "Mercado",
        "event": "Evento",
        "roi_bt": "ROI Backtest",
        "n_features": "N Features",
        "context_features_count": "Context Features",
        "n_train": "Treino",
        "n_test": "Teste",
    }, inplace=True)
    print(view.to_string(index=False), flush=True)
else:
    print("Nenhum mercado processado para exibir.", flush=True)
print("=" * 55, flush=True)
