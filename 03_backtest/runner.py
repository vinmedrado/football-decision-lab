import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from functions import (
    calcular_media_movel,
    calcular_desvio_padrao,
    calcular_zscore,
    calcular_media_liga,
    calcular_std_liga,
    calcular_zscore_liga,
    calcular_media_movel_grupo,
    calcular_std_grupo,
    calcular_zscore_grupo,
    gerar_sinal_inteligente,
    aplicar_walk_forward_probabilidade,
    calcular_kelly_fractional,
    calcular_metricas_apostas,
)

# ==========================================================
# FOOTBALL LAB — BACKTEST PROFISSIONAL POR ODDS REAIS
# ==========================================================
# Regra central:
# - mercado sem odd real NÃO é backtestado;
# - sem odd_default para apostar;
# - sem escanteio/chutes/cartões fake;
# - odds 1X2 não são usadas como gols por time;
# - cada mercado tem target compatível com a própria odd.
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
BASE_OFICIAL = ROOT_DIR / "data" / "base_oficial.csv"
CONFIG_FILE = BASE_DIR / "config.json"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "backtest_log.txt"
LOG_FILE.write_text("", encoding="utf-8")

WINDOWS = [3, 5, 7, 10]


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "sim", "s", "yes", "y"}


def env_int(name: str, default: str) -> int:
    try:
        return int(float(os.getenv(name, default).replace(",", ".")))
    except Exception:
        return int(default)


def env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default).replace(",", "."))
    except Exception:
        return float(default)


# Config de backtest/ML save
MIN_TRAIN_GAMES = 100
MIN_GAMES_PER_MARKET = 500
USE_FRACTIONAL_KELLY = False
KELLY_FRACTION = 0.25
INITIAL_BANKROLL = 1000.0
FIXED_STAKE = 1.0

if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        backtest_config = cfg.get("backtest", {})
        strategy_config = cfg.get("strategy", {})
        MIN_TRAIN_GAMES = int(backtest_config.get("MIN_TRAIN_GAMES", MIN_TRAIN_GAMES))
        MIN_GAMES_PER_MARKET = int(backtest_config.get("MIN_GAMES_PER_MARKET", MIN_GAMES_PER_MARKET))
        USE_FRACTIONAL_KELLY = bool(backtest_config.get("USE_FRACTIONAL_KELLY", USE_FRACTIONAL_KELLY))
        KELLY_FRACTION = float(backtest_config.get("KELLY_FRACTION", KELLY_FRACTION))
        INITIAL_BANKROLL = float(backtest_config.get("INITIAL_BANKROLL", INITIAL_BANKROLL))
        FIXED_STAKE = float(backtest_config.get("FIXED_STAKE", FIXED_STAKE))
    except Exception:
        strategy_config = {}
else:
    strategy_config = {}

SAVE_ALL_ML_DATASETS = env_bool("BT_SAVE_ALL_ML_DATASETS", "0")
MIN_ML_APOSTAS_TO_SAVE = env_int("BT_MIN_ML_APOSTAS_TO_SAVE", "1000")
MIN_ML_ROI_TO_SAVE = env_float("BT_MIN_ML_ROI_TO_SAVE", "0.03")

# Resgate por liga: mesmo que o mercado inteiro falhe no corte global acima
# (ex: DC12 com ROI agregado ruim), ele ainda é salvo se tiver PELO MENOS UMA
# liga, sozinha, com volume e ROI bons o suficiente -- é essa liga que o
# motor de elegibilidade (04_ml/controles/mercados/elegibilidade_liga.py) e o
# dataset_builder vão aproveitar depois. Sem isso, o arquivo _ml.csv nunca
# chega a existir para esses mercados e o resgate por liga não tem o que ler.
# Embutido no código para não depender de lembrar de setar variável de ambiente.
BT_MIN_APOSTAS_LIGA_RESGATE = env_int("BT_MIN_APOSTAS_LIGA_RESGATE", "30")
BT_MIN_ROI_LIGA_RESGATE = env_float("BT_MIN_ROI_LIGA_RESGATE", "0.03")

# Filtros operacionais do backtest.
# Mantém a análise como simulação histórica: só entra quando existe edge mínimo
# contra a probabilidade implícita da odd e, em mercados de gols, contexto temporal
# compatível com Over/Under.
BT_MIN_CONF_EDGE = env_float("BT_MIN_CONF_EDGE", "0.02")
BT_MIN_EV = env_float("BT_MIN_EV", "0.02")
BT_TOTALS_CONTEXT_Z = env_float("BT_TOTALS_CONTEXT_Z", "0.10")

# Filtro contextual sem leakage: usa apenas histórico anterior por mercado/liga/time.
BT_USE_CONTEXT_FILTER = env_bool("BT_USE_CONTEXT_FILTER", "1")
BT_CONTEXT_MIN_LIGA = env_int("BT_CONTEXT_MIN_LIGA", "10")
BT_CONTEXT_MIN_TIME = env_int("BT_CONTEXT_MIN_TIME", "5")
BT_CONTEXT_MIN_LIGA_TIME = env_int("BT_CONTEXT_MIN_LIGA_TIME", "5")
BT_CONTEXT_BAD_SCORE = env_float("BT_CONTEXT_BAD_SCORE", "-0.10")
BT_CONTEXT_GOOD_SCORE = env_float("BT_CONTEXT_GOOD_SCORE", "0.03")


def log(msg: str, status: str = "info") -> None:
    now = datetime.now().strftime("%H:%M:%S")
    color_reset = "\033[0m"
    colors = {
        "info": "\033[90m",
        "ok": "\033[92m",
        "warn": "\033[93m",
        "erro": "\033[91m",
    }
    prefix = colors.get(status, colors["info"])
    print(f"[{now}] {prefix}{msg}{color_reset}", flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {msg}\n")


def limpar_resultados_antigos() -> None:
    for p in OUTPUT_DIR.glob("*_ml.csv"):
        try:
            p.unlink()
        except OSError:
            pass


def odd_valida(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan) > 1.0


def mercado_exists(df: pd.DataFrame, required_cols: list[str]) -> bool:
    return all(c in df.columns for c in required_cols)




def _parse_goal_minutes(value):
    """Converte colunas G_H_Min/G_A_Min em lista de minutos inteiros.

    A base pode trazer valores como:
    - "[14]"
    - "[45, 62, 90]"
    - "[]"
    - NaN

    Usamos regex para ser tolerante a formatos diferentes.
    """
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    nums = re.findall(r"\d+", text)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            pass
    return out


def _count_first_half_goals(value):
    # Critério conservador: minuto <= 45.
    # Se a fonte trouxer acréscimos como 45+2, a regex captura 45 e 2;
    # por isso, na prática, 45+2 ainda conta como primeiro tempo.
    return sum(1 for m in _parse_goal_minutes(value) if 0 < int(m) <= 45)


def reparar_gols_ht_por_minutos(base: pd.DataFrame) -> pd.DataFrame:
    """Corrige G_H_HT/G_A_HT/TG_HT a partir de G_H_Min/G_A_Min.

    Diagnóstico do projeto:
    a base tinha casos como G_H_HT=0 e G_H_Min="[14]".
    Isso fazia Draw HT e Under HT ficarem artificialmente lucrativos,
    porque gols reais de primeiro tempo estavam sendo tratados como 0x0.

    Regra:
    - Se G_H_Min/G_A_Min existirem, elas são a fonte de verdade para HT.
    - Recalcula G_H_HT, G_A_HT e TG_HT antes do catálogo de mercados.
    """
    if "G_H_Min" not in base.columns or "G_A_Min" not in base.columns:
        log("G_H_Min/G_A_Min ausentes; HT não foi recalculado por minutos.", "warn")
        return base

    old_tg = pd.to_numeric(base.get("TG_HT", pd.Series(0, index=base.index)), errors="coerce").fillna(0)

    base = base.copy()
    base["G_H_HT"] = base["G_H_Min"].apply(_count_first_half_goals).astype(int)
    base["G_A_HT"] = base["G_A_Min"].apply(_count_first_half_goals).astype(int)
    base["TG_HT"] = base["G_H_HT"] + base["G_A_HT"]

    changed = int((old_tg != base["TG_HT"]).sum())
    pct_over_05 = float((base["TG_HT"] > 0).mean()) if len(base) else 0.0
    pct_draw_ht = float((base["G_H_HT"] == base["G_A_HT"]).mean()) if len(base) else 0.0

    log(
        f"HT recalculado por minutos: {changed} linhas alteradas | "
        f"Over 0.5 HT real={pct_over_05:.2%} | Draw HT real={pct_draw_ht:.2%}",
        "ok",
    )
    return base


def target_over_total(df: pd.DataFrame, period: str, line: float) -> pd.Series:
    col = "TG_FT" if period == "FT" else "TG_HT"
    return (pd.to_numeric(df[col], errors="coerce") > line).astype(int)


def target_under_total(df: pd.DataFrame, period: str, line: float) -> pd.Series:
    col = "TG_FT" if period == "FT" else "TG_HT"
    return (pd.to_numeric(df[col], errors="coerce") < line).astype(int)


def target_btts_yes(df: pd.DataFrame) -> pd.Series:
    return ((pd.to_numeric(df["G_H_FT"], errors="coerce") > 0) & (pd.to_numeric(df["G_A_FT"], errors="coerce") > 0)).astype(int)


def target_btts_no(df: pd.DataFrame) -> pd.Series:
    return (1 - target_btts_yes(df)).astype(int)


def target_1x2(df: pd.DataFrame, period: str, side: str) -> pd.Series:
    gh = pd.to_numeric(df["G_H_FT" if period == "FT" else "G_H_HT"], errors="coerce")
    ga = pd.to_numeric(df["G_A_FT" if period == "FT" else "G_A_HT"], errors="coerce")
    if side == "H":
        return (gh > ga).astype(int)
    if side == "D":
        return (gh == ga).astype(int)
    if side == "A":
        return (gh < ga).astype(int)
    raise ValueError(side)


def target_dc(df: pd.DataFrame, kind: str) -> pd.Series:
    gh = pd.to_numeric(df["G_H_FT"], errors="coerce")
    ga = pd.to_numeric(df["G_A_FT"], errors="coerce")
    if kind == "1X":
        return (gh >= ga).astype(int)
    if kind == "12":
        return (gh != ga).astype(int)
    if kind == "X2":
        return (gh <= ga).astype(int)
    raise ValueError(kind)


def target_ah(df: pd.DataFrame, side: str, home_handicap: float) -> pd.Series:
    """Liquidação correta das colunas AH_* da base FutPython.

    Na base, o sufixo da coluna representa o handicap aplicado ao MANDANTE.

    Exemplo:
    - AH_Home_pos_0_5 = aposta no mandante com Home +0.5
    - AH_Away_pos_0_5 = aposta no visitante contra Home +0.5, ou seja, Away -0.5
    - AH_Home_neg_1 = aposta no mandante com Home -1.0
    - AH_Away_neg_1 = aposta no visitante contra Home -1.0, ou seja, Away +1.0

    O código antigo aplicava o handicap diretamente no lado escolhido.
    Isso fazia AH_AWAY_POS_2_5 virar Away +2.5, quando na verdade é Away contra Home +2.5.
    Resultado: ROI absurdo e falso.
    """
    gh = pd.to_numeric(df["G_H_FT"], errors="coerce")
    ga = pd.to_numeric(df["G_A_FT"], errors="coerce")
    adjusted_home = gh + float(home_handicap)

    if side == "Home":
        return (adjusted_home > ga).astype(int)
    if side == "Away":
        return (adjusted_home < ga).astype(int)
    raise ValueError(side)


def void_ah(df: pd.DataFrame, side: str, home_handicap: float) -> pd.Series:
    gh = pd.to_numeric(df["G_H_FT"], errors="coerce")
    ga = pd.to_numeric(df["G_A_FT"], errors="coerce")
    adjusted_home = gh + float(home_handicap)
    return np.isclose(adjusted_home, ga)


def ah_display_line(side: str, home_handicap: float) -> float:
    # Para exibição: mostra o handicap real do lado apostado.
    return float(home_handicap) if side == "Home" else -float(home_handicap)


def target_eh(df: pd.DataFrame, outcome: str, handicap_home: float) -> pd.Series:
    # Handicap europeu: aplica handicap ao mandante e decide 1X2 ajustado.
    gh = pd.to_numeric(df["G_H_FT"], errors="coerce") + handicap_home
    ga = pd.to_numeric(df["G_A_FT"], errors="coerce")
    if outcome == "Home":
        return (gh > ga).astype(int)
    if outcome == "Draw":
        return (gh == ga).astype(int)
    if outcome == "Away":
        return (gh < ga).astype(int)
    raise ValueError(outcome)


def safe_mercado_name(raw: str) -> str:
    return raw.replace("-", "NEG").replace("+", "POS").replace(".", "_").replace(" ", "_")


def build_mercado_catalog(df: pd.DataFrame) -> list[dict]:
    mercados: list[dict] = []

    # 1X2 HT/FT — odds reais existentes
    for period in ["HT", "FT"]:
        for side, label, odd_col in [
            ("H", "Home Win", f"O_H_{period}"),
            ("D", "Draw", f"O_D_{period}"),
            ("A", "Away Win", f"O_A_{period}"),
        ]:
            required = [odd_col, f"G_H_{period}", f"G_A_{period}"]
            if mercado_exists(df, required):
                mercado = f"R_{period}_{side}"
                mercados.append({
                    "mercado": mercado,
                    "event": mercado,
                    "description": f"Resultado {period} - {label}",
                    "odd_col": odd_col,
                    "value_func": lambda d, p=period, s=side: target_1x2(d, p, s),
                    "target_func": lambda d, p=period, s=side: target_1x2(d, p, s),
                    "void_func": lambda d: pd.Series(False, index=d.index),
                })

    # Over/Under gols totais HT/FT — colunas O_/U_ reais.
    for period in ["HT", "FT"]:
        lines = [0.5, 1.5, 2.5]
        if period == "FT":
            # colunas extras existentes na base completa
            if "Over_FT_3_5" in df.columns:
                lines.append(3.5)
            if "Over_FT_4_5" in df.columns:
                lines.append(4.5)

        for line in lines:
            code = str(line).replace(".", "_")
            compact = str(line).replace(".", "")

            if period == "FT" and line in {3.5, 4.5}:
                over_col = f"Over_FT_{code}"
                under_col = f"Under_FT_{code}"
            else:
                over_col = f"O_{compact.zfill(2)}_{period}"
                under_col = f"U_{compact.zfill(2)}_{period}"

            total_col = f"TG_{period}"
            if over_col in df.columns and total_col in df.columns:
                mercado = f"TG_{period}_O{compact.zfill(2)}"
                mercados.append({
                    "mercado": mercado,
                    "event": mercado,
                    "description": f"Total gols {period} over {line:.1f}",
                    "odd_col": over_col,
                    "line": line,
                    "value_func": lambda d, p=period: pd.to_numeric(d[f"TG_{p}"], errors="coerce"),
                    "target_func": lambda d, p=period, l=line: target_over_total(d, p, l),
                    "void_func": lambda d: pd.Series(False, index=d.index),
                    "direction": "over",
                })

            if under_col in df.columns and total_col in df.columns:
                mercado = f"TG_{period}_U{compact.zfill(2)}"
                mercados.append({
                    "mercado": mercado,
                    "event": mercado,
                    "description": f"Total gols {period} under {line:.1f}",
                    "odd_col": under_col,
                    "line": line,
                    "value_func": lambda d, p=period: pd.to_numeric(d[f"TG_{p}"], errors="coerce"),
                    "target_func": lambda d, p=period, l=line: target_under_total(d, p, l),
                    "void_func": lambda d: pd.Series(False, index=d.index),
                    "direction": "under",
                })

    # BTTS FT
    if mercado_exists(df, ["O_BTTS_Y", "O_BTTS_N", "G_H_FT", "G_A_FT"]):
        mercados += [
            {
                "mercado": "BTTS_Y",
                "event": "BTTS_Y",
                "description": "Ambos marcam - Sim",
                "odd_col": "O_BTTS_Y",
                "value_func": target_btts_yes,
                "target_func": target_btts_yes,
                "void_func": lambda d: pd.Series(False, index=d.index),
            },
            {
                "mercado": "BTTS_N",
                "event": "BTTS_N",
                "description": "Ambos marcam - Não",
                "odd_col": "O_BTTS_N",
                "value_func": target_btts_no,
                "target_func": target_btts_no,
                "void_func": lambda d: pd.Series(False, index=d.index),
            },
        ]

    # Dupla chance FT
    for kind, odd_col in [("1X", "O_DC_1X"), ("12", "O_DC_12"), ("X2", "O_DC_X2")]:
        if mercado_exists(df, [odd_col, "G_H_FT", "G_A_FT"]):
            mercado = f"DC_{kind}"
            mercados.append({
                "mercado": mercado,
                "event": mercado,
                "description": f"Dupla chance {kind}",
                "odd_col": odd_col,
                "value_func": lambda d, k=kind: target_dc(d, k),
                "target_func": lambda d, k=kind: target_dc(d, k),
                "void_func": lambda d: pd.Series(False, index=d.index),
            })

    # Asian Handicap FT — usa somente colunas reais da base.
    # IMPORTANTE: o sufixo pos/neg representa o handicap aplicado ao MANDANTE,
    # não necessariamente ao lado apostado. A liquidação usa target_ah(side, home_handicap).
    ah_lines = [
        ("neg_2_5", -2.5), ("neg_2", -2.0), ("neg_1_5", -1.5), ("neg_1", -1.0), ("neg_0_5", -0.5),
        ("pos_0_5", 0.5), ("pos_1", 1.0), ("pos_1_5", 1.5), ("pos_2", 2.0), ("pos_2_5", 2.5),
    ]
    for suffix, home_handicap in ah_lines:
        for side in ["Home", "Away"]:
            odd_col = f"AH_{side}_{suffix}"
            if odd_col in df.columns and odd_valida(df[odd_col]).sum() >= MIN_TRAIN_GAMES:
                display_line = ah_display_line(side, home_handicap)
                sign = "POS" if display_line > 0 else "NEG"
                abs_code = str(abs(display_line)).replace(".", "_")
                mercado = f"AH_{side.upper()}_{sign}_{abs_code}"
                mercados.append({
                    "mercado": mercado,
                    "event": mercado,
                    "description": f"Asian Handicap {side} {display_line:+.1f}",
                    "odd_col": odd_col,
                    "line": display_line,
                    "home_handicap": home_handicap,
                    "value_func": lambda d, h=home_handicap: pd.to_numeric(d["G_H_FT"], errors="coerce") + h - pd.to_numeric(d["G_A_FT"], errors="coerce"),
                    "target_func": lambda d, s=side, h=home_handicap: target_ah(d, s, h),
                    "void_func": lambda d, s=side, h=home_handicap: pd.Series(void_ah(d, s, h), index=d.index),
                })

    # European Handicap FT — 3 vias sem void.
    eh_lines = [("neg_3", -3.0), ("neg_2", -2.0), ("neg_1", -1.0), ("1", 1.0), ("2", 2.0), ("3", 3.0)]
    for suffix, handicap_home in eh_lines:
        for outcome in ["Home", "Draw", "Away"]:
            odd_col = f"EH_{outcome}_{suffix}"
            if odd_col in df.columns and odd_valida(df[odd_col]).sum() >= MIN_TRAIN_GAMES:
                sign = "POS" if handicap_home > 0 else "NEG"
                abs_code = str(abs(handicap_home)).replace(".", "_")
                mercado = f"EH_{outcome.upper()}_{sign}_{abs_code}"
                mercados.append({
                    "mercado": mercado,
                    "event": mercado,
                    "description": f"European Handicap {outcome} {handicap_home:+.0f}",
                    "odd_col": odd_col,
                    "line": handicap_home,
                    "value_func": lambda d, h=handicap_home: pd.to_numeric(d["G_H_FT"], errors="coerce") + h - pd.to_numeric(d["G_A_FT"], errors="coerce"),
                    "target_func": lambda d, o=outcome, h=handicap_home: target_eh(d, o, h),
                    "void_func": lambda d: pd.Series(False, index=d.index),
                })

    # Correct Score FT — odds reais CS_x_y. Não gera ML por padrão se amostra baixa.
    for col in [c for c in df.columns if c.startswith("CS_")]:
        parts = col.split("_")
        if len(parts) != 3:
            continue
        try:
            h, a = int(parts[1]), int(parts[2])
        except Exception:
            continue
        if odd_valida(df[col]).sum() < MIN_TRAIN_GAMES:
            continue
        mercado = f"CS_{h}_{a}"
        mercados.append({
            "mercado": mercado,
            "event": mercado,
            "description": f"Placar correto {h}-{a}",
            "odd_col": col,
            "line": np.nan,
            "value_func": lambda d, hh=h, aa=a: ((pd.to_numeric(d["G_H_FT"], errors="coerce") == hh) & (pd.to_numeric(d["G_A_FT"], errors="coerce") == aa)).astype(int),
            "target_func": lambda d, hh=h, aa=a: ((pd.to_numeric(d["G_H_FT"], errors="coerce") == hh) & (pd.to_numeric(d["G_A_FT"], errors="coerce") == aa)).astype(int),
            "void_func": lambda d: pd.Series(False, index=d.index),
        })

    # Dedup preservando ordem
    seen = set()
    unique = []
    for m in mercados:
        if m["mercado"] in seen:
            continue
        seen.add(m["mercado"])
        unique.append(m)
    return unique


def create_feature_frame(base: pd.DataFrame, mercado_def: dict) -> pd.DataFrame:
    required_meta = ["Game_ID", "Date", "League_std", "Home", "Away"]
    df = base.copy()
    for col in required_meta:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    out = df[required_meta].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True, format="mixed")
    out["value"] = pd.to_numeric(mercado_def["value_func"](df), errors="coerce")
    out[mercado_def["event"]] = pd.to_numeric(mercado_def["target_func"](df), errors="coerce").fillna(0).astype(int)
    out["odd"] = pd.to_numeric(df[mercado_def["odd_col"]], errors="coerce")
    out["odd_real_available"] = out["odd"].notna() & (out["odd"] > 1.0)
    out["void"] = pd.Series(mercado_def["void_func"](df), index=df.index).fillna(False).astype(bool).values

    out = out.dropna(subset=["Date", "value"]).sort_values("Date").reset_index(drop=True)
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in WINDOWS:
        temp = calcular_media_movel(df.copy(), w)
        temp = calcular_desvio_padrao(temp, w)
        temp = calcular_zscore(temp)
        df[f"ma_{w}"] = temp["ma"].values
        df[f"std_{w}"] = temp["std"].values
        df[f"z_{w}"] = temp["zscore"].values

    df = calcular_media_liga(df)
    df = calcular_std_liga(df)
    df = calcular_zscore_liga(df)

    for w in WINDOWS:
        for team_col in ["Home", "Away"]:
            df = calcular_media_movel_grupo(df, team_col, w)
            df = calcular_std_grupo(df, team_col, w)
            df = calcular_zscore_grupo(df, team_col)

    z_cols = [f"z_{w}" for w in WINDOWS if f"z_{w}" in df.columns]
    if z_cols:
        weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(z_cols)]
        weights = weights / weights.sum()
        df["zscore"] = sum(df[col].fillna(0) * w for col, w in zip(z_cols, weights))
    else:
        df["zscore"] = 0.0

    df = gerar_sinal_inteligente(
        df,
        z_global=float(strategy_config.get("z_global", 1.5)),
        z_liga=float(strategy_config.get("z_liga", 1.0)),
        z_team=float(strategy_config.get("z_team", 1.0)),
    )
    return df




def _serie_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _contexto_total_gols(df: pd.DataFrame, direction: str) -> pd.Series:
    """Contexto temporal para mercados de totais, sem usar o resultado atual.

    As colunas usadas aqui são geradas com shift(1) nas funções temporais.
    Over exige ambiente levemente acima do histórico; Under exige ambiente
    levemente abaixo. O corte é propositalmente moderado para não matar a amostra.
    """
    z_global = _serie_default(df, "zscore")
    z_liga = _serie_default(df, "zscore_liga")
    z_home = _serie_default(df, "zscore_Home")
    z_away = _serie_default(df, "zscore_Away")
    z_team = ((z_home + z_away) / 2.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    z = float(BT_TOTALS_CONTEXT_Z)
    if direction == "over":
        score = (z_global >= -z).astype(int) + (z_liga >= -z).astype(int) + (z_team >= -z).astype(int)
    elif direction == "under":
        score = (z_global <= z).astype(int) + (z_liga <= z).astype(int) + (z_team <= z).astype(int)
    else:
        return pd.Series(True, index=df.index)
    return score >= 2



def _context_return_unit(df: pd.DataFrame, target_col: str) -> pd.Series:
    odd = pd.to_numeric(df.get("odd"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    target = pd.to_numeric(df.get(target_col), errors="coerce").fillna(0).astype(int)
    void = df.get("void", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ret = np.where(void, 0.0, np.where(target.eq(1), odd - 1.0, -1.0))
    return pd.Series(ret, index=df.index, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_group_context(df: pd.DataFrame, group_cols: list[str], prefix: str, target_col: str) -> pd.DataFrame:
    """Adiciona ROI/winrate/odd médio pré-jogo por contexto, sempre com shift(1).

    Reescrita para ser vetorizada (cumcount/cumsum) em vez de groupby().transform(lambda
    ...expanding()...). Produz exatamente o mesmo resultado de antes (mesma semântica de
    "apenas resultados anteriores, sem leakage"), mas sem chamar código Python por grupo/
    janela -- isso é o que deixava o backtest lento com centenas de mercados. Requer que
    `df` já esteja ordenado por Date (isso é garantido em add_contextual_market_features,
    que ordena antes de chamar esta função).
    """
    if not all(c in df.columns for c in group_cols):
        df[f"context_{prefix}_apostas"] = 0
        df[f"context_{prefix}_roi"] = 0.0
        df[f"context_{prefix}_winrate"] = 0.0
        df[f"context_{prefix}_odd_media"] = 0.0
        return df

    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    odd = pd.to_numeric(df.get("odd"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    ret = _context_return_unit(df, target_col)
    odd_valid = odd.notna().astype(int)
    odd_filled = odd.fillna(0.0)

    # cumcount() = posição do jogo dentro do grupo (0-based) = nº de jogos ANTERIORES
    # no mesmo grupo. Já equivale a shift(1).expanding().count().
    grp_keys = [df[c] for c in group_cols]

    aux = pd.DataFrame({
        "_grp_target": target.to_numpy(),
        "_grp_ret": ret.to_numpy(),
        "_grp_odd_filled": odd_filled.to_numpy(),
        "_grp_odd_valid": odd_valid.to_numpy(),
    }, index=df.index)
    grp_aux = aux.groupby(grp_keys, sort=False)

    prev_count = grp_aux.cumcount().astype("int64")

    # cumsum() inclui a linha atual; subtraindo o valor da própria linha isolamos a soma
    # de tudo ANTES dela -- equivalente a shift(1).expanding().sum(), mas vetorizado.
    wins_cumsum = grp_aux["_grp_target"].cumsum()
    prev_wins = wins_cumsum - aux["_grp_target"]

    profit_cumsum = grp_aux["_grp_ret"].cumsum()
    prev_profit = profit_cumsum - aux["_grp_ret"]

    odd_sum_cumsum = grp_aux["_grp_odd_filled"].cumsum()
    prev_odd_sum = odd_sum_cumsum - aux["_grp_odd_filled"]

    odd_n_cumsum = grp_aux["_grp_odd_valid"].cumsum()
    prev_odd_n = odd_n_cumsum - aux["_grp_odd_valid"]

    prev_odd = (prev_odd_sum / prev_odd_n.replace(0, np.nan)).fillna(0.0)

    count_safe = prev_count.replace(0, np.nan)
    df[f"context_{prefix}_apostas"] = prev_count.astype(int)
    df[f"context_{prefix}_roi"] = (prev_profit / count_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[f"context_{prefix}_winrate"] = (prev_wins / count_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[f"context_{prefix}_odd_media"] = prev_odd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def add_contextual_market_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Cria features contextuais por liga/time para backtest e ML, sem leakage.

    Cada linha usa somente resultados anteriores do mesmo mercado/contexto.
    Isso permite que o backtest e o dataset de ML aprendam que o mesmo mercado
    muda de comportamento por liga, mandante e visitante.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()

    df = _add_group_context(df, ["League_std"], "liga", target_col)
    df = _add_group_context(df, ["Home"], "home", target_col)
    df = _add_group_context(df, ["Away"], "away", target_col)
    df = _add_group_context(df, ["League_std", "Home"], "liga_home", target_col)
    df = _add_group_context(df, ["League_std", "Away"], "liga_away", target_col)

    liga_ready = df["context_liga_apostas"] >= int(BT_CONTEXT_MIN_LIGA)
    home_ready = df["context_home_apostas"] >= int(BT_CONTEXT_MIN_TIME)
    away_ready = df["context_away_apostas"] >= int(BT_CONTEXT_MIN_TIME)
    liga_home_ready = df["context_liga_home_apostas"] >= int(BT_CONTEXT_MIN_LIGA_TIME)
    liga_away_ready = df["context_liga_away_apostas"] >= int(BT_CONTEXT_MIN_LIGA_TIME)

    def usable(col, mask):
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).where(mask, 0.0)

    df["context_score"] = (
        usable("context_liga_roi", liga_ready) * 0.40
        + usable("context_home_roi", home_ready) * 0.20
        + usable("context_away_roi", away_ready) * 0.20
        + usable("context_liga_home_roi", liga_home_ready) * 0.10
        + usable("context_liga_away_roi", liga_away_ready) * 0.10
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["context_amostra"] = (
        df["context_liga_apostas"]
        + df["context_home_apostas"]
        + df["context_away_apostas"]
        + df["context_liga_home_apostas"]
        + df["context_liga_away_apostas"]
    ).astype(int)

    df["context_ready"] = (liga_ready | home_ready | away_ready | liga_home_ready | liga_away_ready).astype(bool)
    df["context_bad"] = (df["context_ready"] & (df["context_score"] <= float(BT_CONTEXT_BAD_SCORE))).astype(bool)
    df["context_good"] = (df["context_ready"] & (df["context_score"] >= float(BT_CONTEXT_GOOD_SCORE))).astype(bool)
    df["context_allowed"] = (~df["context_bad"]).astype(bool)
    return df


def exportar_contexto_mercado(df: pd.DataFrame, mercado: str, event: str) -> None:
    """Exporta relatórios contextuais do mercado processado para auditoria do laboratório."""
    try:
        out_dir = OUTPUT_DIR / "contexto"
        out_dir.mkdir(parents=True, exist_ok=True)
        apostas = df[(df.get("apostar", False) == True) & (pd.to_numeric(df.get("stake"), errors="coerce").fillna(0) > 0)].copy()
        if apostas.empty:
            return

        def resumo(grupos):
            g = apostas.groupby(grupos).agg(
                apostas=(event, "count"),
                ganhos=(event, "sum"),
                lucro=("retorno", "sum"),
                stake=("stake", "sum"),
                odd_media=("odd", "mean"),
                context_score_medio=("context_score", "mean"),
            ).reset_index()
            g["winrate"] = g["ganhos"] / g["apostas"]
            g["roi"] = g["lucro"] / g["stake"].replace(0, np.nan)
            g["mercado"] = mercado
            return g.replace([np.inf, -np.inf], np.nan).fillna(0).sort_values("lucro", ascending=False)

        resumo(["League_std"]).to_csv(out_dir / f"{mercado}_liga.csv", index=False, encoding="utf-8-sig")
        resumo(["Home"]).to_csv(out_dir / f"{mercado}_home.csv", index=False, encoding="utf-8-sig")
        resumo(["Away"]).to_csv(out_dir / f"{mercado}_away.csv", index=False, encoding="utf-8-sig")
        resumo(["League_std", "Home"]).to_csv(out_dir / f"{mercado}_liga_home.csv", index=False, encoding="utf-8-sig")
        resumo(["League_std", "Away"]).to_csv(out_dir / f"{mercado}_liga_away.csv", index=False, encoding="utf-8-sig")
    except Exception:
        return

def aplicar_sinal_por_mercado(df: pd.DataFrame, mercado_def: dict) -> pd.DataFrame:
    """Gera sinal operacional real para a simulação histórica.

    O runner antigo fazia `signal=True`, então qualquer jogo com EV positivo entrava.
    Agora o sinal exige:
      1. odd real válida;
      2. probabilidade walk-forward pronta;
      3. edge mínimo sobre a probabilidade implícita da odd;
      4. EV mínimo;
      5. para Over/Under, contexto temporal coerente com a direção do mercado.

    Isso reduz volume inútil e força o backtest a avaliar cenários com vantagem
    estimada, sem usar informação pós-jogo.
    """
    odd = pd.to_numeric(df.get("odd"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    prob = pd.to_numeric(df.get("wf_probability"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    implied = (1.0 / odd).replace([np.inf, -np.inf], np.nan)

    df["implied_prob"] = implied.fillna(0.0)
    df["confianca_edge"] = (prob - implied).replace([np.inf, -np.inf], np.nan).fillna(-1.0)

    base_signal = (
        (df.get("wf_ready", False) == True)
        & (df.get("odd_real_available", False) == True)
        & odd.gt(1.0)
        & prob.between(0.0, 1.0, inclusive="both")
        & df["confianca_edge"].ge(float(BT_MIN_CONF_EDGE))
        & ((prob * odd) - 1.0).ge(float(BT_MIN_EV))
    )

    direction = mercado_def.get("direction")
    if str(mercado_def.get("mercado", "")).startswith("TG_") and direction in {"over", "under"}:
        context_signal = _contexto_total_gols(df, direction)
    else:
        context_signal = pd.Series(True, index=df.index)

    df["context_signal"] = context_signal.fillna(False).astype(bool)

    if "context_allowed" in df.columns and bool(BT_USE_CONTEXT_FILTER):
        df["context_filter_ok"] = df["context_allowed"].fillna(True).astype(bool)
    else:
        df["context_filter_ok"] = True

    df["signal"] = (base_signal & df["context_signal"] & df["context_filter_ok"]).fillna(False).astype(bool)
    return df


def _feature_cols_modelo(df: pd.DataFrame, target_col: str) -> list[str]:
    """Seleciona somente features pré-jogo para o modelo.

    Não entra:
    - value atual, porque é o resultado do próprio jogo/mercado;
    - target do mercado;
    - retorno/apostar/EV, porque são calculados depois;
    - colunas de identificação/texto.
    """
    proibidas = {
        target_col,
        "value",
        "Date",
        "Game_ID",
        "League_std",
        "Home",
        "Away",
        "retorno",
        "retorno_unitario",
        "apostar",
        "stake",
        "stake_pct",
        "ev",
        "signal",
        "context_signal",
        "wf_probability",
        "wf_ready",
        "wf_train_games",
        "probabilidade_valida",
        "odd_valida",
        "odd_real_available",
        "void",
    }

    permitidos_prefixos = (
        "ma_",
        "std_",
        "z_",
        "zscore",
        "liga_",
        "context_",
    )
    permitidas_exatas = {"odd", "implied_prob"}

    cols = []
    for c in df.columns:
        if c in proibidas:
            continue
        if c in permitidas_exatas or str(c).startswith(permitidos_prefixos):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return sorted(set(cols))


def aplicar_walk_forward_probabilidade_por_sinal(
    df: pd.DataFrame,
    target_col: str,
    min_train_games: int = 100,
    probability_floor: float = 0.01,
    probability_cap: float = 0.99,
) -> pd.DataFrame:
    """Probabilidade profissional: modelo supervisionado walk-forward temporal.

    Antes o backtest usava apenas frequência histórica da liga. Isso era simples
    demais e fazia o sistema operar como estatística histórica, não como modelo.

    Agora:
    - cria folds por ano;
    - treina sempre em anos anteriores;
    - prevê apenas o ano seguinte/atual;
    - usa somente features pré-jogo já calculadas com shift(1);
    - nunca usa `value` atual nem target atual como feature;
    - se sklearn não estiver disponível, cai para o walk-forward estatístico antigo.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    min_train_games = max(int(min_train_games or 1), 1)

    # Probabilidade implícita da odd é feature pré-jogo e também ajuda o modelo
    # a entender o preço de mercado sem usar resultado futuro.
    odd_num = pd.to_numeric(df.get("odd"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["implied_prob"] = (1.0 / odd_num).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Base estatística sem leakage para fallback e para primeira janela.
    hist = aplicar_walk_forward_probabilidade(
        df.copy(),
        target_col=target_col,
        min_train_games=min_train_games,
        probability_floor=probability_floor,
        probability_cap=probability_cap,
    )
    df["wf_train_games"] = hist["wf_train_games"].values
    df["wf_probability"] = hist["wf_probability"].values
    df["wf_ready"] = hist["wf_ready"].values
    df["prob_model_source"] = "historical_fallback"

    feature_cols = _feature_cols_modelo(df, target_col)
    if not feature_cols:
        return df

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return df

    y_all = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    if y_all.nunique() < 2:
        return df

    years = pd.to_datetime(df["Date"], errors="coerce").dt.year
    unique_years = sorted([int(y) for y in years.dropna().unique()])
    if len(unique_years) < 2:
        return df

    max_train_rows = env_int("BT_MODEL_MAX_TRAIN_ROWS", "80000")
    min_class_count = env_int("BT_MODEL_MIN_CLASS_COUNT", "50")

    for year in unique_years:
        test_mask = years == year
        train_mask = years < year

        if int(train_mask.sum()) < min_train_games or int(test_mask.sum()) == 0:
            continue

        y_train = y_all.loc[train_mask]
        if y_train.nunique() < 2:
            continue
        counts = y_train.value_counts()
        if int(counts.min()) < min_class_count:
            continue

        train_idx = df.index[train_mask].to_numpy()
        if max_train_rows > 0 and len(train_idx) > max_train_rows:
            train_idx = train_idx[-max_train_rows:]

        X_train = df.loc[train_idx, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train_fit = y_all.loc[train_idx]
        X_test = df.loc[test_mask, feature_cols].replace([np.inf, -np.inf], np.nan)

        try:
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=80,
                            learning_rate=0.06,
                            max_leaf_nodes=31,
                            l2_regularization=0.05,
                            random_state=42,
                            # early_stopping: para de treinar assim que o modelo para de
                            # melhorar (em vez de sempre rodar as 80 iterações inteiras).
                            # Reduz o tempo/aquecimento por fold sem trocar de estratégia
                            # (não é paralelização, é fazer menos conta por treino).
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=10,
                            tol=1e-4,
                        ),
                    ),
                ]
            )
            model.fit(X_train, y_train_fit)
            proba = model.predict_proba(X_test)[:, 1]
            df.loc[test_mask, "wf_probability"] = np.clip(proba, probability_floor, probability_cap)
            df.loc[test_mask, "wf_ready"] = True
            df.loc[test_mask, "wf_train_games"] = len(train_idx)
            df.loc[test_mask, "prob_model_source"] = "model_walk_forward_year"
        except Exception:
            # Mantém fallback histórico para este fold se o modelo falhar.
            continue

    return df

def motivo_dataset_ml(metricas: dict, odd_real_found: bool) -> str:
    motivos = []
    if not odd_real_found:
        motivos.append("sem_odd_real")
    if int(metricas.get("apostas", 0)) < MIN_ML_APOSTAS_TO_SAVE:
        motivos.append(f"apostas<{MIN_ML_APOSTAS_TO_SAVE}")
    if float(metricas.get("roi", 0.0)) <= MIN_ML_ROI_TO_SAVE:
        motivos.append(f"roi<={MIN_ML_ROI_TO_SAVE}")
    return ";".join(motivos) if motivos else "ok"


def _tem_liga_com_chance(apostas: pd.DataFrame) -> bool:
    """True se pelo menos uma liga, sozinha, já tem volume e ROI bons dentro
    deste mercado -- mesmo que a média de todas as ligas juntas seja ruim.
    Usado só para decidir se vale a pena salvar o _ml.csv (o motor de
    elegibilidade por liga decide depois, com mais rigor, o que de fato entra
    no treino)."""
    if apostas.empty or "League_std" not in apostas.columns:
        return False
    if "retorno" not in apostas.columns or "stake" not in apostas.columns:
        return False
    por_liga = apostas.groupby("League_std").agg(
        apostas=("retorno", "size"),
        lucro=("retorno", "sum"),
        stake=("stake", "sum"),
    )
    por_liga["roi"] = por_liga["lucro"] / por_liga["stake"].replace(0, np.nan)
    elegivel = (por_liga["apostas"] >= BT_MIN_APOSTAS_LIGA_RESGATE) & (por_liga["roi"] > BT_MIN_ROI_LIGA_RESGATE)
    return bool(elegivel.any())


def process_mercado(base: pd.DataFrame, mercado_def: dict) -> dict:
    mercado = mercado_def["mercado"]
    event = mercado_def["event"]
    odd_col = mercado_def["odd_col"]

    df = create_feature_frame(base, mercado_def)
    if df.empty:
        return {"skip": True, "mercado": mercado, "motivo": "dataset_vazio"}

    df = add_temporal_features(df)
    df = add_contextual_market_features(df, event)

    df = aplicar_walk_forward_probabilidade_por_sinal(
        df,
        target_col=event,
        min_train_games=MIN_TRAIN_GAMES,
    )

    df["probabilidade_valida"] = df["wf_probability"].between(0.0, 1.0, inclusive="both")
    df["odd_valida"] = df["odd_real_available"] & (pd.to_numeric(df["odd"], errors="coerce") > 1.0)
    df["ev"] = (df["wf_probability"] * df["odd"]) - 1.0
    df["ev"] = df["ev"].replace([np.inf, -np.inf], np.nan).fillna(-1.0)

    df = aplicar_sinal_por_mercado(df, mercado_def)

    df["stake_pct"] = 0.0
    if USE_FRACTIONAL_KELLY:
        df["stake_pct"] = df.apply(
            lambda r: calcular_kelly_fractional(r.get("odd"), r.get("wf_probability"), KELLY_FRACTION),
            axis=1,
        )
        df["stake"] = INITIAL_BANKROLL * df["stake_pct"]
    else:
        df["stake"] = FIXED_STAKE

    df["apostar"] = (
        (df["signal"] == True)
        & (df["wf_ready"] == True)
        & (df["probabilidade_valida"] == True)
        & (df["odd_valida"] == True)
        & (df["ev"] > 0)
    )
    if USE_FRACTIONAL_KELLY:
        df["apostar"] = df["apostar"] & (df["stake"] > 0)

    df["stake"] = df["stake"].where(df["apostar"], 0.0)
    win_mask = df[event] == 1
    df["retorno_unitario"] = np.where(df["void"], 0.0, np.where(win_mask, df["odd"] - 1.0, -1.0))
    df["retorno"] = np.where(df["apostar"], df["retorno_unitario"] * df["stake"], 0.0)

    apostas = df[(df["apostar"] == True) & (df["stake"] > 0)].copy()
    metricas = calcular_metricas_apostas(apostas, event, initial_bankroll=INITIAL_BANKROLL)
    exportar_contexto_mercado(df, mercado, event)

    target_pct = float(df[event].mean()) if len(df) else 0.0
    void_pct = float(df["void"].mean()) if len(df) else 0.0
    odd_real_found = bool(df["odd_real_available"].any())
    signal_rate = float(df["signal"].mean()) if "signal" in df.columns and len(df) else 0.0
    avg_conf_edge = float(pd.to_numeric(apostas.get("confianca_edge", pd.Series(dtype="float64")), errors="coerce").mean()) if len(apostas) else 0.0
    ml_motivo = motivo_dataset_ml(metricas, odd_real_found)
    tem_liga_com_chance = _tem_liga_com_chance(apostas)
    save_ml = SAVE_ALL_ML_DATASETS or (ml_motivo == "ok") or tem_liga_com_chance
    if ml_motivo != "ok" and tem_liga_com_chance:
        ml_motivo = f"{ml_motivo};resgatado_por_liga"

    out_path = OUTPUT_DIR / f"{mercado}_ml.csv"
    if save_ml:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        ml_saved = True
    else:
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        ml_saved = False

    return {
        "skip": False,
        "event": event,
        "mercado": mercado,
        "mercado_base": mercado_def.get("mercado_base", mercado),
        "line": mercado_def.get("line", np.nan),
        "line_code": "",
        "mercado_description": mercado_def["description"],
        "apostas": int(metricas["apostas"]),
        "lucro": round(float(metricas["lucro"]), 6),
        "winrate": round(float(metricas["winrate"]), 6),
        "roi": round(float(metricas["roi"]), 6),
        "profit_factor": round(float(metricas["profit_factor"]), 6),
        "drawdown_max": round(float(metricas["drawdown_max"]), 6),
        "sharpe": round(float(metricas["sharpe"]), 6),
        "sortino": round(float(metricas["sortino"]), 6),
        "cagr": round(float(metricas["cagr"]), 6),
        "expectancy": round(float(metricas["expectancy"]), 6),
        "odd_media": round(float(metricas["odd_media"]), 3),
        "target_pct": round(target_pct, 4),
        "void_pct": round(void_pct, 4),
        "odd_col": odd_col,
        "signal_rate": round(signal_rate, 6),
        "avg_conf_edge": round(avg_conf_edge, 6),
        "context_filter_enabled": bool(BT_USE_CONTEXT_FILTER),
        "context_bad_count": int(df.get("context_bad", pd.Series(False, index=df.index)).sum()),
        "context_good_count": int(df.get("context_good", pd.Series(False, index=df.index)).sum()),
        "context_signal_rate": round(float(df.get("context_filter_ok", pd.Series(True, index=df.index)).mean()), 6),
        "odd_col_original": odd_col,
        "odd_candidates": odd_col,
        "odd_real_col_found": odd_real_found,
        "ml_dataset_saved": bool(ml_saved),
        "ml_dataset_motivo": ml_motivo,
        "walk_forward": "expanding",
        "kelly_enabled": USE_FRACTIONAL_KELLY,
    }


def main() -> None:
    log("Carregando base_oficial com odds reais...")
    if not BASE_OFICIAL.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {BASE_OFICIAL}")

    base = pd.read_csv(BASE_OFICIAL, encoding="utf-8-sig", low_memory=False)
    base.columns = base.columns.astype(str).str.strip()

    if "Game_ID" not in base.columns:
        base_cols = [c for c in ["Date", "Country", "League_std", "Home", "Away"] if c in base.columns]
        base["Game_ID"] = base[base_cols].fillna("").astype(str).agg("__".join, axis=1)

    base = reparar_gols_ht_por_minutos(base)

    odd_cols = [c for c in base.columns if c.startswith("O_") or c.startswith("U_") or c.startswith("Over_FT_") or c.startswith("Under_FT_") or c.startswith("AH_") or c.startswith("EH_") or c.startswith("CS_")]
    log(f"{len(base)} jogos carregados | {len(odd_cols)} colunas de odds detectadas", "ok")

    limpar_resultados_antigos()
    mercados = build_mercado_catalog(base)
    log("INÍCIO DO BACKTEST")
    log("=" * 60)
    log(f"Mercados reais detectados: {len(mercados)}", "ok")

    resumo_final = []
    segmentos_final = []
    descartados_final = []

    for i, mercado_def in enumerate(mercados, start=1):
        result = process_mercado(base, mercado_def)
        if result.get("skip"):
            log(f"[{i}/{len(mercados)}] {result['mercado']} ignorado: {result['motivo']}", "warn")
            continue
        resumo_final.append(result)
        status = "ok" if result["ml_dataset_saved"] else "warn"
        log(
            f"[{i}/{len(mercados)}] {result['mercado']:<22} | {result['mercado_description']:<34} | "
            f"apostas: {result['apostas']:>5} | ROI: {result['roi']:+.3f} | "
            f"PF: {result['profit_factor']:.2f} | odd_media: {result['odd_media']:.2f} | "
            f"ML: {result['ml_dataset_motivo']}",
            status,
        )

    resumo_df = pd.DataFrame(resumo_final)
    resumo_path = OUTPUT_DIR / "resumo.csv"
    resumo_df.to_csv(resumo_path, index=False, encoding="utf-8-sig")

    # Segmentação real por liga fica para o ML/banca, mas mantém arquivos para compatibilidade.
    pd.DataFrame(segmentos_final).to_csv(OUTPUT_DIR / "segmentacao_liga_mercado.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(descartados_final).to_csv(OUTPUT_DIR / "segmentacao_descartados_amostra.csv", index=False, encoding="utf-8-sig")

    log("=" * 60)
    log("BACKTEST FINALIZADO", "ok")
    if resumo_df.empty:
        log("Nenhum mercado processado.", "warn")
        return

    classificacao_cols = [
        "mercado", "mercado_description", "apostas", "roi", "profit_factor",
        "drawdown_max", "odd_media", "target_pct", "void_pct", "signal_rate", "avg_conf_edge",
        "context_filter_enabled", "context_bad_count", "context_good_count", "context_signal_rate", "odd_col",
        "ml_dataset_saved", "ml_dataset_motivo",
    ]
    log("\nRANKING POR ROI:\n" + resumo_df.sort_values("roi", ascending=False)[classificacao_cols].to_string(index=False), "info")
    log(f"Resumo salvo em: {resumo_path}", "ok")


if __name__ == "__main__":
    main()