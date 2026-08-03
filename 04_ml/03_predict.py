from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import date, datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sklearn.exceptions import InconsistentVersionWarning
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "04_ml"
BACKTEST_DIR = ROOT_DIR / "03_backtest"
HIST_DIR = ROOT_DIR / "data" / "eventos"
DAILY_DIR = ROOT_DIR / "data" / "raw" / "futpython" / "daily"
DICT_TIMES_PATH = ROOT_DIR / "data" / "dicionario_times.csv"
DICT_LIGAS_PATH = ROOT_DIR / "data" / "dicionario_ligas.csv"
SCHEMA_PATH = ROOT_DIR / "data" / "schema.json"
MODEL_DIR = ML_DIR / "models"
DATASET_DIR = ML_DIR / "datasets"
RESUMO_MODELOS_PATH = MODEL_DIR / "resumo_modelos.pkl"
ENCODERS_PATH = DATASET_DIR / "label_encoders.pkl"
CONFIG_BT_PATH = BACKTEST_DIR / "config.json"
REPORTS_DIR = ML_DIR / "reports"
PREVISOES_HISTORICAS_DIR = ML_DIR / "previsoes_historicas"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controles.operacao.controle_operacional import evaluate_operational_guard
from core.calibration_recovery_engine import apply_calibrated_probability, calibration_absence_blocks_operation
from controles.mercados.status import get_mercado_lifecycle, is_mercado_allowed_by_lifecycle
from core.perfil_operacional import config_mercado
from core.context_score_engine import evaluate_context
from utils.prediction_paths import historical_prediction_path, normal_prediction_path

CONFIDENCE_THRESHOLD = 0.62
MIN_EV = 0.05

# Travas operacionais de recomendação.
# O modelo pode prever e gravar a linha no CSV, mas só recomenda aposta
# quando passar por estes filtros mínimos de segurança operacional.
ODD_MIN = 1.35
ODD_MAX = 3.50

CONTEXT_BLACKLIST_PATH = REPORTS_DIR / "context_blacklist.json"

CONTEXT_SCORES_DIR = BACKTEST_DIR / "results" / "contexto"
_CONTEXT_SCORE_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}

# Fusão hierárquica de modelos contextuais.
# O evento alvo já vem do dataset_builder como classe 1.
# Portanto NÃO inverta probabilidades de mercados Under: P(classe 1) já é P(evento).
CONTEXT_FUSION_ENABLED = os.getenv("ML_CONTEXT_FUSION_ENABLED", "1").strip().lower() in {"1", "true", "sim", "s", "yes", "y"}
CONTEXT_FUSION_GLOBAL_WEIGHT = float(os.getenv("ML_CONTEXT_FUSION_GLOBAL_WEIGHT", "0.55").replace(",", "."))
CONTEXT_FUSION_MIN_SPECIALIST_WEIGHT = float(os.getenv("ML_CONTEXT_FUSION_MIN_SPECIALIST_WEIGHT", "0.10").replace(",", "."))


def _read_context_score_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()
    for col in df.columns:
        if col in {"League_std", "Home", "Away", "mercado"}:
            df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ["apostas", "ganhos", "lucro", "stake", "odd_media", "winrate", "roi", "context_score_medio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def carregar_scores_contexto_mercado(mercado: str) -> Dict[str, pd.DataFrame]:
    """Carrega scores contextuais exportados pelo backtest contextual.

    Arquivos esperados em 03_backtest/results/contexto:
    - {mercado}_liga.csv
    - {mercado}_home.csv
    - {mercado}_away.csv
    - {mercado}_liga_home.csv
    - {mercado}_liga_away.csv
    """
    mercado_key = str(mercado).strip().upper()
    if mercado_key in _CONTEXT_SCORE_CACHE:
        return _CONTEXT_SCORE_CACHE[mercado_key]

    payload = {
        "liga": _read_context_score_csv(CONTEXT_SCORES_DIR / f"{mercado_key}_liga.csv"),
        "home": _read_context_score_csv(CONTEXT_SCORES_DIR / f"{mercado_key}_home.csv"),
        "away": _read_context_score_csv(CONTEXT_SCORES_DIR / f"{mercado_key}_away.csv"),
        "liga_home": _read_context_score_csv(CONTEXT_SCORES_DIR / f"{mercado_key}_liga_home.csv"),
        "liga_away": _read_context_score_csv(CONTEXT_SCORES_DIR / f"{mercado_key}_liga_away.csv"),
    }
    _CONTEXT_SCORE_CACHE[mercado_key] = payload
    return payload


def _context_lookup(df: pd.DataFrame, filtros: Dict[str, str]) -> Dict[str, float]:
    if df.empty:
        return {"apostas": 0.0, "roi": 0.0, "winrate": 0.0, "odd_media": 0.0}
    mask = pd.Series(True, index=df.index)
    for col, val in filtros.items():
        if col not in df.columns:
            return {"apostas": 0.0, "roi": 0.0, "winrate": 0.0, "odd_media": 0.0}
        mask &= df[col].astype(str).str.strip().str.upper().eq(str(val).strip().upper())
    hit = df.loc[mask]
    if hit.empty:
        return {"apostas": 0.0, "roi": 0.0, "winrate": 0.0, "odd_media": 0.0}
    row = hit.sort_values("apostas", ascending=False).iloc[0]
    return {
        "apostas": float(row.get("apostas", 0.0)),
        "roi": float(row.get("roi", 0.0)),
        "winrate": float(row.get("winrate", 0.0)),
        "odd_media": float(row.get("odd_media", 0.0)),
    }


def aplicar_features_contextuais_live(
    row_pred: pd.DataFrame,
    *,
    mercado: str,
    liga: str,
    home: str,
    away: str,
) -> pd.DataFrame:
    """Adiciona features contextuais no predict usando o backtest contextual.

    Isso mantém o contrato do ML: se o treino usou context_liga_roi,
    context_home_winrate etc., o predict também recebe essas colunas.
    """
    row_pred = row_pred.copy()
    scores = carregar_scores_contexto_mercado(mercado)
    liga_key = _norm_contexto(liga)
    home_key = _norm_contexto(home)
    away_key = _norm_contexto(away)

    blocos = {
        "liga": _context_lookup(scores.get("liga", pd.DataFrame()), {"League_std": liga_key}),
        "home": _context_lookup(scores.get("home", pd.DataFrame()), {"Home": home_key}),
        "away": _context_lookup(scores.get("away", pd.DataFrame()), {"Away": away_key}),
        "liga_home": _context_lookup(scores.get("liga_home", pd.DataFrame()), {"League_std": liga_key, "Home": home_key}),
        "liga_away": _context_lookup(scores.get("liga_away", pd.DataFrame()), {"League_std": liga_key, "Away": away_key}),
    }

    for prefix, vals in blocos.items():
        row_pred[f"context_{prefix}_apostas"] = vals["apostas"]
        row_pred[f"context_{prefix}_roi"] = vals["roi"]
        row_pred[f"context_{prefix}_winrate"] = vals["winrate"]
        row_pred[f"context_{prefix}_odd_media"] = vals["odd_media"]

    # Mesma estrutura aproximada do backtest contextual.
    row_pred["context_score"] = (
        row_pred["context_liga_roi"] * 0.40
        + row_pred["context_home_roi"] * 0.20
        + row_pred["context_away_roi"] * 0.20
        + row_pred["context_liga_home_roi"] * 0.10
        + row_pred["context_liga_away_roi"] * 0.10
    )
    row_pred["context_amostra"] = (
        row_pred["context_liga_apostas"]
        + row_pred["context_home_apostas"]
        + row_pred["context_away_apostas"]
        + row_pred["context_liga_home_apostas"]
        + row_pred["context_liga_away_apostas"]
    )
    row_pred["context_ready"] = (row_pred["context_amostra"] > 0).astype(int)
    row_pred["context_bad"] = (row_pred["context_score"] <= -0.10).astype(int)
    row_pred["context_good"] = (row_pred["context_score"] >= 0.05).astype(int)
    row_pred["context_allowed"] = (row_pred["context_bad"] == 0).astype(int)
    return row_pred



def _context_model_key(valor: Any) -> str:
    return _norm_contexto(valor)


def _load_context_models_for_market(mercado: str) -> List[Dict[str, Any]]:
    path = MODEL_DIR / str(mercado) / "context_models.pkl"
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logging.warning(f"[{mercado}] não foi possível carregar especialistas contextuais: {exc}")
        return []


def selecionar_modelo_contextual(
    info: Dict[str, Any],
    *,
    liga: str,
    home: str,
    away: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Seleciona modelo especialista por contexto, com fallback para global.

    Prioridade:
    liga_confronto -> confronto -> liga_home -> liga_away -> liga -> home -> away -> global.
    """
    global_model = info.get("model")
    specialists = info.get("context_models") or []
    if not specialists:
        return global_model, {"tipo": "global", "chave": "", "score": info.get("score")}

    keys = {
        "liga": _context_model_key(liga),
        "home": _context_model_key(home),
        "away": _context_model_key(away),
        "confronto": f"{_context_model_key(home)}||{_context_model_key(away)}",
        "liga_home": f"{_context_model_key(liga)}||{_context_model_key(home)}",
        "liga_away": f"{_context_model_key(liga)}||{_context_model_key(away)}",
        "liga_confronto": f"{_context_model_key(liga)}||{_context_model_key(home)}||{_context_model_key(away)}",
    }
    priority = ["liga_confronto", "confronto", "liga_home", "liga_away", "liga", "home", "away"]

    candidates = []
    for item in specialists:
        ctype = str(item.get("context_type", "")).strip()
        ckey = str(item.get("context_key", "")).strip().upper()
        if ctype in keys and ckey == keys[ctype]:
            candidates.append(item)

    if not candidates:
        return global_model, {"tipo": "global", "chave": "", "score": info.get("score")}

    order = {name: i for i, name in enumerate(priority)}
    best = sorted(
        candidates,
        key=lambda x: (
            order.get(str(x.get("context_type")), 999),
            -float(x.get("score", 0.0)),
            -float(x.get("n_test", 0.0)),
        ),
    )[0]

    return best.get("model", global_model), {
        "tipo": str(best.get("context_type", "global")),
        "chave": str(best.get("context_key", "")),
        "score": float(best.get("score", 0.0)),
        "auc": float(best.get("auc", 0.0)),
        "n_train": int(best.get("n_train", 0)),
        "n_test": int(best.get("n_test", 0)),
    }



def _predict_event_probability(model: Any, X_pred: pd.DataFrame) -> float:
    """Retorna P(classe 1), que é sempre P(evento alvo do mercado).

    O dataset_builder usa a coluna do próprio mercado como target. Logo, para
    TG_HT_U05, classe 1 já significa Under 0.5 HT. Não existe inversão aqui.
    """
    proba = model.predict_proba(X_pred)[0]
    return float(proba[1])


def calcular_probabilidade_fusion_contextual(
    info: Dict[str, Any],
    X_pred: pd.DataFrame,
    *,
    liga: str,
    home: str,
    away: str,
) -> Tuple[float, Dict[str, Any]]:
    """Combina modelo global com especialistas contextuais disponíveis.

    Em vez de escolher um especialista e jogar fora o global, faz blend ponderado.
    Isso evita derrubar o ROI quando um contexto isolado é fraco/instável.
    """
    global_model = info.get("model")
    global_prob = _predict_event_probability(global_model, X_pred)

    if not CONTEXT_FUSION_ENABLED:
        modelo_usado, modelo_info = selecionar_modelo_contextual(info, liga=liga, home=home, away=away)
        prob = _predict_event_probability(modelo_usado, X_pred)
        modelo_info = dict(modelo_info or {})
        modelo_info.update({
            "prob_global": global_prob,
            "prob_final": prob,
            "fusion_enabled": False,
            "fusion_models": 1,
            "fusion_detail": modelo_info.get("tipo", "global"),
        })
        return prob, modelo_info

    specialists = info.get("context_models") or []
    keys = {
        "liga": _context_model_key(liga),
        "home": _context_model_key(home),
        "away": _context_model_key(away),
        "confronto": f"{_context_model_key(home)}||{_context_model_key(away)}",
        "liga_home": f"{_context_model_key(liga)}||{_context_model_key(home)}",
        "liga_away": f"{_context_model_key(liga)}||{_context_model_key(away)}",
        "liga_confronto": f"{_context_model_key(liga)}||{_context_model_key(home)}||{_context_model_key(away)}",
    }
    priority_weight = {
        "liga_confronto": 0.34,
        "confronto": 0.30,
        "liga_home": 0.28,
        "liga_away": 0.22,
        "liga": 0.22,
        "home": 0.16,
        "away": 0.12,
    }

    parts = [{"tipo": "global", "prob": global_prob, "weight": CONTEXT_FUSION_GLOBAL_WEIGHT, "score": float(info.get("score", 0.0) or 0.0), "n_test": 0}]

    for item in specialists:
        ctype = str(item.get("context_type", "")).strip()
        ckey = str(item.get("context_key", "")).strip().upper()
        if ctype not in keys or ckey != keys[ctype]:
            continue
        try:
            p_ctx = _predict_event_probability(item.get("model"), X_pred)
        except Exception:
            continue
        score = float(item.get("score", 0.0) or 0.0)
        n_test = float(item.get("n_test", 0.0) or 0.0)
        sample_factor = min(1.0, max(0.20, n_test / 1000.0))
        score_factor = min(1.25, max(0.50, score / max(float(info.get("score", 0.55) or 0.55), 0.01)))
        weight = max(CONTEXT_FUSION_MIN_SPECIALIST_WEIGHT, priority_weight.get(ctype, 0.10) * sample_factor * score_factor)
        parts.append({
            "tipo": ctype,
            "chave": ckey,
            "prob": p_ctx,
            "weight": weight,
            "score": score,
            "auc": float(item.get("auc", 0.0) or 0.0),
            "n_train": int(item.get("n_train", 0) or 0),
            "n_test": int(item.get("n_test", 0) or 0),
        })

    total_w = sum(float(x["weight"]) for x in parts) or 1.0
    prob_final = sum(float(x["prob"]) * float(x["weight"]) for x in parts) / total_w
    # Blindagem para não deixar especialista instável criar probabilidade absurda.
    prob_final = max(0.01, min(0.99, float(prob_final)))

    best_ctx = max(parts[1:], key=lambda x: (x.get("weight", 0), x.get("score", 0), x.get("n_test", 0)), default=None)
    info_out = {
        "tipo": "fusion" if len(parts) > 1 else "global",
        "chave": " + ".join([str(x.get("tipo")) for x in parts]),
        "score": float(best_ctx.get("score", info.get("score", 0.0))) if best_ctx else float(info.get("score", 0.0) or 0.0),
        "auc": float(best_ctx.get("auc", 0.0)) if best_ctx else float(info.get("auc", 0.0) or 0.0),
        "n_train": int(best_ctx.get("n_train", 0)) if best_ctx else 0,
        "n_test": int(best_ctx.get("n_test", 0)) if best_ctx else 0,
        "prob_global": round(global_prob, 6),
        "prob_final": round(prob_final, 6),
        "fusion_enabled": True,
        "fusion_models": len(parts),
        "fusion_detail": ";".join(f"{x.get('tipo')}={x.get('prob'):.4f}@{x.get('weight'):.3f}" for x in parts),
    }
    return prob_final, info_out


def _norm_contexto(valor: Any) -> str:
    """Normalização única para comparar mercado/liga/time no filtro contextual."""
    if valor is None or pd.isna(valor):
        return ""
    return str(valor).strip().upper()


def carregar_blacklist_contexto() -> Dict[str, Any]:
    """Carrega blacklist dinâmica gerada por 10_gerar_scores_contexto.py.

    Arquivo esperado:
      04_ml/reports/context_blacklist.json

    Se o arquivo não existir, o predict segue funcionando sem bloquear por contexto.
    """
    vazio = {
        "ligas": set(),
        "times": set(),
        "liga_time": set(),
        "raw_loaded": False,
    }

    if not CONTEXT_BLACKLIST_PATH.exists():
        return vazio

    try:
        data = json.loads(CONTEXT_BLACKLIST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning(f"Não foi possível ler blacklist contextual: {exc}")
        return vazio

    ligas = set()
    for item in data.get("ligas", []) or []:
        mercado = _norm_contexto(item.get("mercado"))
        liga = _norm_contexto(item.get("liga"))
        if mercado and liga:
            ligas.add((mercado, liga))

    times = set()
    for item in data.get("times", []) or []:
        mercado = _norm_contexto(item.get("mercado"))
        time = _norm_contexto(item.get("time"))
        if mercado and time:
            times.add((mercado, time))

    liga_time = set()
    for item in data.get("liga_time", []) or []:
        mercado = _norm_contexto(item.get("mercado"))
        liga = _norm_contexto(item.get("liga"))
        time = _norm_contexto(item.get("time"))
        if mercado and liga and time:
            liga_time.add((mercado, liga, time))

    return {
        "ligas": ligas,
        "times": times,
        "liga_time": liga_time,
        "raw_loaded": True,
    }


def avaliar_contexto_operacional(
    *,
    mercado: str,
    liga: str,
    home: str,
    away: str,
    contexto: Dict[str, Any] | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Filtro operacional real por contexto bom/ruim.

    Agora a decisão não é mais mercado puro. A entrada só passa quando liga/time/
    confronto demonstram contexto positivo e nenhum contexto confiável está ruim.
    O score é gerado por 04_ml/10_gerar_scores_contexto.py e pode mudar a cada
    settlement, liberando ou removendo entradas automaticamente.
    """
    ok, motivo, meta = evaluate_context(
        mercado=mercado,
        liga=liga,
        home=home,
        away=away,
        require_positive=True,
    )
    checks = {
        "contexto_arquivo_carregado": bool(meta.get("context_loaded", False)),
        "context_liga_ok": bool(ok),
        "context_home_ok": bool(ok),
        "context_away_ok": bool(ok),
        "context_liga_home_ok": bool(ok),
        "context_liga_away_ok": bool(ok),
        "context_score_final": meta.get("context_score_final", 50.0),
        "context_has_positive": bool(meta.get("context_has_positive", False)),
        "context_has_negative": bool(meta.get("context_has_negative", False)),
        "context_positive_reason": meta.get("context_positive_reason", ""),
        "context_negative_reason": meta.get("context_negative_reason", ""),
        "context_policy": meta.get("context_policy", "CONTEXTUAL_REAL"),
        "context_bootstrap": bool(meta.get("context_bootstrap", False)),
        "context_data_status": meta.get("context_data_status", "DESCONHECIDO"),
        "context_block_reason": meta.get("context_block_reason", motivo),
    }
    return ok, motivo, checks


def carregar_ciclo_vida_confianca() -> Dict[str, Any]:
    """Carrega ciclo de vida das faixas de confiança para observabilidade.

    FASE 20: somente adiciona status_confianca no output.
    Não bloqueia, não libera apostas e não altera guards.
    """
    path = REPORTS_DIR / "ciclo_vida_confianca.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("faixas", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def status_confianca_por_probabilidade(probabilidade: float, ciclo: Dict[str, Any]) -> str:
    try:
        p = float(probabilidade)
        if p > 1.0:
            p = p / 100.0
    except Exception:
        return "SEM_AMOSTRA"
    if 0.50 <= p < 0.60:
        codigo = "50_60"
    elif 0.60 <= p < 0.70:
        codigo = "60_70"
    elif 0.70 <= p < 0.80:
        codigo = "70_80"
    elif 0.80 <= p < 0.90:
        codigo = "80_90"
    elif 0.90 <= p <= 1.01:
        codigo = "90_100"
    else:
        return "SEM_AMOSTRA"
    item = ciclo.get(codigo, {}) if isinstance(ciclo, dict) else {}
    return str(item.get("status") or "SEM_AMOSTRA")

# Faixas operacionais agora vêm de 04_ml/reports/perfil_operacional_mercados.json.
# ODD_MIN/ODD_MAX ficam apenas como fallback de compatibilidade.
MIN_MODEL_AUC = 0.58
MIN_CONFIDENCE_EDGE = 0.05
MAX_BETS_PER_GAME = 1

WINDOWS = [3, 5, 7, 10]

COL_CANDIDATES = {
    "liga": ["League_std", "League_padronizada", "League", "liga", "Liga", "liga"],
    "home": ["Home", "home", "Time_Home", "Home_Name", "Mandante", "mandante"],
    "away": ["Away", "away", "Time_Away", "Away_Name", "Visitante", "visitante"],
    "date": ["Date", "date", "Data", "data"],
    "time": ["Time", "time", "Hora", "hora"],
}

ODD_BY_MARKET_FALLBACK = {
    # Resultado HT/FT
    "R_HT_H": "O_H_HT",
    "R_HT_D": "O_D_HT",
    "R_HT_A": "O_A_HT",
    "R_FT_H": "O_H_FT",
    "R_FT_D": "O_D_FT",
    "R_FT_A": "O_A_FT",

    # Totais HT
    "TG_HT_O05": "O_05_HT",
    "TG_HT_U05": "U_05_HT",
    "TG_HT_O15": "O_15_HT",
    "TG_HT_U15": "U_15_HT",
    "TG_HT_O25": "O_25_HT",
    "TG_HT_U25": "U_25_HT",

    # Totais FT
    "TG_FT_O05": "O_05_FT",
    "TG_FT_U05": "U_05_FT",
    "TG_FT_O15": "O_15_FT",
    "TG_FT_U15": "U_15_FT",
    "TG_FT_O25": "O_25_FT",
    "TG_FT_U25": "U_25_FT",
    "TG_FT_O35": "Over_FT_3_5",
    "TG_FT_U35": "Under_FT_3_5",
    "TG_FT_O45": "Over_FT_4_5",
    "TG_FT_U45": "Under_FT_4_5",

    # BTTS
    "BTTS_Y": "O_BTTS_Y",
    "BTTS_N": "O_BTTS_N",
    "O_BTTS_Y": "O_BTTS_Y",
    "O_BTTS_N": "O_BTTS_N",

    # Dupla chance
    "DC_1X": "O_DC_1X",
    "DC_12": "O_DC_12",
    "DC_X2": "O_DC_X2",
}

DAILY_ODDS_RENAMES = {
    "Odd_H_HT": "O_H_HT",
    "Odd_D_HT": "O_D_HT",
    "Odd_A_HT": "O_A_HT",

    "Odd_Over05_HT": "O_05_HT",
    "Odd_Under05_HT": "U_05_HT",
    "Odd_Over15_HT": "O_15_HT",
    "Odd_Under15_HT": "U_15_HT",
    "Odd_Over25_HT": "O_25_HT",
    "Odd_Under25_HT": "U_25_HT",

    "Odd_H_FT": "O_H_FT",
    "Odd_D_FT": "O_D_FT",
    "Odd_A_FT": "O_A_FT",

    "Odd_Over05_FT": "O_05_FT",
    "Odd_Under05_FT": "U_05_FT",
    "Odd_Over15_FT": "O_15_FT",
    "Odd_Under15_FT": "U_15_FT",
    "Odd_Over25_FT": "O_25_FT",
    "Odd_Under25_FT": "U_25_FT",
    "Odd_Over35_FT": "Over_FT_3_5",
    "Odd_Under35_FT": "Under_FT_3_5",
    "Odd_Over45_FT": "Over_FT_4_5",
    "Odd_Under45_FT": "Under_FT_4_5",

    "Odd_BTTS_Yes": "O_BTTS_Y",
    "Odd_BTTS_No": "O_BTTS_N",

    "Odd_1X_FT": "O_DC_1X",
    "Odd_12_FT": "O_DC_12",
    "Odd_X2_FT": "O_DC_X2",

    "Odd_CS_0x0": "CS_0_0",
    "Odd_CS_0x1": "CS_0_1",
    "Odd_CS_0x2": "CS_0_2",
    "Odd_CS_0x3": "CS_0_3",
    "Odd_CS_1x0": "CS_1_0",
    "Odd_CS_1x1": "CS_1_1",
    "Odd_CS_1x2": "CS_1_2",
    "Odd_CS_1x3": "CS_1_3",
    "Odd_CS_2x0": "CS_2_0",
    "Odd_CS_2x1": "CS_2_1",
    "Odd_CS_2x2": "CS_2_2",
    "Odd_CS_2x3": "CS_2_3",
    "Odd_CS_3x0": "CS_3_0",
    "Odd_CS_3x1": "CS_3_1",
    "Odd_CS_3x2": "CS_3_2",
    "Odd_CS_3x3": "CS_3_3",

    "AH_H_neg_2_5": "AH_Home_neg_2_5",
    "AH_H_neg_2": "AH_Home_neg_2",
    "AH_H_neg_1_5": "AH_Home_neg_1_5",
    "AH_H_neg_1": "AH_Home_neg_1",
    "AH_H_neg_0_5": "AH_Home_neg_0_5",
    "AH_H_pos_0_5": "AH_Home_pos_0_5",
    "AH_H_pos_1": "AH_Home_pos_1",
    "AH_H_pos_1_5": "AH_Home_pos_1_5",
    "AH_H_pos_2": "AH_Home_pos_2",
    "AH_H_pos_2_5": "AH_Home_pos_2_5",

    "AH_A_pos_2_5": "AH_Away_neg_2_5",
    "AH_A_pos_2": "AH_Away_neg_2",
    "AH_A_pos_1_5": "AH_Away_neg_1_5",
    "AH_A_pos_1": "AH_Away_neg_1",
    "AH_A_pos_0_5": "AH_Away_neg_0_5",
    "AH_A_neg_0_5": "AH_Away_pos_0_5",
    "AH_A_neg_1": "AH_Away_pos_1",
    "AH_A_neg_1_5": "AH_Away_pos_1_5",
    "AH_A_neg_2": "AH_Away_pos_2",
    "AH_A_neg_2_5": "AH_Away_pos_2_5",
}

# ==============================
# LINHAS DE MERCADO / ODDS DINÂMICAS
# ==============================
GOAL_MARKET_BASES = ("G_H_FT", "G_A_FT", "TG_FT", "G_H_HT", "G_A_HT", "TG_HT")


def line_code_to_float(code: str) -> Optional[float]:
    code = str(code).strip()
    if not code or not code.isdigit():
        return None
    if len(code) == 1:
        return float(code)
    return float(f"{int(code[:-1])}.{int(code[-1])}")


def extrair_base_e_linha_mercado(mercado: str, event: Optional[str] = None) -> Tuple[str, Optional[float], str]:
    """Extrai base e linha de mercados expandidos, ex: G_A_FT_O15 -> G_A_FT, 1.5."""
    texto = str(mercado or "").strip()
    for base in GOAL_MARKET_BASES:
        prefix = f"{base}_O"
        if texto.startswith(prefix):
            code = texto.replace(prefix, "", 1)
            line = line_code_to_float(code)
            return base, line, code

    # fallback pelo event, caso o resumo venha com event line-coded
    event_txt = str(event or "").strip()
    if "_O" in event_txt:
        code = event_txt.rsplit("_O", 1)[-1]
        line = line_code_to_float(code)
        if line is not None:
            for base in GOAL_MARKET_BASES:
                if texto.startswith(base):
                    return base, line, code
    return texto, None, ""


def format_line(line: Optional[float]) -> str:
    if line is None:
        return ""
    return f"{float(line):.1f}"


def descricao_mercado_linha(mercado: str, event: Optional[str] = None) -> str:
    base, line, _ = extrair_base_e_linha_mercado(mercado, event)
    line_txt = format_line(line)
    if base == "G_A_FT" and line_txt:
        return f"Visitante over {line_txt} gols"
    if base == "G_H_FT" and line_txt:
        return f"Mandante over {line_txt} gols"
    if base == "TG_FT" and line_txt:
        return f"Total over {line_txt} gols"
    if base == "G_A_HT" and line_txt:
        return f"Visitante HT over {line_txt} gols"
    if base == "G_H_HT" and line_txt:
        return f"Mandante HT over {line_txt} gols"
    if base == "TG_HT" and line_txt:
        return f"Total HT over {line_txt} gols"
    return str(event or mercado)


def inferir_odd_candidates_por_linha(mercado: str, event: Optional[str] = None, odd_col: Optional[str] = None) -> List[str]:
    base, line, code = extrair_base_e_linha_mercado(mercado, event)
    candidates: List[str] = []
    if odd_col:
        candidates.append(str(odd_col))
    if mercado in ODD_BY_MARKET_FALLBACK:
        candidates.append(ODD_BY_MARKET_FALLBACK[mercado])
    if base in ODD_BY_MARKET_FALLBACK:
        # usa fallback antigo só para a linha padrão do mercado base
        if (
            (base in {"G_H_FT", "G_A_FT", "G_H_HT", "G_A_HT"} and (line is None or abs(float(line) - 0.5) < 1e-9))
            or (base in {"TG_FT", "TG_HT"} and (line is None or abs(float(line) - 2.5) < 1e-9))
        ):
            candidates.append(ODD_BY_MARKET_FALLBACK[base])

    if line is not None:
        code_alt = str(float(line)).replace(".", "")
        if base == "G_H_FT":
            candidates += [f"O_H_O{code}_FT", f"O_H_{code}_FT", f"O_HOME_O{code}_FT", f"O_H_FT_O{code}"]
            if code == "05":
                candidates += ["O_H_FT"]
        elif base == "G_A_FT":
            candidates += [f"O_A_O{code}_FT", f"O_A_{code}_FT", f"O_AWAY_O{code}_FT", f"O_A_FT_O{code}"]
            if code == "05":
                candidates += ["O_A_FT"]
        elif base == "TG_FT":
            candidates += [f"O_{code}_FT", f"O_O{code}_FT", f"O_TG_O{code}_FT", f"O_TOTAL_O{code}_FT", f"O_{code_alt}_FT"]
            if code == "25":
                candidates += ["O_25_FT"]
        elif base == "G_H_HT":
            candidates += [f"O_H_O{code}_HT", f"O_H_{code}_HT", f"O_H_HT_O{code}"]
            if code == "05":
                candidates += ["O_H_HT"]
        elif base == "G_A_HT":
            candidates += [f"O_A_O{code}_HT", f"O_A_{code}_HT", f"O_A_HT_O{code}"]
            if code == "05":
                candidates += ["O_A_HT"]
        elif base == "TG_HT":
            candidates += [f"O_{code}_HT", f"O_TG_O{code}_HT", f"O_TOTAL_O{code}_HT"]
            if code == "05":
                candidates += ["O_05_HT"]

    candidates.extend([mercado, f"Odd_{mercado}", base, f"Odd_{base}"])
    return list(dict.fromkeys([c for c in candidates if c]))



def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def sair_limpo(mensagem: str, codigo: int = 1) -> None:
    print(f"\n❌ {mensagem}")
    raise SystemExit(codigo)


def to_float_safe(valor: Any, default: float = 0.0) -> float:
    """Converte valores vindos de meta/config para float sem quebrar o predict."""
    try:
        if valor is None or pd.isna(valor):
            return float(default)
        out = float(valor)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def normalizar_nome(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    texto = str(valor).strip().lower()
    for token in [".", "-", "_", "'", '"']:
        texto = texto.replace(token, " ")
    for token in [" fc ", " cf ", " club "]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())


def encontrar_coluna(df: pd.DataFrame, papel: str, obrigatoria: bool = True) -> Optional[str]:
    candidatos = COL_CANDIDATES[papel]
    mapa_lower = {str(c).strip().lower(): c for c in df.columns}
    for candidato in candidatos:
        if candidato in df.columns:
            return candidato
        if candidato.lower() in mapa_lower:
            return mapa_lower[candidato.lower()]
    if obrigatoria:
        disponiveis = ", ".join(map(str, df.columns))
        sair_limpo(
            f"Coluna obrigatória de {papel!r} não encontrada. "
            f"Colunas disponíveis no CSV: {disponiveis}"
        )
    return None


def carregar_csv_obrigatorio(path: Path, descricao: str) -> pd.DataFrame:
    if not path.exists():
        sair_limpo(f"{descricao} não encontrado: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception as exc:
        sair_limpo(f"Erro ao ler {descricao}: {path}\nDetalhe: {exc}")
    df.columns = df.columns.astype(str).str.strip()
    return df


def carregar_schema_renames() -> Dict[str, str]:
    renames = dict(DAILY_ODDS_RENAMES)

    if not SCHEMA_PATH.exists():
        logging.warning("schema.json não encontrado; usando mapeamento interno das odds Daily.")
        return renames

    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning(f"Não foi possível ler schema.json: {exc}")
        return renames

    for origem, cfg in schema.items():
        if isinstance(cfg, dict) and cfg.get("rename"):
            renames[str(origem)] = str(cfg["rename"])

    return renames


def padronizar_colunas_por_schema(df: pd.DataFrame, renames: Dict[str, str]) -> pd.DataFrame:
    rename_map = {origem: destino for origem, destino in renames.items() if origem in df.columns and destino not in df.columns}
    return df.rename(columns=rename_map)


def pares_dicionario(df_dic: pd.DataFrame, tipo: str) -> List[Tuple[str, str]]:
    if df_dic.empty:
        return []
    cols = list(df_dic.columns)
    norm_cols = {c.lower(): c for c in cols}

    if tipo == "liga":
        origem_cands = ["League", "liga", "Liga", "liga", "League_original", "Nome", "nome"]
        destino_cands = ["League_std", "League_padronizada", "Footystats_nome", "Liga_padronizada", "padronizado"]
    else:
        origem_cands = ["Time", "time", "Team", "team", "Nome", "nome", "Time_original"]
        destino_cands = ["Time_padronizado", "Team_std", "Time_std", "Nome_padronizado", "padronizado"]

    origem = next((c for c in origem_cands if c in df_dic.columns), None)
    destino = next((c for c in destino_cands if c in df_dic.columns), None)

    if origem is None:
        origem = norm_cols.get("origem") or norm_cols.get("source") or cols[0]
    if destino is None:
        destino = norm_cols.get("destino") or norm_cols.get("target") or (cols[1] if len(cols) > 1 else cols[0])

    pares = []
    for _, row in df_dic.iterrows():
        src = row.get(origem)
        dst = row.get(destino)
        if pd.notna(src) and pd.notna(dst) and str(src).strip() and str(dst).strip():
            pares.append((str(src).strip(), str(dst).strip()))
    return pares


def aplicar_dicionario(valor: Any, pares: List[Tuple[str, str]]) -> Tuple[str, bool]:
    texto = "" if pd.isna(valor) else str(valor).strip()
    if not texto:
        return texto, False
    mapa = {normalizar_nome(src): dst for src, dst in pares}
    chave = normalizar_nome(texto)
    if chave in mapa:
        return mapa[chave], True
    matches = get_close_matches(chave, list(mapa.keys()), n=1, cutoff=0.88)
    if matches:
        return mapa[matches[0]], True
    return texto, False


def montar_data_hora(
    row: pd.Series,
    col_date: str,
    col_time: Optional[str],
    data_alvo: str,
) -> pd.Timestamp:
    data_base = pd.to_datetime(
        data_alvo,
        format="%Y-%m-%d",
        errors="raise",
    )

    valor_hora = row.get(col_time) if col_time else None

    if pd.notna(valor_hora) and str(valor_hora).strip():
        horario = pd.to_datetime(
            str(valor_hora).strip(),
            errors="coerce",
        )

        if pd.notna(horario):
            data_base = data_base.replace(
                hour=horario.hour,
                minute=horario.minute,
                second=horario.second,
            )

    return pd.Timestamp(data_base)


def carregar_jogos_futpython(data_alvo: str, pares_ligas: List[Tuple[str, str]], pares_times: List[Tuple[str, str]], renames: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    daily_path = DAILY_DIR / f"jogos_do_dia_{data_alvo}.csv"
    if not daily_path.exists():
        sair_limpo(
            f"CSV diário da FutPython não encontrado: {daily_path}\n"
            f"Rode antes: python 01_scripts/01_fetch_futpython_daily.py --date {data_alvo}"
        )

    df = carregar_csv_obrigatorio(daily_path, "CSV diário da FutPython")
    df = padronizar_colunas_por_schema(df, renames)

    col_liga = encontrar_coluna(df, "liga")
    col_home = encontrar_coluna(df, "home")
    col_away = encontrar_coluna(df, "away")
    col_date = encontrar_coluna(df, "date")
    col_time = encontrar_coluna(df, "time", obrigatoria=False)

    jogos = []
    ignorados = 0
    for _, row in df.iterrows():
        liga_raw = row.get(col_liga)
        home_raw = row.get(col_home)
        away_raw = row.get(col_away)
        if pd.isna(liga_raw) or pd.isna(home_raw) or pd.isna(away_raw):
            ignorados += 1
            continue

        liga_std, _ = aplicar_dicionario(liga_raw, pares_ligas)
        home_std, _ = aplicar_dicionario(home_raw, pares_times)
        away_std, _ = aplicar_dicionario(away_raw, pares_times)
        data_jogo = montar_data_hora(row, col_date, col_time, data_alvo)

        registro = row.to_dict()
        registro.update({
            "Date": data_jogo,
            "League_std": liga_std,
            "Home": home_std,
            "Away": away_std,
            "League_original": str(liga_raw).strip(),
            "Home_original": str(home_raw).strip(),
            "Away_original": str(away_raw).strip(),
        })
        jogos.append(registro)

    return pd.DataFrame(jogos), ignorados


def carregar_config_bt() -> Dict[str, Any]:
    if not CONFIG_BT_PATH.exists():
        sair_limpo(f"Config do backtest não encontrada: {CONFIG_BT_PATH}")
    try:
        return json.loads(CONFIG_BT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        sair_limpo(f"Erro ao ler config do backtest: {exc}")


def carregar_modelos(
    config_bt: Dict[str, Any],
    modo_auditoria: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, List[str]]]:
    """Carrega modelos treinados.

    No uso operacional, respeita o ciclo de vida dos mercados antes de carregar
    modelos. No modo auditoria histórica, carrega modelos mesmo quando o
    mercado está APOSENTADA, OBSERVACAO ou BLOQUEADA, mas essa liberação é
    apenas para backfill/estudo histórico: as previsões continuam com
    operacao_real=False, recomendacao_operacional=False e apostar=False.
    """
    diagnostico = {
        "mercados_carregados": [],
        "mercados_sem_modelo": [],
        "mercados_ignorados_por_status": [],
        "mercados_ignorados_por_erro_tecnico": [],
    }

    faltantes = [p for p in [RESUMO_MODELOS_PATH, ENCODERS_PATH] if not p.exists()]
    if faltantes:
        logging.warning("Artefatos de ML ausentes:")
        for p in faltantes:
            logging.warning(f"  - {p}")
        return {}, {}, diagnostico

    try:
        with open(RESUMO_MODELOS_PATH, "rb") as f:
            resumo_modelos = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except Exception as exc:
        sair_limpo(f"Erro ao carregar artefatos de ML: {exc}")

    thresholds = {
        e.get("source"): {
            "thresholds_liga": e.get("thresholds_liga", {}),
            "threshold_default": e.get("threshold_default", 0.5),
            "odd_col": e.get("odd_col"),
        }
        for e in config_bt.get("events", [])
    }

    if modo_auditoria:
        print("Modo auditoria: carregando modelos independentemente do status operacional do mercado.")

    modelos = {}
    for meta in resumo_modelos:
        mercado = meta.get("mercado") or meta.get("market") or meta.get("source")
        if not mercado:
            continue
        mercado = str(mercado)

        # No modo operacional, mantém o filtro histórico de ROI do backtest.
        # No modo auditoria, o objetivo é reproduzir previsões históricas para estudo,
        # então modelos existentes podem ser carregados mesmo quando o mercado não é elegível operacionalmente.
        
        #if not modo_auditoria and float(meta.get("roi_bt", 0)) < MIN_ROI_BT:
            #diagnostico["mercados_ignorados_por_status"].append(f"{mercado}: ROI_BACKTEST_ABAIXO_DO_MINIMO")
            #continue

        status_info = get_mercado_lifecycle(mercado)
        status_ciclo_vida = str(status_info.get("status_ciclo_vida") or "DESCONHECIDA").upper()
        if status_ciclo_vida in {"APOSENTADA", "BLOQUEADA", "OBSERVACAO", "OBSERVAÇÃO"} and not modo_auditoria:
            logging.warning(
                f"[{mercado}] mercado {status_ciclo_vida}; modelo não será carregado no predict. "
                f"Motivo: {status_info.get('motivo') or 'STATUS_MERCADO'}"
            )
            diagnostico["mercados_ignorados_por_status"].append(f"{mercado}: {status_ciclo_vida}")
            continue

        model_path = MODEL_DIR / mercado / "model.pkl"
        feat_path = DATASET_DIR / mercado / "feature_columns.pkl"

        if not model_path.exists():
            logging.warning(f"[{mercado}] modelo ausente — pulando.")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if feat_path.exists():
            with open(feat_path, "rb") as f:
                feature_columns = pickle.load(f)
        elif hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        else:
            logging.warning(f"[{mercado}] feature_columns ausente e modelo sem feature_names_in_ — pulando.")
            continue

        info = thresholds.get(mercado, {})
        context_models = _load_context_models_for_market(mercado)
        if context_models:
            logging.info(f"[{mercado}] especialistas contextuais carregados: {len(context_models)}")

        modelos[mercado] = {
            "model": model,
            "context_models": context_models,
            "feature_columns": list(feature_columns),
            "meta": meta,
            "score": meta.get("score"),
            "thresholds_liga": info.get("thresholds_liga", {}),
            "threshold_default": info.get("threshold_default", 0.5),
            "odd_col": info.get("odd_col") or (inferir_odd_candidates_por_linha(mercado, meta.get("event"))[0] if inferir_odd_candidates_por_linha(mercado, meta.get("event")) else ODD_BY_MARKET_FALLBACK.get(mercado)),
            "status_ciclo_vida": status_ciclo_vida,
            "motivo_ciclo_vida": status_info.get("motivo") or "STATUS_MERCADO",
            "carregado_modo_auditoria": bool(modo_auditoria and status_ciclo_vida != "ATIVA"),
            # O perfil permanece imutável durante uma execução do predict.
            # Carregá-lo uma vez por mercado evita leituras repetidas do JSON.
            "cfg_mercado_operacional": config_mercado(mercado),
        }
        diagnostico["mercados_carregados"].append(mercado)
    return modelos, encoders, diagnostico


def carregar_historico() -> pd.DataFrame:
    if not HIST_DIR.exists():
        logging.warning(f"Pasta de histórico não encontrada: {HIST_DIR}")
        return pd.DataFrame()
    arquivos = sorted(HIST_DIR.glob("*.csv"))
    if not arquivos:
        logging.warning(f"Nenhum CSV histórico encontrado em {HIST_DIR}")
        return pd.DataFrame()
    partes = []
    for arquivo in arquivos:
        try:
            partes.append(pd.read_csv(arquivo, encoding="utf-8-sig"))
        except Exception as exc:
            logging.warning(f"Erro ao carregar histórico {arquivo.name}: {exc}")
    if not partes:
        return pd.DataFrame()
    df = pd.concat(partes, ignore_index=True)
    if "Date" not in df.columns:
        logging.warning("Histórico sem coluna Date; não será possível calcular features temporais.")
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    return df


def importar_funcoes_backtest() -> Optional[Dict[str, Any]]:
    sys.path.insert(0, str(BACKTEST_DIR))
    try:
        from functions import (  # type: ignore
            calcular_desvio_padrao,
            calcular_media_liga,
            calcular_media_movel,
            calcular_media_movel_grupo,
            calcular_std_grupo,
            calcular_std_liga,
        )
        return locals()
    except Exception as exc:
        logging.warning(f"Não foi possível importar funções do backtest: {exc}")
        return None



def _serie_numerica(df: pd.DataFrame, col: str = "value") -> pd.Series:
    """Retorna uma série numérica limpa para cálculo de features pré-jogo."""
    if df.empty or col not in df.columns:
        return pd.Series(dtype="float64")
    serie = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return serie.astype(float)


def _media_ultimos(df: pd.DataFrame, n: int, fallback: float = 0.0) -> float:
    serie = _serie_numerica(df).tail(n)
    if serie.empty:
        return float(fallback)
    return float(serie.mean())


def _std_ultimos(df: pd.DataFrame, n: int, fallback: float = 0.0001) -> float:
    serie = _serie_numerica(df).tail(n)
    if len(serie) <= 1:
        return float(fallback)
    valor = float(serie.std(ddof=1))
    if not np.isfinite(valor) or valor <= 0:
        return float(fallback)
    return valor


def _filtrar_time(df: pd.DataFrame, coluna: str, time: str) -> pd.DataFrame:
    if df.empty or coluna not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df[coluna].astype(str).str.strip() == str(time).strip()].copy()


def _filtrar_time_geral(df: pd.DataFrame, time: str) -> pd.DataFrame:
    partes = []
    for coluna in ["Home", "Away"]:
        if coluna in df.columns:
            partes.append(_filtrar_time(df, coluna, time))
    if not partes:
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(partes, ignore_index=True).drop_duplicates()
    if "Date" in out.columns:
        out = out.sort_values("Date")
    return out


def calcular_features_jogo(home: str, away: str, liga: str, data_jogo: pd.Timestamp, historico: pd.DataFrame, funcoes: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Calcula features pré-jogo usando somente histórico anterior à data do jogo.

    A versão anterior adicionava uma linha sintética com ``value = média da liga`` e
    passava essa linha pelas funções do backtest. Isso deixava muitos confrontos com
    praticamente o mesmo vetor de features e gerava probabilidades repetidas no predict.

    Esta versão calcula diretamente as estatísticas disponíveis antes do jogo:
    - médias/desvios recentes da liga;
    - média/desvio histórico da liga;
    - desempenho histórico recente do mandante como mandante;
    - desempenho histórico recente do visitante como visitante;
    - fallback para histórico geral do time e, por último, para a liga.
    """
    if historico.empty:
        return None

    col_liga = "League_std" if "League_std" in historico.columns else ("League" if "League" in historico.columns else None)
    if not col_liga or "Date" not in historico.columns:
        return None

    df_liga = historico[(historico[col_liga] == liga) & (historico["Date"] < data_jogo)].copy()
    if df_liga.empty:
        return None

    if "League_std" not in df_liga.columns:
        df_liga["League_std"] = df_liga[col_liga]
    if "value" not in df_liga.columns:
        numericas = df_liga.select_dtypes(include=[np.number]).columns.tolist()
        df_liga["value"] = df_liga[numericas[0]] if numericas else 0.0

    df_liga["value"] = pd.to_numeric(df_liga["value"], errors="coerce")
    df_liga = df_liga.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if df_liga.empty:
        return None

    if "Date" in df_liga.columns:
        df_liga = df_liga.sort_values("Date")

    serie_liga = _serie_numerica(df_liga)
    liga_mean = float(serie_liga.mean()) if not serie_liga.empty else 0.0
    liga_std = float(serie_liga.std(ddof=1)) if len(serie_liga) > 1 else 0.0001
    if not np.isfinite(liga_std) or liga_std <= 0:
        liga_std = 0.0001

    # Recortes específicos do confronto. O fallback geral evita zerar times que
    # aparecem em casa/fora com pouco histórico, mas continua usando apenas dados passados.
    df_home = _filtrar_time(df_liga, "Home", home)
    if df_home.empty:
        df_home = _filtrar_time_geral(df_liga, home)

    df_away = _filtrar_time(df_liga, "Away", away)
    if df_away.empty:
        df_away = _filtrar_time_geral(df_liga, away)

    registro: Dict[str, Any] = {
        "Game_ID": 0,
        "Date": data_jogo,
        "Home": home,
        "Away": away,
        "League_std": liga,
        # Mantido apenas por compatibilidade; não é usado como feature atual.
        "value": liga_mean,
        "liga_mean": liga_mean,
        "liga_std": liga_std,
        "ma_Home": _media_ultimos(df_home, 10, fallback=liga_mean),
        "std_Home": _std_ultimos(df_home, 10, fallback=liga_std),
        "ma_Away": _media_ultimos(df_away, 10, fallback=liga_mean),
        "std_Away": _std_ultimos(df_away, 10, fallback=liga_std),
    }

    for w in WINDOWS:
        registro[f"ma_{w}"] = _media_ultimos(df_liga, w, fallback=liga_mean)
        registro[f"std_{w}"] = _std_ultimos(df_liga, w, fallback=liga_std)

    row = pd.DataFrame([registro])
    row = row.replace([np.inf, -np.inf], 0).fillna(0)
    return row

def odd_default_config(config_bt: Dict[str, Any]) -> float:
    for chave in ["odd_default", "default_odd"]:
        if chave in config_bt:
            try:
                return float(config_bt[chave])
            except Exception:
                pass
    return 1.50


def extrair_odd(row: pd.Series, mercado: str, odd_col: Optional[str], renames: Dict[str, str], event: Optional[str] = None) -> Tuple[float, bool]:
    candidatos = inferir_odd_candidates_por_linha(mercado, event, odd_col)

    if mercado in ODD_BY_MARKET_FALLBACK:
        candidatos.insert(0, ODD_BY_MARKET_FALLBACK[mercado])

    origem_por_destino = {destino: origem for origem, destino in renames.items()}
    for col in list(candidatos):
        if col in origem_por_destino:
            candidatos.append(origem_por_destino[col])

    for col in dict.fromkeys(candidatos):
        if col in row.index:
            valor = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(valor) and float(valor) > 1.0:
                return float(valor), True
    return np.nan, False




def resolver_threshold_confianca(info: Dict[str, Any], liga: str) -> Tuple[Any, float, str]:
    """Resolve o threshold de confiança de forma defensiva.

    Thresholds de confiança precisam ser probabilidades válidas entre 0.50 e 0.99.
    Valores acima de 0.99 são preservados em threshold_raw, mas não entram na decisão
    porque normalmente representam linha operacional/de mercado, como 1.5, 2.5 ou 2.7.
    """
    thresholds_liga = info.get("thresholds_liga") or {}
    threshold_default = info.get("threshold_default")

    threshold_raw = thresholds_liga.get(liga, threshold_default)

    try:
        threshold_float = float(threshold_raw)
    except (TypeError, ValueError):
        return threshold_raw, float(CONFIDENCE_THRESHOLD), "fallback_confianca_threshold"

    if 0.50 <= threshold_float <= 0.99:
        if liga in thresholds_liga:
            return threshold_raw, threshold_float, "threshold_liga"
        return threshold_raw, threshold_float, "threshold_default"

    return threshold_raw, float(CONFIDENCE_THRESHOLD), "fallback_confianca_threshold"


def avaliar_filtros_operacionais(
    *,
    prob_evento: float,
    threshold_usado: float,
    odd: float,
    tem_odd_real: bool,
    ev: float,
    auc_modelo: float,
    roi_bt: float,
    mercado: str = "",
    cfg_mercado: Dict[str, Any] | None = None,
) -> Tuple[bool, str, Dict[str, bool]]:
    """Aplica travas finais antes de recomendar aposta.

    A previsão continua sendo registrada no CSV mesmo quando a recomendação é bloqueada.
    Isso permite auditar se o bloqueio veio de odd fora da faixa, confiança colada no
    threshold, EV baixo, ausência de odd real, AUC fraco ou ROI histórico negativo.
    """
    margem_confianca = float(prob_evento) - float(threshold_usado)
    cfg_mercado = cfg_mercado or config_mercado(mercado)
    odd_min_mercado = float(cfg_mercado["odd_min"])
    odd_max_mercado = float(cfg_mercado["odd_max"])
    min_ev_mercado = float(cfg_mercado["min_ev"])

    checks = {
        "tem_odd_real": bool(tem_odd_real),
        "perfil_ativo_ok": bool(cfg_mercado["ativo"]),
        "ev_ok": bool(float(ev) >= min_ev_mercado),
        "odd_ok": bool(odd_min_mercado <= float(odd) <= odd_max_mercado),
        "auc_ok": bool(float(auc_modelo) >= MIN_MODEL_AUC),
        "roi_bt_ok": bool(cfg_mercado["roi_policy_ok"]),
        "confianca_edge_ok": bool(margem_confianca >= MIN_CONFIDENCE_EDGE),
    }

    motivos = []
    if not checks["tem_odd_real"]:
        motivos.append("sem_odd_real")
    if not checks["perfil_ativo_ok"]:
        motivos.append("mercado_inativo_perfil")
    if not checks["ev_ok"]:
        motivos.append("ev_baixo")
    if not checks["odd_ok"]:
        motivos.append("odd_fora_da_faixa")
    if not checks["auc_ok"]:
        motivos.append("auc_modelo_baixo")
    if not checks["roi_bt_ok"]:
        motivos.append("roi_policy_reprovada")
    if not checks["confianca_edge_ok"]:
        motivos.append("confianca_colada_no_threshold")

    aprovado = all(checks.values())
    return aprovado, "ok" if aprovado else ";".join(motivos), checks



def aplicar_mercado_governance_status(mercado: str, guard: Dict[str, Any]) -> Tuple[bool, str, str]:
    lifecycle_allowed, status_ciclo_vida, lifecycle_motivo = is_mercado_allowed_by_lifecycle(str(mercado))
    if not lifecycle_allowed:
        return False, status_ciclo_vida, lifecycle_motivo

    mercados_bloqueados = set(str(x) for x in guard.get("mercados_bloqueados", []) or [])

    if str(mercado) in mercados_bloqueados:
        motivos = (guard.get("mercado_block_motivos", {}) or {}).get(str(mercado), [])
        motivo = ";".join(map(str, motivos)) if motivos else "MERCADO_BLOQUEADO_BY_GOVERNANCE"
        return False, "BLOQUEADA", motivo

    return True, status_ciclo_vida, ""


def aplicar_limite_apostas_por_jogo(df_out: pd.DataFrame) -> pd.DataFrame:
    if df_out.empty or "apostar" not in df_out.columns:
        return df_out

    df_out = df_out.copy()

    if "margem_confianca" not in df_out.columns:
        df_out["margem_confianca"] = 0.0
    if "ev" not in df_out.columns:
        df_out["ev"] = 0.0

    df_out["score_operacional"] = (
        pd.to_numeric(df_out["ev"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df_out["margem_confianca"], errors="coerce").fillna(0.0)
    ).round(6)

    df_out["limite_1_aposta_por_jogo_ok"] = True

    chaves_jogo = ["data", "liga", "home", "away"]
    if not all(col in df_out.columns for col in chaves_jogo):
        return df_out

    mask_apostas = df_out["apostar"].astype(bool)
    if not mask_apostas.any():
        return df_out

    melhores_idx = (
        df_out.loc[mask_apostas]
        .sort_values(
            chaves_jogo + ["score_operacional", "ev", "prob_evento"],
            ascending=[True, True, True, True, False, False, False],
        )
        .groupby(chaves_jogo, as_index=False)
        .head(int(MAX_BETS_PER_GAME))
        .index
    )

    mask_excedente = mask_apostas & ~df_out.index.isin(melhores_idx)
    df_out.loc[mask_excedente, "apostar"] = False
    df_out.loc[mask_excedente, "limite_1_aposta_por_jogo_ok"] = False
    df_out.loc[mask_excedente, "motivo_nao_apostar"] = "limite_1_aposta_por_jogo"

    return df_out

def prever(data_alvo: str, modo_auditoria: bool = False) -> None:
    setup_logging()
    warnings.filterwarnings(
        "once",
        category=InconsistentVersionWarning,
    )
    logging.info("=" * 60)
    logging.info(f"Football Lab Predict — FutPython Daily — {data_alvo}")
    if modo_auditoria:
        logging.info("Modo auditoria histórica ativo — operação real e recomendações permanecem desativadas.")

    guard = evaluate_operational_guard()
    if str(guard.get("motivo", "")).upper() == "PROTECAO_SIMULACAO":
        print("\n" + "=" * 60)
        print("⚠️ GUARD EM PROTECAO_SIMULACAO")
        print("=" * 60)
        print("Gerando previsões para banca de teste.")
        print("Operação real controlada pelo próprio usuário.")
        print(f"Status : {guard.get('status')}")
        print(f"Motivo : {guard.get('motivo')}")
        print("=" * 60)
   
    elif modo_auditoria:
        print("\n" + "=" * 60)
        print("ℹ️ GUARD OPERACIONAL BLOQUEADO, MAS MODO AUDITORIA SEGUE")
        print("=" * 60)
        print("Nenhuma operação real ou recomendação será gerada nesta rodada.")
        print(f"Status : {guard.get('status')}")
        print(f"Motivo : {guard.get('motivo')}")
        print("=" * 60)

    elif not bool(guard.get("permitir_previsoes", False)):
        print("\n" + "=" * 60)
        print("🛑 PREDICT BLOQUEADO PELO OPERATIONAL GUARD")
        print("=" * 60)
        print(f"Status : {guard.get('status')}")
        print(f"Motivo : {guard.get('motivo')}")
        print(f"Arquivo: {REPORTS_DIR / 'status_operacional.json'}")
        print("Nenhuma previsão, shortlist, recomendação ou aposta foi gerada.")
        print("=" * 60)
        sair_limpo(
            "Operational guard bloqueou a geração. O pipeline diário foi interrompido "
            "para impedir consumo de um CSV de previsões antigo.",
            codigo=2,
        )
    
    if modo_auditoria and not bool(guard.get("permitir_previsoes")):
        print("\nℹ️ Modo auditoria histórica ativo: guard operacional bloqueado para operação real, mas o backfill histórico seguirá sem recomendações.")
        print(f"Status operacional atual: {guard.get('status')} | Motivo: {guard.get('motivo')}")

    renames = carregar_schema_renames()
    df_ligas = carregar_csv_obrigatorio(DICT_LIGAS_PATH, "Dicionário ativo de ligas")
    df_times = carregar_csv_obrigatorio(DICT_TIMES_PATH, "Dicionário ativo de times")
    pares_ligas = pares_dicionario(df_ligas, "liga")
    pares_times = pares_dicionario(df_times, "time")

    df_jogos, ignorados_iniciais = carregar_jogos_futpython(data_alvo, pares_ligas, pares_times, renames)
    jogos_carregados = len(df_jogos) + ignorados_iniciais
    if df_jogos.empty:
        sair_limpo("Nenhum jogo válido encontrado no CSV diário da FutPython.")

    config_bt = carregar_config_bt()
    modelos, encoders, diagnostico_modelos = carregar_modelos(config_bt, modo_auditoria=modo_auditoria)
    if not modelos:
        print(f"Jogos carregados: {jogos_carregados}")
        print(f"Arquivo diário lido: {DAILY_DIR / f'jogos_do_dia_{data_alvo}.csv'}")
        if modo_auditoria:
            print("Mercados sem modelo       : " + ", ".join(diagnostico_modelos.get("mercados_sem_modelo", [])[:20]))
            print("Erros técnicos            : " + ", ".join(diagnostico_modelos.get("mercados_ignorados_por_erro_tecnico", [])[:20]))
        sair_limpo(
            "Nenhum modelo publicável foi carregado. O pipeline foi bloqueado; "
            "treine e publique um resumo_modelos.pkl não vazio.",
            codigo=2,
        )

    historico = carregar_historico()
    funcoes = importar_funcoes_backtest()
    odd_padrao = odd_default_config(config_bt)
    resultados = []
    calibradores_ausentes_logados = set()
    ciclo_vida_confianca = carregar_ciclo_vida_confianca()
    jogos_validos = 0
    jogos_ignorados = ignorados_iniciais

    for _, jogo in df_jogos.iterrows():
        home = str(jogo["Home"]).strip()
        away = str(jogo["Away"]).strip()
        liga = str(jogo["League_std"]).strip()
        data_jogo = pd.Timestamp(jogo["Date"])
        row_features = calcular_features_jogo(home, away, liga, data_jogo, historico, funcoes)
        if row_features is None:
            jogos_ignorados += 1
            logging.warning(f"Sem histórico anterior suficiente para '{liga}' — pulando {home} x {away}.")
            continue
        jogos_validos += 1

        for col, le in (encoders or {}).items():
            if col in row_features.columns:
                val = str(row_features[col].values[0])
                row_features[col] = le.transform([val])[0] if hasattr(le, "classes_") and val in le.classes_ else -1
        row_features.replace([np.inf, -np.inf], 0, inplace=True)
        row_features.fillna(0, inplace=True)

        for mercado, info in modelos.items():
            feature_columns = info["feature_columns"]
            row_pred = row_features.copy()
            row_pred = aplicar_features_contextuais_live(
                row_pred,
                mercado=mercado,
                liga=liga,
                home=home,
                away=away,
            )

            odd, tem_odd_real = extrair_odd(
                jogo,
                mercado,
                info.get("odd_col"),
                renames,
                info.get("event", mercado),
            )

            odd_para_modelo = odd if tem_odd_real else np.nan

            for col in feature_columns:
                if col not in row_pred.columns:
                    row_pred[col] = 0

            if "implied_prob" in row_pred.columns:
                row_pred["implied_prob"] = (
                    1.0 / float(odd_para_modelo)
                    if pd.notna(odd_para_modelo) and float(odd_para_modelo) > 1.0
                    else 0.0
                )

            if "odd_real_available" in row_pred.columns:
                row_pred["odd_real_available"] = 1 if tem_odd_real else 0

            X_pred = row_pred[feature_columns]
            try:
                prob, modelo_contextual_info = calcular_probabilidade_fusion_contextual(
                    info,
                    X_pred,
                    liga=liga,
                    home=home,
                    away=away,
                )
            except Exception as exc:
                logging.warning(f"[{mercado}] erro ao prever {home} x {away}: {exc}")
                continue

            prob_modelo = prob
            probabilidade_bruta = prob_modelo
            status_ciclo_vida_modelo = str(info.get("status_ciclo_vida") or "DESCONHECIDA").upper()
            motivo_ciclo_vida_modelo = str(info.get("motivo_ciclo_vida") or "STATUS_MERCADO")
            probabilidade_calibrada, calibration_method, calibrator_found = apply_calibrated_probability(mercado, probabilidade_bruta)
            calibrator_absence_blocks = False
            if modo_auditoria and not calibrator_found:
                if mercado not in calibradores_ausentes_logados:
                    logging.warning(
                        f"[{mercado}] sem calibrador externo adicional no modo auditoria — "
                        "usando diretamente a probabilidade produzida pelo modelo."
                    )
                    calibradores_ausentes_logados.add(mercado)

                calibration_method = "modelo_sem_calibrador_externo"
                probabilidade_calibrada = probabilidade_bruta
            prob_evento = probabilidade_calibrada
            confianca = prob_evento
            threshold_raw, threshold_usado, threshold_fonte = resolver_threshold_confianca(info, liga)
            apostar_conf = prob_evento >= threshold_usado
            prob_extrema = prob_evento >= 0.95
            prob_100 = prob_evento >= 0.9999

            if not tem_odd_real:
                odd = odd_padrao
                logging.info(f"[{mercado}] odd real ausente em {home} x {away}; usando fallback explícito {odd_padrao:.2f}.")
            ev = (prob_evento * float(odd)) - 1
            meta = info["meta"]
            auc_modelo = to_float_safe(meta.get("auc"), 0.0)
            roi_bt = to_float_safe(meta.get("roi_bt"), 0.0)
            margem_confianca = float(prob_evento) - float(threshold_usado)

            operacional_ok, motivo_nao_apostar, checks_operacionais = avaliar_filtros_operacionais(
                prob_evento=prob_evento,
                threshold_usado=threshold_usado,
                odd=float(odd),
                tem_odd_real=tem_odd_real,
                ev=ev,
                auc_modelo=auc_modelo,
                roi_bt=roi_bt,
                mercado=mercado,
                cfg_mercado=info["cfg_mercado_operacional"],
            )
            cfg_mercado_operacional = info["cfg_mercado_operacional"]
            context_ok, context_motivo, checks_contexto = avaliar_contexto_operacional(
                mercado=mercado,
                liga=liga,
                home=home,
                away=away,
            )

            # Contexto operacional é filtro de execução: mercado só entra
            # quando liga/time/confronto provam que estão bons. Não é bloqueio fixo;
            # os scores são recalculados após settlement.
            if not context_ok:
                operacional_ok = False
                if motivo_nao_apostar == "ok":
                    motivo_nao_apostar = context_motivo
                else:
                    motivo_nao_apostar = f"{motivo_nao_apostar};{context_motivo}"

            modo_teste_real = (
                not modo_auditoria
                and str(guard.get("motivo", "")).upper() == "PROTECAO_SIMULACAO"
            )

            if modo_auditoria:
                mercado_allowed = True
                mercado_status = status_ciclo_vida_modelo
                mercado_block_motivo = "BACKFILL_HISTORICO"
            elif modo_teste_real:
                mercado_allowed = True
                mercado_status = "TESTE_REAL"
                mercado_block_motivo = ""
            else:
                mercado_allowed, mercado_status, mercado_block_motivo = aplicar_mercado_governance_status(mercado, guard)
                if not calibrator_found:
                    calibrator_absence_blocks = bool(calibration_absence_blocks_operation(mercado))
                    if calibrator_absence_blocks:
                        mercado_allowed = False
                        mercado_status = "CALIBRADOR_NAO_ENCONTRADO"
                        mercado_block_motivo = "CALIBRADOR_NAO_ENCONTRADO"
                if not mercado_allowed:
                    operacional_ok = False
                    if motivo_nao_apostar == "ok":
                        motivo_nao_apostar = mercado_block_motivo
                    else:
                        motivo_nao_apostar = f"{motivo_nao_apostar};{mercado_block_motivo}"

            if modo_auditoria:
                # Auditoria histórica:
                # importa tudo que passou apenas no threshold do modelo.
                apostar_final = bool(apostar_conf)
            else:
                apostar_final = bool(
                    apostar_conf
                    and operacional_ok
                    and mercado_allowed
                    and context_ok
                )

            status_confianca = status_confianca_por_probabilidade(prob_evento, ciclo_vida_confianca)

            mercado_base, line_aposta, line_code = extrair_base_e_linha_mercado(mercado, meta.get("event"))
            mercado_descricao = descricao_mercado_linha(mercado, meta.get("event"))

            resultados.append({
                "data": data_jogo.date().isoformat(),
                "kickoff_at": data_jogo.isoformat(),
                "previsao_gerada_em": datetime.now().astimezone().isoformat(timespec="seconds"),
                "Round": jogo.get("Round", jogo.get("Rodada", "")),
                "liga": liga,
                "home": home,
                "away": away,
                "mercado": mercado,
                "modelo_contextual_tipo": modelo_contextual_info.get("tipo", "global"),
                "modelo_contextual_chave": modelo_contextual_info.get("chave", ""),
                "modelo_contextual_score": modelo_contextual_info.get("score", ""),
                "modelo_contextual_auc": modelo_contextual_info.get("auc", ""),
                "modelo_contextual_n_train": modelo_contextual_info.get("n_train", ""),
                "modelo_contextual_n_test": modelo_contextual_info.get("n_test", ""),
                "prob_global": modelo_contextual_info.get("prob_global", ""),
                "prob_final_fusion": modelo_contextual_info.get("prob_final", ""),
                "fusion_enabled": modelo_contextual_info.get("fusion_enabled", ""),
                "fusion_models": modelo_contextual_info.get("fusion_models", ""),
                "fusion_detail": modelo_contextual_info.get("fusion_detail", ""),
                "mercado_status": mercado_status,
                "mercado_allowed": bool(mercado_allowed),
                "block_motivo": mercado_block_motivo,
                "motivo_bloqueio": mercado_block_motivo,
                "status_mercado": mercado_status,
                "modo_auditoria": bool(modo_auditoria),
                "entrada_simulada": bool(apostar_final),
                "operacao_real": bool(apostar_final) if modo_teste_real else (False if modo_auditoria else bool(apostar_final)),
                "recomendacao_operacional": bool(apostar_final) if modo_teste_real else (False if modo_auditoria else bool(apostar_final)),
                "motivo_auditoria": "BACKFILL_HISTORICO" if modo_auditoria else ("TESTE_REAL_PROTECAO_SIMULACAO" if modo_teste_real else ""),
                "mercado_base": mercado_base,
                "line_aposta": line_aposta,
                "line_code": line_code,
                "mercado_descricao": mercado_descricao,
                "event": meta.get("event") or meta.get("name") or mercado,
                "prob": round(prob_modelo, 4),
                "prob_modelo": round(prob_modelo, 4),
                "probabilidade_bruta": round(probabilidade_bruta, 4),
                "probabilidade_calibrada": round(probabilidade_calibrada, 4),
                "calibration_method": calibration_method,
                "calibrador_encontrado": bool(calibrator_found),
                "calibrador_ausente_bloqueia": bool(calibrator_absence_blocks),
                "status_confianca": status_confianca,
                "prob_evento": round(prob_evento, 4),
                "confianca": round(confianca, 4),
                "odd": round(float(odd), 2),
                "tem_odd_real": bool(tem_odd_real),
                "ev": round(float(ev), 4),
                "apostar": apostar_final,
                "motivo_nao_apostar": (
                    "BACKFILL_HISTORICO"
                    if modo_auditoria
                    else ("ok" if apostar_final else motivo_nao_apostar)
                ),
                "context_operacional_ok": bool(context_ok),
                "context_block_motivo": "" if context_ok else context_motivo,
                "context_score_final": checks_contexto.get("context_score_final", 50.0),
                "context_has_positive": bool(checks_contexto.get("context_has_positive", False)),
                "context_has_negative": bool(checks_contexto.get("context_has_negative", False)),
                "context_positive_reason": checks_contexto.get("context_positive_reason", ""),
                "context_negative_reason": checks_contexto.get("context_negative_reason", ""),
                "context_policy": checks_contexto.get("context_policy", "CONTEXTUAL_REAL"),
                "context_bootstrap": bool(checks_contexto.get("context_bootstrap", False)),
                "context_data_status": checks_contexto.get("context_data_status", "DESCONHECIDO"),
                "context_block_reason": checks_contexto.get("context_block_reason", context_motivo if not context_ok else ""),
                "contexto_arquivo_carregado": bool(checks_contexto.get("contexto_arquivo_carregado", False)),
                "context_liga_ok": bool(checks_contexto.get("context_liga_ok", True)),
                "context_home_ok": bool(checks_contexto.get("context_home_ok", True)),
                "context_away_ok": bool(checks_contexto.get("context_away_ok", True)),
                "context_liga_home_ok": bool(checks_contexto.get("context_liga_home_ok", True)),
                "context_liga_away_ok": bool(checks_contexto.get("context_liga_away_ok", True)),
                "filtro_operacional_ok": bool(operacional_ok),
                "odd_ok": bool(checks_operacionais["odd_ok"]),
                "auc_ok": bool(checks_operacionais["auc_ok"]),
                "roi_bt_ok": bool(checks_operacionais["roi_bt_ok"]),
                "confianca_edge_ok": bool(checks_operacionais["confianca_edge_ok"]),
                "margem_confianca": round(float(margem_confianca), 4),
                "perfil_operacional_ativo": bool(cfg_mercado_operacional["ativo"]),
                "perfil_operacional_status": cfg_mercado_operacional.get("status", "SEM_PERFIL"),
                "perfil_operacional_motivo": cfg_mercado_operacional.get("motivo", ""),
                "odd_min": cfg_mercado_operacional["odd_min"],
                "odd_max": cfg_mercado_operacional["odd_max"],
                "min_ev": cfg_mercado_operacional["min_ev"],
                "min_model_auc": MIN_MODEL_AUC,
                "min_confianca_edge": MIN_CONFIDENCE_EDGE,
                "max_bets_per_game": MAX_BETS_PER_GAME,
                "roi_bt": meta.get("roi_bt"),
                "winrate_bt": meta.get("winrate_bt"),
                "auc": meta.get("auc"),
                "threshold_raw": threshold_raw,
                "threshold_usado": round(float(threshold_usado), 4),
                "threshold_fonte": threshold_fonte,
                "prob_extrema": bool(prob_extrema),
                "prob_100": bool(prob_100),
            })

    if modo_auditoria:
        output_path = historical_prediction_path(data_alvo, ensure_dir=True)
    else:
        output_path = normal_prediction_path(data_alvo, ensure_dir=True)
    df_out = pd.DataFrame(resultados)
    df_out = aplicar_limite_apostas_por_jogo(df_out)
    if modo_auditoria and not df_out.empty:
        df_out["modo_auditoria"] = True
        df_out["operacao_real"] = False
        df_out["recomendacao_operacional"] = False
        df_out["motivo_auditoria"] = "BACKFILL_HISTORICO"
    if not df_out.empty:
        temp_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            df_out.to_csv(temp_output, index=False, encoding="utf-8-sig")
            os.replace(temp_output, output_path)
        finally:
            if temp_output.exists():
                temp_output.unlink()
    elif output_path.exists():
        # Uma nova execução sem resultados não pode deixar o CSV antigo parecer
        # uma previsão atual válida para terminal, dashboard ou retomada.
        output_path.unlink()

    total_previsoes = int(len(df_out))
    apostas = int(df_out["apostar"].sum()) if "apostar" in df_out.columns else 0
    previsoes_com_odd_real = int(df_out["tem_odd_real"].sum()) if "tem_odd_real" in df_out.columns else 0
    previsoes_sem_odd_real = total_previsoes - previsoes_com_odd_real
    previsoes_prob_extrema = int(df_out["prob_extrema"].sum()) if "prob_extrema" in df_out.columns else 0
    previsoes_prob_100 = int(df_out["prob_100"].sum()) if "prob_100" in df_out.columns else 0
    bloqueadas_odd = int((~df_out["odd_ok"].astype(bool)).sum()) if "odd_ok" in df_out.columns else 0
    bloqueadas_auc = int((~df_out["auc_ok"].astype(bool)).sum()) if "auc_ok" in df_out.columns else 0
    bloqueadas_conf_edge = int((~df_out["confianca_edge_ok"].astype(bool)).sum()) if "confianca_edge_ok" in df_out.columns else 0
    bloqueadas_contexto = int((~df_out["context_operacional_ok"].astype(bool)).sum()) if "context_operacional_ok" in df_out.columns else 0
    bloqueadas_limite_jogo = int((~df_out["limite_1_aposta_por_jogo_ok"].astype(bool)).sum()) if "limite_1_aposta_por_jogo_ok" in df_out.columns else 0
    bloqueadas_mercado_gov = int((~df_out["mercado_allowed"].astype(bool)).sum()) if "mercado_allowed" in df_out.columns else 0
    print("\n" + "=" * 60)
    print(f"🎯 {'PREVISÕES HISTÓRICAS — MODO AUDITORIA' if modo_auditoria else 'PREVISÕES DO DIA'} — {data_alvo}")
    print("=" * 60)
    print(f"Jogos carregados       : {jogos_carregados}")
    print(f"Jogos válidos          : {jogos_validos}")
    print(f"Jogos ignorados        : {jogos_ignorados}")
    print(f"Modelos carregados     : {len(modelos)}")
    if modo_auditoria:
        print(f"Modo auditoria         : True")
        print(f"Operação real          : False")
        print(f"Recomendação operacional: False")
        print("Mercados carregados    : " + (", ".join(diagnostico_modelos.get("mercados_carregados", [])[:20]) or "nenhum"))
        print("Mercados sem modelo    : " + (", ".join(diagnostico_modelos.get("mercados_sem_modelo", [])[:20]) or "nenhum"))
        print("Ignorados erro técnico : " + (", ".join(diagnostico_modelos.get("mercados_ignorados_por_erro_tecnico", [])[:20]) or "nenhum"))
    print(f"Total de previsões     : {total_previsoes}")
    print(f"Apostas recomendadas   : {apostas}")
    print(f"Previsões com odd real : {previsoes_com_odd_real}")
    print(f"Previsões sem odd real : {previsoes_sem_odd_real}")
    print(f"Probabilidades extremas: {previsoes_prob_extrema}")
    print(f"Probabilidades 100%    : {previsoes_prob_100}")
    print(f"Bloqueadas por odd     : {bloqueadas_odd}")
    print(f"Bloqueadas por AUC     : {bloqueadas_auc}")
    print(f"Bloqueadas por margem  : {bloqueadas_conf_edge}")
    print(f"Bloqueadas por contexto: {bloqueadas_contexto}")
    print(f"Bloqueadas por jogo    : {bloqueadas_limite_jogo}")
    print(f"Bloqueadas por mercado : {bloqueadas_mercado_gov}")
    if df_out.empty:
        print("Arquivo salvo           : não gerado; sem previsões válidas.")
        print("Motivo provável         : falta de histórico anterior, modelos ou features compatíveis.")
    else:
        print(f"Arquivo salvo           : {output_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera previsões usando FutPython Daily.")
    parser.add_argument("--date", dest="data", default=date.today().isoformat(), help="Data no formato YYYY-MM-DD. Padrão: hoje.")
    parser.add_argument("--modo-auditoria", action="store_true", help="Gera previsões históricas sem operação real, sem recomendação operacional e com apostar=False.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pd.Timestamp(args.data)
    except Exception:
        sair_limpo("Data inválida. Use o formato YYYY-MM-DD.")
    prever(args.data, modo_auditoria=bool(args.modo_auditoria))


if __name__ == "__main__":
    main()
