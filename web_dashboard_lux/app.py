#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import hashlib
import os
import pickle
import socket
import time
import subprocess
import sys
import threading
import unicodedata
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
APP_STARTED_AT = time.time()
ML_DIR = ROOT / "04_ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from utils.prediction_paths import (  # noqa: E402
    HISTORICAL_PREDICTIONS_DIR,
    PREDICTIONS_DIR,
    available_prediction_dates as list_prediction_dates,
    latest_prediction_path,
    prediction_date_from_path,
    prediction_path_for_date as resolve_prediction_path_for_date,
)

BASE_OFICIAL = ROOT / "data" / "base_oficial.csv"
BACKTEST_RESUMO = ROOT / "03_backtest" / "results" / "resumo.csv"
MODELOS_RESUMO = ROOT / "04_ml" / "models" / "resumo_modelos.pkl"
BASELINE_MODELOS = ROOT / "04_ml" / "models" / "baseline_metrics.json"
HISTORICO = ROOT / "04_ml" / "banca" / "historico_apostas.csv"
BANCA_ESTADO = ROOT / "04_ml" / "banca" / "banca_estado.json"
PERFIL = ROOT / "04_ml" / "reports" / "perfil_operacional_mercados.json"
STATUS_OPERACIONAL = ROOT / "04_ml" / "reports" / "status_operacional.json"
STATUS_AMBIENTE = ROOT / "04_ml" / "reports" / "status_ambiente.json"
SAUDE_OPERACIONAL = ROOT / "04_ml" / "reports" / "saude_operacional.json"
CONTEXT_SCORES = ROOT / "04_ml" / "reports" / "context_operational_scores.json"
STATUS_DASHBOARD = ROOT / "04_ml" / "reports" / "status_dashboard.json"
PAPER_MONITOR = ROOT / "04_ml" / "reports" / "paper_monitor.json"
PAPER_CONFIG = ROOT / "04_ml" / "config" / "paper_mode.json"
AUTOMATION_STATE = ROOT / "04_ml" / "reports" / "automation_state.json"
CACHE_STATUS = ROOT / "cache" / "status.json"
PREV_DIR = PREDICTIONS_DIR
PREV_HIST_DIR = HISTORICAL_PREDICTIONS_DIR
RAW_DAILY_DIRS = [
    ROOT / "data" / "raw" / "daily",
    ROOT / "data" / "raw" / "futpython" / "daily",
]
TASK_STATUS = APP_DIR / "cache" / "task_status.json"
TASK_LOG = APP_DIR / "cache" / "task.log"
CRESTS_DIR = APP_DIR / "static" / "crests"
LEAGUES_DIR = APP_DIR / "static" / "leagues"

_DATAFRAME_CACHE: dict[str, tuple[int, int, pd.DataFrame]] = {}
_JSON_CACHE: dict[str, tuple[int, int, Any]] = {}
_PICKLE_CACHE: dict[str, tuple[int, int, Any]] = {}
_ROW_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}
_CACHE_LOCK = threading.RLock()

LOGS = [
    TASK_LOG,
    ROOT / "pipeline.log",
    ROOT / "logs" / "pipeline.log",
    ROOT / "cache" / "autopilot.log",
    ROOT / "04_ml" / "logs" / "pipeline.log",
]

TASK_LOCK = threading.Lock()

app = FastAPI(title="Football Decision Lab Control Center", version="4.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def age(path: Path) -> str:
    if not path.exists():
        return "ausente"
    try:
        sec = max(0, int((datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()))
        if sec < 60:
            return "agora"
        if sec < 3600:
            return f"{sec // 60} min"
        if sec < 172800:
            return f"{sec // 3600} h"
        return f"{sec // 86400} d"
    except Exception:
        return "-"


def _prediction_meta_files() -> list[Path]:
    files = list(PREV_DIR.glob("previsoes_*.csv")) if PREV_DIR.exists() else []
    paper_dir = active_paper_predictions_dir()
    if paper_dir.exists():
        files.extend(paper_dir.glob("previsoes_*.csv"))
    return sorted(path for path in files if path.is_file() and path.stat().st_size > 0)


def _version_from_prediction_meta() -> dict[str, Any]:
    files = _prediction_meta_files()
    latest = files[-1] if files else None
    latest_stat = latest.stat() if latest else None
    version_parts = []
    for path in files:
        try:
            stat = path.stat()
            version_parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            continue
    versao = hashlib.sha256("|".join(version_parts).encode("utf-8")).hexdigest()[:16] if version_parts else "sem-previsoes"
    ultima_data = prediction_date_from_path(latest, fallback="") if latest else None
    return {
        "versao_dados": versao,
        "ultima_previsao_data": ultima_data,
        "ultimo_arquivo": str(latest.relative_to(ROOT)) if latest else None,
        "ultimo_arquivo_tamanho": latest_stat.st_size if latest_stat else 0,
        "ultimo_arquivo_modificado_em": datetime.fromtimestamp(latest_stat.st_mtime).isoformat(timespec="seconds") if latest_stat else None,
        "quantidade_arquivos": len(files),
        "pipeline_status": "desconhecido",
        "etapa_atual": "monitoramento",
        "datas_concluidas": len(files),
        "datas_totais": None,
        "percentual": None,
        "ultima_atualizacao": datetime.now().isoformat(timespec="seconds"),
        "nova_previsao_disponivel": False,
    }


def dashboard_version() -> dict[str, Any]:
    data = read_json(STATUS_DASHBOARD, {})
    computed = _version_from_prediction_meta()
    if not isinstance(data, dict) or not data:
        return computed
    if data.get("versao_dados") == computed.get("versao_dados"):
        return data

    # O predict também pode ser executado fora do regenerador que mantém
    # status_dashboard.json. Nesse caso, os arquivos são a fonte observável mais
    # recente e não devemos exibir contagem/data antigas no painel.
    reconciled = dict(data)
    reconciled.update(computed)
    reconciled["pipeline_status"] = "sincronizado_por_arquivos"
    reconciled["etapa_atual"] = "reconciliacao_automatica"
    reconciled["nova_previsao_disponivel"] = True
    return reconciled


def light_status() -> dict[str, Any]:
    version = dashboard_version()
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    return {
        "agora": agora,
        "ip": local_ip(),
        "modo": "MONITOR",
        "health": 0,
        "health_label": "Leve",
        "health_score_oficial": None,
        "alertas": ["Toque em Atualizar agora para carregar o status completo."],
        "base_linhas": 0,
        "idade_base": "-",
        "idade_backtest": "-",
        "idade_modelos": "-",
        "idade_previsoes": age(ROOT / str(version["ultimo_arquivo"])) if version.get("ultimo_arquivo") else "ausente",
        "operacional": {},
        "ambiente": {},
        "previsoes": {
            "total": 0,
            "sinais": 0,
            "jogos": 0,
            "arquivo": version.get("ultimo_arquivo") or "-",
            "linhas": [],
            "top": [],
            "datas_disponiveis": [],
        },
        "historico": {
            "total": 0,
            "ganhou": 0,
            "perdeu": 0,
            "pendente": 0,
            "saldo": 0,
            "lucro": 0,
            "roi": 0,
            "winrate": 0,
            "drawdown": 0,
            "stake_total": 0,
            "stake_media": 0,
            "curva": [],
            "por_mercado": [],
            "por_liga": [],
            "linhas": [],
            "heatmap": [],
        },
        "paper": {
            "enabled": True, "status": "AGUARDANDO", "cycle_id": "", "metrics": {},
            "sample_progress": {}, "por_mercado": [], "por_liga": [], "por_mes": [],
            "automation": {},
        },
        "mercados": {"qtd": 0, "itens": []},
        "modelos": {
            "total": 0,
            "aprovados": 0,
            "contextos": 0,
            "auc_medio": 0,
            "brier": None,
            "itens": [],
        },
        "task": task_state(),
        "logs": [],
        "telemetry": {
            "cpu": None,
            "ram": None,
            "process_mb": None,
            "uptime": "-",
        },
        "dashboard_version": version,
    }


def rows_csv(path: Path) -> int:
    signature = _file_signature(path)
    if signature is None or signature[1] <= 0:
        return 0
    key = str(path.resolve())
    with _CACHE_LOCK:
        cached = _ROW_COUNT_CACHE.get(key)
        if cached and cached[:2] == signature:
            return cached[2]
    try:
        with path.open("rb") as f:
            count = max(0, sum(1 for _ in f) - 1)
        with _CACHE_LOCK:
            _ROW_COUNT_CACHE[key] = (signature[0], signature[1], count)
        return count
    except Exception:
        return 0


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def read_csv(path: Path) -> pd.DataFrame:
    """Lê CSV somente quando o arquivo mudou.

    O DataFrame guardado nunca é devolvido diretamente: uma cópia rasa protege
    o cache das conversões e colunas temporárias feitas pelas telas.
    """
    signature = _file_signature(path)
    if signature is None or signature[1] <= 0:
        return pd.DataFrame()
    key = str(path.resolve())
    with _CACHE_LOCK:
        cached = _DATAFRAME_CACHE.get(key)
        if cached and cached[:2] == signature:
            return cached[2].copy(deep=True)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()
    with _CACHE_LOCK:
        _DATAFRAME_CACHE[key] = (signature[0], signature[1], frame)
    return frame.copy(deep=True)


def read_json(path: Path, default: Any) -> Any:
    signature = _file_signature(path)
    if signature is None or signature[1] <= 0:
        return default
    key = str(path.resolve())
    with _CACHE_LOCK:
        cached = _JSON_CACHE.get(key)
        if cached and cached[:2] == signature:
            return cached[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    with _CACHE_LOCK:
        _JSON_CACHE[key] = (signature[0], signature[1], payload)
    return payload


def read_pickle(path: Path, default: Any) -> Any:
    signature = _file_signature(path)
    if signature is None or signature[1] <= 0:
        return default
    key = str(path.resolve())
    with _CACHE_LOCK:
        cached = _PICKLE_CACHE.get(key)
        if cached and cached[:2] == signature:
            return cached[2]
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return default
    with _CACHE_LOCK:
        _PICKLE_CACHE[key] = (signature[0], signature[1], payload)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "sim", "s", "yes", "y"}:
        return True
    try:
        return float(normalized) == 1.0
    except (TypeError, ValueError):
        return False


def yes_count(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].map(as_bool).sum())


def fmt_float(v: Any, nd: int = 2) -> str:
    try:
        value = float(v)
        if pd.isna(value):
            return ""
        return f"{value:.{nd}f}"
    except Exception:
        return ""


def safe_pct(v: Any) -> float:
    try:
        value = float(v)
        if pd.isna(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def slugify_asset(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def asset_url(folder: Path, public_folder: str, name: Any) -> str:
    slug = slugify_asset(name)
    if not slug:
        return ""
    path = folder / f"{slug}.png"
    return f"/static/{public_folder}/{slug}.png" if path.exists() else ""


def split_match_name(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    for separator in (" x ", " X ", " vs ", " VS ", " v "):
        if separator in text:
            home, away = text.split(separator, 1)
            return home.strip(), away.strip()
    return text, ""


def formatar_nome_previsao(path: Path | None) -> str:
    """Retorna apenas a data do arquivo de previsões no formato DD/MM/YYYY."""
    if path is None:
        return "Sem arquivo"

    nome = path.stem
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", nome)
    if not match:
        return path.name

    ano, mes, dia = match.groups()
    return f"{dia}/{mes}/{ano}"


def match_stats_url(home: Any, away: Any, match_date: Any = "", explicit_url: Any = "") -> str:
    """Retorna uma URL segura para consultar placar e estatísticas públicas.

    Usa a URL já presente no CSV quando existir. Caso contrário, monta uma
    pesquisa direcionada ao SofaScore com times e data da partida.
    """
    explicit = str(explicit_url or "").strip()
    if explicit.startswith(("https://", "http://")):
        return explicit

    home_name = str(home or "").strip()
    away_name = str(away or "").strip()
    date_text = str(match_date or "").strip()
    if not home_name or not away_name:
        return ""

    query = f'site:sofascore.com/pt/football/match "{home_name}" "{away_name}" {date_text}'
    return "https://www.google.com/search?q=" + quote_plus(query)


def format_result_search_date(value: Any) -> str:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        pass
    try:
        return datetime.strptime(text, "%d/%m/%Y").strftime("%d/%m/%Y")
    except Exception:
        return text


def result_search_url(home: Any, away: Any, match_date: Any, resultado: Any) -> str:
    if str(resultado or "").strip().lower() != "pendente":
        return ""
    home_name = str(home or "").strip()
    away_name = str(away or "").strip()
    date_text = format_result_search_date(match_date)
    if not home_name or not away_name or not date_text:
        return ""
    return "https://www.google.com/search?q=" + quote_plus(f"{home_name} {away_name} {date_text}")


def active_paper_predictions_dir() -> Path:
    config = read_json(PAPER_CONFIG, {})
    return ROOT / "04_ml" / "previsoes_paper" / str(config.get("cycle_id", "_disabled"))


def available_prediction_dates() -> list[str]:
    """Lista as datas de todos os CSVs de previsão disponíveis."""
    dates = set(list_prediction_dates(include_historical=True, include_legacy=True))
    paper_dir = active_paper_predictions_dir()
    if paper_dir.exists():
        dates.update(prediction_date_from_path(path, fallback="") for path in paper_dir.glob("previsoes_*.csv"))
    return sorted(value for value in dates if value)


def prediction_path_for_date(selected_date: str | None) -> Path | None:
    if not selected_date:
        return None
    try:
        normalized = datetime.strptime(str(selected_date).strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None
    paper = active_paper_predictions_dir() / f"previsoes_{normalized}.csv"
    if paper.exists():
        return paper
    return resolve_prediction_path_for_date(normalized, include_historical=True, include_legacy=True)


def latest_prev_path(selected_date: str | None = None) -> Path | None:
    if selected_date:
        return prediction_path_for_date(selected_date)
    paper_dir = active_paper_predictions_dir()
    today_paper = paper_dir / f"previsoes_{date.today().isoformat()}.csv"
    if today_paper.exists():
        return today_paper
    paper_files = list(paper_dir.glob("previsoes_*.csv")) if paper_dir.exists() else []
    regular = latest_prediction_path(selected_date, include_historical=True, include_legacy=True)
    candidates = paper_files + ([regular] if regular else [])
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def prediction_source(path: Path | None) -> dict[str, Any]:
    """Classifica a origem sem transformar previsões antigas em sinais do paper."""
    config = read_json(PAPER_CONFIG, {})
    cycle_id = str(config.get("cycle_id", "")).strip()
    paper_dir = active_paper_predictions_dir()
    markets = [
        str(item.get("market", "")).strip()
        for item in config.get("models", [])
        if isinstance(item, dict) and str(item.get("market", "")).strip()
    ]
    if path is None:
        return {
            "oficial": False,
            "origem": "sem_arquivo",
            "rotulo_origem": "Aguardando previsões",
            "mensagem_origem": "Ainda não existe arquivo de previsões para a data selecionada.",
            "ciclo_id": cycle_id,
            "mercados_oficiais": markets,
        }
    is_official = bool(
        path.parent.resolve() == paper_dir.resolve()
        and bool(config.get("enabled"))
        and str(config.get("mode", "")).strip().lower() == "paper_only"
    )
    if is_official:
        return {
            "oficial": True,
            "origem": "paper_oficial",
            "rotulo_origem": "Ciclo paper atual",
            "mensagem_origem": (
                f"Indicações oficiais do ciclo {cycle_id}."
                if cycle_id else "Indicações oficiais do ciclo paper atual."
            ),
            "ciclo_id": cycle_id,
            "mercados_oficiais": markets,
        }
    return {
        "oficial": False,
        "origem": "legado_pre_ciclo",
        "rotulo_origem": "Consulta histórica — não é indicação",
        "mensagem_origem": (
            "Este arquivo foi gerado fora do ciclo paper atual. Suas previsões ficam "
            "guardadas para auditoria, mas não aparecem como apostas no painel."
        ),
        "ciclo_id": cycle_id,
        "mercados_oficiais": markets,
    }


def normalize_match_team(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\b(fc|cf|sc|afc|ac|club|futebol|football)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def raw_daily_file(match_date: Any) -> Path | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(match_date or ""))
    if not match:
        return None
    day = match.group(1)
    candidates: list[Path] = []
    for folder in RAW_DAILY_DIRS:
        if folder.exists():
            candidates.extend(folder.glob(f"*{day}*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flashscore_matches(match_date: Any) -> dict[tuple[str, str], dict[str, str]]:
    """Mapeia mandante/visitante ao ID do Flashscore salvo no raw daily."""
    path = raw_daily_file(match_date)
    df = read_csv(path) if path else pd.DataFrame()
    if df.empty:
        return {}

    columns = {str(col).strip().lower(): col for col in df.columns}
    id_col = next((columns[key] for key in ("id", "flashscore_id", "match_id", "event_id") if key in columns), None)
    home_col = next((columns[key] for key in ("home", "home_std", "mandante", "home_team") if key in columns), None)
    away_col = next((columns[key] for key in ("away", "away_std", "visitante", "away_team") if key in columns), None)
    if id_col is None or home_col is None or away_col is None:
        return {}

    out: dict[tuple[str, str], dict[str, str]] = {}
    for _, row in df.iterrows():
        match_id = str(row.get(id_col, "")).strip()
        home = str(row.get(home_col, "")).strip()
        away = str(row.get(away_col, "")).strip()
        if not match_id or match_id.lower() in {"nan", "none"}:
            continue
        key = (normalize_match_team(home), normalize_match_team(away))
        out[key] = {
            "id": match_id,
            "url": f"https://www.flashscore.com/match/{match_id}/#/match-summary",
        }
    return out


def previsoes(selected_date: str | None = None) -> dict[str, Any]:
    path = latest_prev_path(selected_date)
    source = prediction_source(path)
    df = read_csv(path) if path else pd.DataFrame()
    empty = {
        "arquivo": "ausente", "total": 0, "sinais": 0, "operacionais": 0,
        "sinais_no_arquivo": 0,
        "mercados": 0, "ligas": 0, "jogos": 0, "top": [], "linhas": [],
        "por_mercado": [], "por_liga": [],
        "datas_disponiveis": available_prediction_dates(),
        "data_selecionada": selected_date or "",
        **source,
    }
    if df.empty:
        return empty

    for col in ["ev", "decision_score", "confianca", "prob_modelo", "prob", "odd", "prob_evento", "auc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "apostar" not in df.columns:
        df["apostar"] = False
    df["_sinal_no_arquivo"] = df["apostar"].map(as_bool)
    if source["oficial"]:
        official_mask = df["_sinal_no_arquivo"].copy()
        if "paper_signal" in df.columns:
            official_mask &= df["paper_signal"].map(as_bool)
        if source["ciclo_id"] and "paper_cycle_id" in df.columns:
            official_mask &= df["paper_cycle_id"].fillna("").astype(str).eq(source["ciclo_id"])
        if "origem" in df.columns:
            official_mask &= df["origem"].fillna("").astype(str).str.strip().str.lower().eq("paper_forward")
        df["_apostar"] = official_mask
    else:
        df["_apostar"] = False
    sort_cols = [c for c in ["_apostar", "ev", "decision_score", "confianca", "prob_modelo"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    flash_map = flashscore_matches(selected_date or (path.stem if path else ""))

    def row_to_dict(r: pd.Series) -> dict[str, Any]:
        home = str(r.get("home", r.get("Home", ""))).strip()
        away = str(r.get("away", r.get("Away", ""))).strip()
        flash = flash_map.get((normalize_match_team(home), normalize_match_team(away)), {})
        explicit_match_url = r.get("flashscore_url", r.get("sofascore_url", r.get("match_url", "")))
        stats_url = str(explicit_match_url or "").strip() or str(flash.get("url", ""))
        if not stats_url:
            stats_url = match_stats_url(home, away, r.get("data", r.get("Date", "")), "")
        return {
            "data": str(r.get("data", r.get("Date", ""))),
            "liga": str(r.get("liga", r.get("League_std", ""))),
            "home": home,
            "away": away,
            "jogo": f"{home} x {away}".strip(" x "),
            "home_crest": asset_url(CRESTS_DIR, "crests", home),
            "away_crest": asset_url(CRESTS_DIR, "crests", away),
            "league_crest": asset_url(LEAGUES_DIR, "leagues", r.get("liga", r.get("League_std", ""))),
            "mercado": str(r.get("mercado", "")),
            "descricao": str(r.get("mercado_descricao", r.get("event", ""))),
            "odd": fmt_float(r.get("odd", ""), 2),
            "prob": fmt_float(r.get("prob_modelo", r.get("prob", "")), 3),
            "conf": fmt_float(r.get("confianca", r.get("prob_evento", "")), 3),
            "ev": fmt_float(r.get("ev", ""), 3),
            "auc": fmt_float(r.get("auc", ""), 3),
            "status": str(r.get("status_mercado", r.get("mercado_status", ""))),
            "motivo": str(r.get("motivo_nao_apostar", r.get("block_motivo", ""))),
            "contexto": str(r.get("context_positive_reason", r.get("context_block_motivo", ""))),
            "sinal": "Sim" if as_bool(r.get("_apostar", False)) else "Não",
            "sinal_no_arquivo": "Sim" if as_bool(r.get("_sinal_no_arquivo", False)) else "Não",
            "origem_previsao": source["origem"],
            "indicacao_oficial": bool(source["oficial"] and as_bool(r.get("apostar", False))),
            "match_status": str(r.get("match_status", r.get("status_jogo", r.get("Status", "")))),
            "home_score": fmt_float(r.get("home_score", r.get("G_H_FT", "")), 0),
            "away_score": fmt_float(r.get("away_score", r.get("G_A_FT", "")), 0),
            "flashscore_id": str(flash.get("id", "")),
            "stats_source": "Flashscore" if flash.get("id") else ("Link salvo" if explicit_match_url else "Pesquisa"),
            "stats_url": stats_url,
        }

    approved_df = df[df["_apostar"]]
    linhas = [row_to_dict(r) for _, r in approved_df.head(150).iterrows()]
    top_df = approved_df.head(12)
    top = [row_to_dict(r) for _, r in top_df.iterrows()]

    por_mercado: list[dict[str, Any]] = []
    if "mercado" in df.columns:
        g = df.groupby("mercado", dropna=False).agg(
            previsoes=("mercado", "size"),
            sinais=("_apostar", "sum"),
            ev_medio=("ev", "mean") if "ev" in df.columns else ("_apostar", "mean"),
        ).reset_index()
        for _, r in g.sort_values(["sinais", "ev_medio"], ascending=False).head(20).iterrows():
            por_mercado.append({
                "mercado": str(r["mercado"]), "previsoes": int(r["previsoes"]),
                "sinais": int(r["sinais"]), "ev_medio": safe_pct(r["ev_medio"]),
            })

    liga_col = "liga" if "liga" in df.columns else "League_std" if "League_std" in df.columns else None
    por_liga: list[dict[str, Any]] = []
    if liga_col:
        g = df.groupby(liga_col, dropna=False).agg(
            previsoes=(liga_col, "size"),
            sinais=("_apostar", "sum"),
            ev_medio=("ev", "mean") if "ev" in df.columns else ("_apostar", "mean"),
        ).reset_index()
        for _, r in g.sort_values(["sinais", "ev_medio"], ascending=False).head(20).iterrows():
            por_liga.append({
                "liga": str(r[liga_col]), "previsoes": int(r["previsoes"]),
                "sinais": int(r["sinais"]), "ev_medio": safe_pct(r["ev_medio"]),
            })

    game_keys = set()
    for _, r in df.iterrows():
        home = str(r.get("home", r.get("Home", ""))).strip()
        away = str(r.get("away", r.get("Away", ""))).strip()
        game_keys.add((home, away))

    return {
        "arquivo": formatar_nome_previsao(path),
        "arquivo_completo": str(path.relative_to(ROOT)) if path else "ausente",
        "total": int(len(df)),
        "sinais": int(df["_apostar"].sum()),
        "sinais_no_arquivo": int(df["_sinal_no_arquivo"].sum()),
        "operacionais": max(yes_count(df, "operacao_real"), yes_count(df, "recomendacao_operacional")),
        "mercados": int(df["mercado"].nunique()) if "mercado" in df.columns else 0,
        "ligas": int(df[liga_col].nunique()) if liga_col else 0,
        "jogos": len(game_keys),
        "top": top,
        "linhas": linhas,
        "por_mercado": por_mercado,
        "por_liga": por_liga,
        "datas_disponiveis": available_prediction_dates(),
        "data_selecionada": selected_date or (re.search(r"(\d{4}-\d{2}-\d{2})", path.stem).group(1) if path and re.search(r"(\d{4}-\d{2}-\d{2})", path.stem) else ""),
        **source,
    }


def historico() -> dict[str, Any]:
    df = read_csv(HISTORICO)
    estado_banca = read_json(BANCA_ESTADO, {})
    saldo_estado = safe_pct(estado_banca.get("banca_atual", 250.0)) if isinstance(estado_banca, dict) else 250.0
    empty = {
        "total": 0, "saldo": saldo_estado, "roi": 0.0, "lucro": 0.0, "ganhou": 0,
        "perdeu": 0, "pendente": 0, "winrate": 0.0, "stake_total": 0.0,
        "stake_media": 0.0, "drawdown": 0.0, "linhas": [], "por_mercado": [],
        "por_liga": [], "curva": [], "heatmap": [],
    }
    if df.empty:
        return empty

    for c in ["valor_apostado", "lucro", "banca_apos", "odd", "prob_modelo", "confianca"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    res = df["resultado"].astype(str).str.lower().str.strip() if "resultado" in df.columns else pd.Series([], dtype=str)
    settled_mask = res.isin(["ganhou", "perdeu"]) if not res.empty else pd.Series(False, index=df.index)
    settled = df.loc[settled_mask].copy()
    stake = float(settled["valor_apostado"].sum()) if "valor_apostado" in settled.columns else 0.0
    lucro = float(settled["lucro"].sum()) if "lucro" in settled.columns else 0.0
    banca_inicial = safe_pct(estado_banca.get("banca_inicial", 250.0)) if isinstance(estado_banca, dict) else 250.0
    if isinstance(estado_banca, dict) and estado_banca.get("banca_atual") is not None:
        saldo = safe_pct(estado_banca.get("banca_atual"))
    elif "banca_apos" in settled.columns and not settled["banca_apos"].dropna().empty:
        saldo = float(settled["banca_apos"].dropna().iloc[-1])
    else:
        saldo = banca_inicial + lucro

    if "banca_apos" in settled.columns and not settled.empty:
        curva_s = settled["banca_apos"].replace(0, pd.NA).ffill().fillna(banca_inicial)
    else:
        curva_s = banca_inicial + settled.get("lucro", pd.Series([0] * len(settled))).cumsum()
    ganhos = int((res == "ganhou").sum()) if not res.empty else 0
    perdas = int((res == "perdeu").sum()) if not res.empty else 0
    liquidadas = ganhos + perdas
    peaks = curva_s.cummax().replace(0, pd.NA)
    dd = ((curva_s - peaks) / peaks).fillna(0)
    drawdown = float(dd.min()) if not dd.empty else 0.0

    linhas = []
    for _, r in df.tail(60).iloc[::-1].iterrows():
        jogo = str(r.get("jogo", ""))
        home = str(r.get("home", "")).strip()
        away = str(r.get("away", "")).strip()
        if not home or not away:
            home, away = split_match_name(jogo)
        liga = str(r.get("liga", ""))
        resultado = str(r.get("resultado", ""))
        data_aposta = str(r.get("data", ""))
        linhas.append({
            "data": data_aposta, "jogo": jogo,
            "home": home, "away": away,
            "home_crest": asset_url(CRESTS_DIR, "crests", home),
            "away_crest": asset_url(CRESTS_DIR, "crests", away),
            "league_crest": asset_url(LEAGUES_DIR, "leagues", liga),
            "mercado": str(r.get("mercado", "")), "liga": liga,
            "odd": fmt_float(r.get("odd", ""), 2), "resultado": resultado,
            "lucro": fmt_float(r.get("lucro", 0), 2), "stake": fmt_float(r.get("valor_apostado", 0), 2),
            "resultado_url": result_search_url(home, away, data_aposta, resultado),
        })

    def group_performance(group_col: str, label: str) -> list[dict[str, Any]]:
        if group_col not in df.columns or "resultado" not in df.columns:
            return []
        tmp = df[res.isin(["ganhou", "perdeu"])].copy()
        if tmp.empty:
            return []
        g = tmp.groupby(group_col).agg(
            apostas=("resultado", "size"),
            ganhos=("resultado", lambda s: (s.astype(str).str.lower() == "ganhou").sum()),
            lucro=("lucro", "sum"),
            stake=("valor_apostado", "sum"),
        ).reset_index()
        g["roi"] = g["lucro"] / g["stake"].replace(0, pd.NA)
        g["winrate"] = g["ganhos"] / g["apostas"].replace(0, pd.NA)
        out = []
        for _, r in g.sort_values(["roi", "apostas"], ascending=False).head(30).iterrows():
            out.append({
                label: str(r[group_col]), "apostas": int(r["apostas"]),
                "lucro": float(r["lucro"]), "roi": safe_pct(r["roi"]),
                "winrate": safe_pct(r["winrate"]),
            })
        return out

    curve_len = len(curva_s)
    step = max(1, curve_len // 120)
    curva = [{"x": int(i + 1), "y": float(v)} for i, v in enumerate(curva_s.iloc[::step].tolist())]
    if curve_len and (not curva or curva[-1]["x"] != curve_len):
        curva.append({"x": curve_len, "y": float(curva_s.iloc[-1])})

    heatmap: list[dict[str, Any]] = []
    if "data" in df.columns and not res.empty:
        settled = df[res.isin(["ganhou", "perdeu"])].copy()
        if not settled.empty:
            dts = pd.to_datetime(settled["data"], dayfirst=True, errors="coerce")
            settled = settled.assign(_dt=dts)
            settled = settled[settled["_dt"].notna()]
            if not settled.empty:
                g = settled.groupby(settled["_dt"].dt.date).agg(
                    lucro=("lucro", "sum"),
                    apostas=("lucro", "size"),
                ).reset_index(names="_dia")
                heatmap = [
                    {"data": r["_dia"].isoformat(), "lucro": float(r["lucro"]), "apostas": int(r["apostas"])}
                    for _, r in g.iterrows()
                ]

    return {
        "total": int(len(df)), "saldo": saldo, "roi": lucro / stake if stake else 0.0,
        "lucro": lucro, "ganhou": ganhos, "perdeu": perdas,
        "pendente": int((res == "pendente").sum()) if not res.empty else 0,
        "winrate": ganhos / liquidadas if liquidadas else 0.0,
        "stake_total": stake, "stake_media": stake / liquidadas if liquidadas else 0.0,
        "drawdown": drawdown, "linhas": linhas,
        "por_mercado": group_performance("mercado", "mercado"),
        "por_liga": group_performance("liga", "liga"), "curva": curva,
        "heatmap": heatmap,
    }


def paper_dashboard() -> dict[str, Any]:
    config = read_json(PAPER_CONFIG, {})
    monitor = read_json(PAPER_MONITOR, {})
    automation = read_json(AUTOMATION_STATE, {})
    cycle_id = str(config.get("cycle_id", ""))
    frame = read_csv(HISTORICO)
    if not frame.empty and "origem" in frame.columns:
        mask = frame["origem"].fillna("").astype(str).str.strip().str.lower().eq("paper_forward")
        if cycle_id and "paper_cycle_id" in frame.columns:
            mask &= frame["paper_cycle_id"].fillna("").astype(str).eq(cycle_id)
        frame = frame[mask].copy()
    else:
        frame = pd.DataFrame()

    def aggregates(column: str, label: str) -> list[dict[str, Any]]:
        if frame.empty or column not in frame.columns or "resultado" not in frame.columns:
            return []
        result = frame["resultado"].fillna("").astype(str).str.lower()
        settled = frame[result.isin(["ganhou", "perdeu"])].copy()
        if settled.empty:
            return []
        settled["lucro"] = pd.to_numeric(settled.get("lucro"), errors="coerce").fillna(0)
        settled["valor_apostado"] = pd.to_numeric(settled.get("valor_apostado"), errors="coerce").fillna(0)
        grouped = settled.groupby(column, dropna=False).agg(
            apostas=("resultado", "size"),
            ganhos=("resultado", lambda values: values.astype(str).str.lower().eq("ganhou").sum()),
            lucro=("lucro", "sum"),
            stake=("valor_apostado", "sum"),
        ).reset_index()
        grouped["roi"] = grouped["lucro"] / grouped["stake"].replace(0, pd.NA)
        return [{
            label: str(row[column]), "apostas": int(row["apostas"]), "ganhos": int(row["ganhos"]),
            "lucro": float(row["lucro"]), "stake": float(row["stake"]), "roi": safe_pct(row["roi"]),
            "winrate": int(row["ganhos"]) / int(row["apostas"]) if int(row["apostas"]) else 0.0,
        } for _, row in grouped.sort_values(["apostas", "roi"], ascending=False).head(30).iterrows()]

    by_month = []
    if not frame.empty and "data" in frame.columns:
        frame["_mes"] = pd.to_datetime(frame["data"], errors="coerce").dt.strftime("%Y-%m")
        by_month = aggregates("_mes", "mes")

    paper_rows = []
    if not frame.empty:
        for _, row in frame.tail(60).iloc[::-1].iterrows():
            paper_rows.append({
                "data": str(row.get("data", "")),
                "jogo": str(row.get("jogo", "")),
                "mercado": str(row.get("mercado", "")),
                "odd": fmt_float(row.get("odd", ""), 2),
                "resultado": str(row.get("resultado", "")),
                "lucro": fmt_float(row.get("lucro", 0), 2),
            })

    heartbeat = automation.get("heartbeat_at") if isinstance(automation, dict) else None
    automation_status = "SEM_HEARTBEAT"
    heartbeat_minutes = None
    if heartbeat:
        try:
            heartbeat_dt = datetime.fromisoformat(str(heartbeat))
            now = datetime.now(heartbeat_dt.tzinfo) if heartbeat_dt.tzinfo else datetime.now()
            heartbeat_minutes = max(0, int((now - heartbeat_dt).total_seconds() / 60))
            automation_status = "EXECUTANDO" if automation.get("running") else ("OK" if heartbeat_minutes <= 45 else "ATRASADA")
        except Exception:
            pass
    automation_view = {
        "status": automation_status,
        "heartbeat_at": heartbeat,
        "heartbeat_minutes": heartbeat_minutes,
        "running": bool(automation.get("running")) if isinstance(automation, dict) else False,
        "current_job": automation.get("current_job") if isinstance(automation, dict) else None,
    }
    payload = {
        "enabled": bool(config.get("enabled")),
        "real_bets_allowed": bool(config.get("allow_real_bets", False)),
        "cycle_id": cycle_id,
        "policy_version": config.get("policy_version"),
        "started_at": config.get("started_at"),
        "review_not_before": config.get("review_not_before"),
        "timing": {
            "capture_min_minutes": config.get("capture_window_min_minutes"),
            "capture_max_minutes": config.get("capture_window_max_minutes"),
            "settlement_after_minutes": config.get("settlement_first_attempt_minutes_after_kickoff"),
        },
        "status": monitor.get("status", "AGUARDANDO"),
        "message": monitor.get("message", "O monitor paper ainda não foi executado."),
        "calibration_status": monitor.get("calibration_status", "SEM_AMOSTRA"),
        "metrics": monitor.get("metrics", {}),
        "sample_progress": monitor.get("sample_progress", {}),
        "por_mercado": aggregates("mercado", "mercado"),
        "por_liga": aggregates("liga", "liga"),
        "por_mes": by_month,
        "linhas": paper_rows,
        "automation": automation_view,
    }
    return payload


def mercados() -> dict[str, Any]:
    perfil = read_json(PERFIL, {})
    itens = []
    if isinstance(perfil, dict):
        for nome, cfg in perfil.items():
            if not isinstance(cfg, dict):
                continue
            itens.append({
                "mercado": nome,
                "ativo": cfg.get("ativo") is True,
                "status": str(cfg.get("status", "desconhecido")),
                "apostas": cfg.get("apostas", ""),
                "roi": safe_pct(cfg.get("roi", cfg.get("roi_total", cfg.get("roi_bt", 0)))),
                "auc": safe_pct(cfg.get("auc", 0)),
                "odd_min": cfg.get("odd_min", ""),
                "odd_max": cfg.get("odd_max", ""),
                "min_ev": cfg.get("min_ev", ""),
                "modelo": cfg.get("model_name", ""),
                "motivo": str(cfg.get("motivo", cfg.get("status_modelo", ""))),
            })
    itens.sort(key=lambda x: (not x["ativo"], -x["auc"], x["mercado"]))
    return {
        "itens": itens,
        "ativos": [x for x in itens if x["ativo"]],
        "qtd": sum(1 for x in itens if x["ativo"]),
        "total": len(itens),
        "backtest": rows_csv(BACKTEST_RESUMO),
    }


def modelos() -> dict[str, Any]:
    itens: list[dict[str, Any]] = []
    raw = read_pickle(MODELOS_RESUMO, [])
    if isinstance(raw, list):
        for m in raw:
            if not isinstance(m, dict):
                continue
            mercado = str(m.get("mercado") or m.get("market") or m.get("source") or "")
            ctx_file = ROOT / "04_ml" / "models" / mercado / "context_models.pkl"
            ctx = read_pickle(ctx_file, [])
            ctx_count = len(ctx) if isinstance(ctx, list) else 0
            itens.append({
                "mercado": mercado,
                "modelo": str(m.get("model_name", m.get("model", "Modelo"))),
                "auc": safe_pct(m.get("auc", 0)),
                "accuracy": safe_pct(m.get("accuracy_60", m.get("accuracy", 0))),
                "coverage": safe_pct(m.get("coverage", 0)),
                "roi_bt": safe_pct(m.get("roi_bt", 0)),
                "score": safe_pct(m.get("score", 0)),
                "contextos": ctx_count,
                "status": str(m.get("status_modelo", m.get("status", "approved"))),
            })
    itens.sort(key=lambda x: (-x["auc"], x["mercado"]))
    baseline = read_json(BASELINE_MODELOS, {})
    brier = baseline.get("brier_medio", baseline.get("mean_brier", baseline.get("brier_score", 0))) if isinstance(baseline, dict) else 0
    return {
        "total": len(itens),
        "aprovados": sum(1 for x in itens if str(x["status"]).lower() in {"approved", "aprovado"}),
        "contextos": sum(int(x["contextos"]) for x in itens),
        "brier": safe_pct(brier),
        "auc_medio": sum(x["auc"] for x in itens) / len(itens) if itens else 0.0,
        "itens": itens,
    }


def logs() -> list[str]:
    path = next((p for p in LOGS if p.exists() and p.stat().st_size > 0), None)
    if not path:
        return ["Sem log principal encontrado."]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        clean = [line for line in lines[-400:] if "DtypeWarning" not in line and "low_memory=False" not in line]
        return clean[-220:]
    except Exception as exc:
        return [f"Erro ao ler log: {exc}"]


def task_state() -> dict[str, Any]:
    return read_json(TASK_STATUS, {
        "running": False, "task": None, "step": None, "success": None,
        "error": None, "progress": 0,
    })



def telemetry() -> dict[str, Any]:
    """Retorna saúde do notebook e detecta pipelines iniciados fora do painel.

    As tarefas disparadas pelos botões usam ``task_status.json``. Já um backfill
    iniciado pelo terminal não atualiza esse arquivo; por isso a detecção abaixo
    inspeciona os processos Python em execução e informa o estágio encontrado.
    """
    uptime_seconds = max(0, int(time.time() - APP_STARTED_AT))
    out: dict[str, Any] = {
        "cpu": None,
        "ram": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "uptime_seconds": uptime_seconds,
        "uptime": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m",
        "process_mb": None,
        "external_pipeline_running": False,
        "external_pipeline_name": "",
        "external_pipeline_detail": "Aguardando",
        "external_pipeline_pid": None,
    }
    if psutil is None:
        return out

    try:
        vm = psutil.virtual_memory()
        out.update({
            "cpu": round(float(psutil.cpu_percent(interval=None)), 1),
            "ram": round(float(vm.percent), 1),
            "ram_used_gb": round(float(vm.used) / (1024 ** 3), 1),
            "ram_total_gb": round(float(vm.total) / (1024 ** 3), 1),
            "process_mb": round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 2), 1),
        })
    except Exception:
        pass

    process_labels = [
        ("backfill_daily_predictions.py", "Backfill de previsões"),
        ("03_predict.py", "Gerando previsões"),
        ("01_fetch_futpython_daily.py", "Atualizando jogos"),
        ("06_importar_backfill_banca.py", "Importando para a banca"),
        ("05_settle_historico.py", "Liquidando histórico"),
        ("05_settle_flashscore.py", "Consultando resultados no Flashscore"),
        ("04_banca.py", "Reconstruindo banca"),
        ("10_gerar_scores_contexto.py", "Atualizando scores de contexto"),
        ("09_gerar_perfil_operacional_mercados.py", "Atualizando perfil de mercados"),
        ("14_model_registry.py", "Atualizando registro de modelos"),
        ("run_pipeline.py", "Pipeline completo"),
        ("terminal.py", "Centro de comando"),
    ]

    try:
        current_pid = os.getpid()
        matches: list[tuple[int, str, str, int]] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                pid = int(proc.info.get("pid") or 0)
                if not pid or pid == current_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                command = " ".join(str(part) for part in cmdline).strip()
                if not command:
                    continue
                command_lower = command.lower().replace("\\", "/")
                for priority, (needle, label) in enumerate(process_labels):
                    if needle.lower() in command_lower:
                        # Menor prioridade vence: preferimos o estágio filho atual
                        # (predict/settlement) ao processo pai genérico do terminal.
                        matches.append((priority, label, command, pid))
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue

        if matches:
            priority, label, command, pid = sorted(matches, key=lambda item: item[0])[0]
            out.update({
                "external_pipeline_running": True,
                "external_pipeline_name": label,
                "external_pipeline_detail": f"Processo externo detectado · PID {pid}",
                "external_pipeline_pid": pid,
            })
    except Exception:
        pass

    return out

def status(selected_date: str | None = None) -> dict[str, Any]:
    cache = read_json(CACHE_STATUS, {})
    operacional = read_json(STATUS_OPERACIONAL, {})
    ambiente = read_json(STATUS_AMBIENTE, {})
    saude = read_json(SAUDE_OPERACIONAL, {})
    prev = previsoes(selected_date)
    hist = historico()
    merc = mercados()
    mods = modelos()
    paper = paper_dashboard()

    health = 100
    alertas = []
    if rows_csv(BASE_OFICIAL) == 0:
        health -= 30; alertas.append("Base oficial ausente")
    if prev["total"] == 0:
        health -= 12; alertas.append("Previsões não encontradas")
    if hist["total"] == 0:
        health -= 8; alertas.append("Histórico vazio")
    paper_bundle = ROOT / "04_ml" / "paper_models" / str(paper.get("cycle_id", "")) / "manifest.json"
    paper_ready = bool(paper.get("enabled")) and paper_bundle.exists()
    if merc["qtd"] == 0 and not paper_ready:
        health -= 20; alertas.append("Mercados ativos ausentes")
    if mods["total"] == 0 and not paper_ready:
        health -= 20; alertas.append("Modelos não encontrados")
    if operacional.get("permitir_previsoes") is False and not paper_ready:
        health -= 20; alertas.append(str(operacional.get("motivo", "Guard bloqueado")))
    if ambiente.get("status") == "BLOQUEADA":
        health -= 15; alertas.append("Ambiente bloqueado")
    if paper_ready:
        automation_status = str((paper.get("automation") or {}).get("status", "SEM_HEARTBEAT"))
        if automation_status in {"SEM_HEARTBEAT", "ATRASADA"}:
            health -= 20
            automation_label = {
                "SEM_HEARTBEAT": "sem sinal recente",
                "ATRASADA": "atrasada",
            }.get(automation_status, automation_status.lower())
            alertas.append(f"Automação da simulação: {automation_label}")
        if paper.get("calibration_status") == "ALERTA":
            health -= 15
            alertas.append("Calibração paper acima do limite de monitoramento")
        if paper.get("real_bets_allowed") is True:
            health -= 50
            alertas.append("Configuração insegura: apostas reais habilitadas")
    health = max(0, min(100, health))

    label = "Operacional" if health >= 90 else "Atenção" if health >= 70 else "Crítico"
    return {
        "agora": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "ip": local_ip(),
        "modo": "PAPER_ONLY" if paper_ready else str(cache.get("mode", operacional.get("status", "research"))).upper(),
        "health": health,
        "health_label": label,
        "health_score_oficial": saude.get("score") if isinstance(saude, dict) else None,
        "alertas": alertas or ["Tudo certo no momento"],
        "base_linhas": rows_csv(BASE_OFICIAL),
        "idade_base": age(BASE_OFICIAL),
        "idade_backtest": age(BACKTEST_RESUMO),
        "idade_modelos": age(MODELOS_RESUMO),
        "idade_previsoes": age(latest_prev_path() or PREV_DIR / "previsoes_ausente.csv"),
        "operacional": operacional,
        "ambiente": ambiente,
        "previsoes": prev,
        "historico": hist,
        "paper": paper,
        "mercados": merc,
        "modelos": mods,
        "task": task_state(),
        "logs": logs(),
        "telemetry": telemetry(),
    }


def commands_for(task_name: str) -> list[list[str]]:
    today = date.today().isoformat()
    py = sys.executable
    mapping = {
        "fetch_today": [[py, str(ROOT / "01_scripts" / "01_fetch_futpython_daily.py"), "--date", today]],
        "predict_today": [[py, str(ROOT / "04_ml" / "paper_predict.py"), "--date", today]],
        "import_bank": [[
            py, str(ROOT / "04_ml" / "06_registrar_paper.py"), "--date", today,
        ]],
        "settle": [[py, str(ROOT / "04_ml" / "05_settle_historico.py")]],
        "settle_flashscore_preview": [[
            py, str(ROOT / "04_ml" / "05_settle_flashscore.py"),
        ]],
        "settle_flashscore_apply": [[
            py, str(ROOT / "04_ml" / "05_settle_flashscore.py"), "--apply",
        ]],
        "settle_orchestrated": [[
            py, str(ROOT / "04_ml" / "05_settle_historico.py"), "--skip-post-update",
        ]],
        "scores_context": [[py, str(ROOT / "04_ml" / "10_gerar_scores_contexto.py")]],
        "context_report": [[py, str(ROOT / "04_ml" / "11_relatorio_contexto_operacional.py")]],
        "profile_markets": [[py, str(ROOT / "04_ml" / "09_gerar_perfil_operacional_mercados.py")]],
        "rebuild_bank": [[py, str(ROOT / "04_ml" / "04_banca.py"), "--rebuild-bank"]],
        "model_registry": [[py, str(ROOT / "04_ml" / "14_model_registry.py")]],
    }
    if task_name == "full_today":
        return (
            mapping["fetch_today"] + mapping["predict_today"] + mapping["import_bank"]
            + mapping["settle_orchestrated"] + mapping["rebuild_bank"]
        )
    if task_name == "update_bank":
        return (
            mapping["import_bank"] + mapping["settle_orchestrated"] + mapping["rebuild_bank"]
        )
    if task_name not in mapping:
        raise HTTPException(status_code=404, detail="Rotina não encontrada")
    return mapping[task_name]


def execute_task(task_name: str, commands: list[list[str]]) -> None:
    if not TASK_LOCK.acquire(blocking=False):
        return
    started = datetime.now().isoformat()
    try:
        TASK_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_json(TASK_STATUS, {
            "running": True, "task": task_name, "step": "iniciando", "success": None,
            "error": None, "progress": 0, "started_at": started,
        })
        with TASK_LOG.open("w", encoding="utf-8") as log:
            for idx, command in enumerate(commands, start=1):
                step = Path(command[1]).name if len(command) > 1 else command[0]
                progress = int(((idx - 1) / max(len(commands), 1)) * 100)
                write_json(TASK_STATUS, {
                    "running": True, "task": task_name, "step": step,
                    "step_number": idx, "total_steps": len(commands), "success": None,
                    "error": None, "progress": progress, "started_at": started,
                })
                log.write("\n" + "=" * 88 + "\n")
                log.write("$ " + " ".join(command) + "\n")
                log.write("=" * 88 + "\n")
                log.flush()
                proc = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    text=True, env=os.environ.copy(),
                )
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"{step} terminou com código {code}")
        write_json(TASK_STATUS, {
            "running": False, "task": task_name, "step": "finalizado", "success": True,
            "error": None, "progress": 100, "started_at": started,
            "finished_at": datetime.now().isoformat(),
        })
    except Exception as exc:
        write_json(TASK_STATUS, {
            "running": False, "task": task_name, "step": "erro", "success": False,
            "error": str(exc), "progress": 100, "started_at": started,
            "finished_at": datetime.now().isoformat(),
        })
    finally:
        TASK_LOCK.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "d": light_status()})


@app.get("/api/status")
async def api_status(date: str | None = None):
    return JSONResponse(status(date))


@app.get("/api/version")
async def api_version():
    return JSONResponse(dashboard_version())


@app.get("/api/task-status")
async def api_task_status():
    return JSONResponse(task_state())


@app.get("/api/task-log")
async def api_task_log():
    if not TASK_LOG.exists():
        return PlainTextResponse("")
    return PlainTextResponse("\n".join(TASK_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-220:]))


@app.post("/api/run/{task_name}")
async def run_task(task_name: str):
    if TASK_LOCK.locked():
        raise HTTPException(status_code=409, detail="Já existe uma execução em andamento")
    commands = commands_for(task_name)
    missing = [cmd[1] for cmd in commands if len(cmd) > 1 and not Path(cmd[1]).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Script não encontrado: {Path(missing[0]).name}")
    thread = threading.Thread(target=execute_task, args=(task_name, commands), daemon=True)
    thread.start()
    return {"ok": True, "task": task_name, "message": "Execução iniciada"}


if __name__ == "__main__":
    import uvicorn
    print("Football Decision Lab — Control Center")
    print("Notebook:", "http://127.0.0.1:8060")
    print("Celular :", f"http://{local_ip()}:8060")
    print("Modo    : monitoramento e execução local")
    uvicorn.run("app:app", host="0.0.0.0", port=8060, reload=False, app_dir=str(APP_DIR))

