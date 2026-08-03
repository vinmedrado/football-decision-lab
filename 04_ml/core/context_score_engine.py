#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context Score Engine — decisão operacional por liga/time/confronto.

Objetivo: impedir que um mercado globalmente bom entre em contextos ruins.
Não é blacklist fixa. O arquivo é regenerado pelo histórico liquidado e os
contextos podem voltar automaticamente quando o ROI melhorar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "04_ml" / "reports"
SCORE_PATH = REPORTS_DIR / "context_operational_scores.json"
HISTORY_PATH = ROOT_DIR / "04_ml" / "banca" / "historico_apostas.csv"

MIN_BETS_LEAGUE = 10
MIN_BETS_TEAM = 5
MIN_BETS_LEAGUE_TEAM = 3
MIN_BETS_MATCHUP = 3

# Contextos negativos com amostra suficiente bloqueiam a entrada.
BAD_ROI_LEAGUE = -0.05
BAD_ROI_TEAM = -0.08
BAD_ROI_LEAGUE_TEAM = -0.10
BAD_ROI_MATCHUP = -0.12

# Para entrar, precisa existir pelo menos um contexto bom confiável.
GOOD_ROI_LEAGUE = 0.03
GOOD_ROI_TEAM = 0.05
GOOD_ROI_LEAGUE_TEAM = 0.06
GOOD_ROI_MATCHUP = 0.08

# Score mínimo final para aprovar. Fica baixo para não matar amostra,
# mas exige evidência positiva real.
MIN_CONTEXT_SCORE = 52.0

_CACHE: Dict[str, Any] | None = None
_HISTORY_HAS_CLOSED_BETS: bool | None = None


def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def _key(*parts: Any) -> str:
    return "||".join(_norm(p) for p in parts)


def load_context_scores(path: Path | None = None) -> Dict[str, Any]:
    global _CACHE
    path = path or SCORE_PATH
    if _CACHE is not None:
        return _CACHE
    if not path.exists():
        _CACHE = {"loaded": False, "league": {}, "team": {}, "league_team": {}, "matchup": {}, "market": {}}
        return _CACHE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["loaded"] = True
        _CACHE = data
        return data
    except Exception:
        _CACHE = {"loaded": False, "league": {}, "team": {}, "league_team": {}, "matchup": {}, "market": {}}
        return _CACHE


def has_closed_bet_history(path: Path | None = None) -> bool:
    global _HISTORY_HAS_CLOSED_BETS
    path = path or HISTORY_PATH
    if path == HISTORY_PATH and _HISTORY_HAS_CLOSED_BETS is not None:
        return _HISTORY_HAS_CLOSED_BETS
    if not path.exists():
        result = False
    else:
        try:
            hist = pd.read_csv(path, low_memory=False, usecols=lambda c: str(c).lower() == "resultado")
            result = bool(hist["resultado"].astype(str).str.lower().isin(["ganhou", "perdeu"]).any())
        except Exception:
            result = False
    if path == HISTORY_PATH:
        _HISTORY_HAS_CLOSED_BETS = result
    return result


def _empty_metrics() -> Dict[str, float]:
    return {"apostas": 0, "roi": 0.0, "winrate": 0.0, "profit_factor": 0.0, "lucro": 0.0, "score": 50.0}


def _get(bucket: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    return dict(bucket.get(key) or _empty_metrics())


def _score_part(metrics: Dict[str, Any], min_bets: int) -> Tuple[float, float]:
    """Retorna (score_0_100, reliability_0_1)."""
    bets = float(metrics.get("apostas", 0) or 0)
    roi = float(metrics.get("roi", 0) or 0)
    wr = float(metrics.get("winrate", 0) or 0)
    pf = float(metrics.get("profit_factor", 0) or 0)
    reliability = max(0.0, min(1.0, bets / max(float(min_bets * 4), 1.0)))
    # 50 neutro; ROI é o principal. PF/WR refinam.
    score = 50.0 + (roi * 180.0) + ((pf - 1.0) * 12.0) + ((wr - 0.5) * 18.0)
    score = max(0.0, min(100.0, score))
    return score, reliability


def evaluate_context(
    *,
    mercado: str,
    liga: str,
    home: str,
    away: str,
    require_positive: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Avalia se a aposta pode entrar pelo contexto operacional.

    Regra:
    - Sem arquivo de scores: libera, mas marca SEM_SCORE.
    - Contexto ruim confiável bloqueia.
    - Se require_positive=True, precisa ter pelo menos um contexto bom confiável.
    - Score final ponderado precisa ficar acima do mínimo.
    """
    data = load_context_scores()
    mercado_k, liga_k, home_k, away_k = _norm(mercado), _norm(liga), _norm(home), _norm(away)
    if not data.get("loaded"):
        if has_closed_bet_history():
            return False, "CONTEXTO_AUSENTE_COM_HISTORICO_EXISTENTE", {
                "context_score_final": 50.0,
                "context_loaded": False,
                "context_has_positive": False,
                "context_has_negative": False,
                "context_policy": "CONTEXTUAL_INCONSISTENTE",
                "context_bootstrap": False,
                "context_data_status": "AUSENTE_COM_HISTORICO_EXISTENTE",
                "context_block_reason": "CONTEXTO_AUSENTE_COM_HISTORICO_EXISTENTE",
            }
        return True, "sem_score_contextual", {
            "context_score_final": 50.0,
            "context_loaded": False,
            "context_has_positive": False,
            "context_has_negative": False,
            "context_policy": "BOOTSTRAP_SEM_ARQUIVO_CONTEXTUAL",
            "context_bootstrap": True,
            "context_data_status": "SEM_HISTORICO_LIQUIDADO",
            "context_block_reason": "",
        }

    buckets = [data.get(name, {}) for name in ["market", "league", "team", "league_team", "matchup"]]
    context_file_empty = not any(isinstance(bucket, dict) and bucket for bucket in buckets)
    if context_file_empty and has_closed_bet_history():
        return False, "CONTEXTO_VAZIO_COM_HISTORICO_EXISTENTE", {
            "context_score_final": 50.0,
            "context_loaded": True,
            "context_has_positive": False,
            "context_has_negative": False,
            "context_positive_reason": "",
            "context_negative_reason": "",
            "context_policy": "CONTEXTUAL_INCONSISTENTE",
            "context_bootstrap": False,
            "context_data_status": "VAZIO_COM_HISTORICO_EXISTENTE",
            "context_block_reason": "CONTEXTO_VAZIO_COM_HISTORICO_EXISTENTE",
        }

    m_market = _get(data.get("market", {}), _key(mercado_k))
    m_league = _get(data.get("league", {}), _key(mercado_k, liga_k))
    m_home = _get(data.get("team", {}), _key(mercado_k, home_k))
    m_away = _get(data.get("team", {}), _key(mercado_k, away_k))
    m_lhome = _get(data.get("league_team", {}), _key(mercado_k, liga_k, home_k))
    m_laway = _get(data.get("league_team", {}), _key(mercado_k, liga_k, away_k))
    m_match = _get(data.get("matchup", {}), _key(mercado_k, liga_k, home_k, away_k))

    checks = []
    reasons = []

    def neg(metrics, min_bets, bad_roi, label):
        bets = float(metrics.get("apostas", 0) or 0)
        roi = float(metrics.get("roi", 0) or 0)
        if bets >= min_bets and roi <= bad_roi:
            reasons.append(f"{label}_ruim_roi_{roi:+.1%}_bets_{int(bets)}")
            return True
        return False

    def pos(metrics, min_bets, good_roi, label):
        bets = float(metrics.get("apostas", 0) or 0)
        roi = float(metrics.get("roi", 0) or 0)
        ok = bets >= min_bets and roi >= good_roi
        if ok:
            checks.append(f"{label}_bom_roi_{roi:+.1%}_bets_{int(bets)}")
        return ok

    has_negative = any([
        neg(m_league, MIN_BETS_LEAGUE, BAD_ROI_LEAGUE, "liga"),
        neg(m_home, MIN_BETS_TEAM, BAD_ROI_TEAM, "home"),
        neg(m_away, MIN_BETS_TEAM, BAD_ROI_TEAM, "away"),
        neg(m_lhome, MIN_BETS_LEAGUE_TEAM, BAD_ROI_LEAGUE_TEAM, "liga_home"),
        neg(m_laway, MIN_BETS_LEAGUE_TEAM, BAD_ROI_LEAGUE_TEAM, "liga_away"),
        neg(m_match, MIN_BETS_MATCHUP, BAD_ROI_MATCHUP, "confronto"),
    ])

    has_positive = any([
        pos(m_league, MIN_BETS_LEAGUE, GOOD_ROI_LEAGUE, "liga"),
        pos(m_home, MIN_BETS_TEAM, GOOD_ROI_TEAM, "home"),
        pos(m_away, MIN_BETS_TEAM, GOOD_ROI_TEAM, "away"),
        pos(m_lhome, MIN_BETS_LEAGUE_TEAM, GOOD_ROI_LEAGUE_TEAM, "liga_home"),
        pos(m_laway, MIN_BETS_LEAGUE_TEAM, GOOD_ROI_LEAGUE_TEAM, "liga_away"),
        pos(m_match, MIN_BETS_MATCHUP, GOOD_ROI_MATCHUP, "confronto"),
    ])

    # Score ponderado por confiabilidade. Mercado entra como base fraca; contexto manda.
    parts = []
    for metrics, min_bets, weight, label in [
        (m_market, 30, 0.10, "mercado"),
        (m_league, MIN_BETS_LEAGUE, 0.28, "liga"),
        (m_home, MIN_BETS_TEAM, 0.20, "home"),
        (m_away, MIN_BETS_TEAM, 0.20, "away"),
        (m_lhome, MIN_BETS_LEAGUE_TEAM, 0.10, "liga_home"),
        (m_laway, MIN_BETS_LEAGUE_TEAM, 0.08, "liga_away"),
        (m_match, MIN_BETS_MATCHUP, 0.04, "confronto"),
    ]:
        sc, rel = _score_part(metrics, min_bets)
        parts.append((sc, weight * rel, label))
    total_w = sum(w for _, w, _ in parts)
    score_final = (sum(sc * w for sc, w, _ in parts) / total_w) if total_w > 0 else 50.0

    if has_negative:
        return False, ";".join(reasons), {
            "context_score_final": round(score_final, 4),
            "context_loaded": True,
            "context_has_positive": has_positive,
            "context_has_negative": True,
            "context_positive_reason": ";".join(checks),
            "context_negative_reason": ";".join(reasons),
            "context_policy": "CONTEXTUAL_REAL",
            "context_bootstrap": False,
            "context_data_status": "CONTEXTUAL_VALIDO",
            "context_block_reason": ";".join(reasons),
        }
    if require_positive and not has_positive:
        has_reliable_context_sample = any([
            float(m_market.get("apostas", 0) or 0) >= 30,
            float(m_league.get("apostas", 0) or 0) >= MIN_BETS_LEAGUE,
            float(m_home.get("apostas", 0) or 0) >= MIN_BETS_TEAM,
            float(m_away.get("apostas", 0) or 0) >= MIN_BETS_TEAM,
            float(m_lhome.get("apostas", 0) or 0) >= MIN_BETS_LEAGUE_TEAM,
            float(m_laway.get("apostas", 0) or 0) >= MIN_BETS_LEAGUE_TEAM,
            float(m_match.get("apostas", 0) or 0) >= MIN_BETS_MATCHUP,
        ])
        if not has_reliable_context_sample:
            return True, "bootstrap_sem_historico_contextual_valido", {
                "context_score_final": round(score_final, 4),
                "context_loaded": True,
                "context_has_positive": False,
                "context_has_negative": False,
                "context_positive_reason": "",
                "context_negative_reason": "",
                "context_policy": "BOOTSTRAP_SEM_HISTORICO_VALIDO",
                "context_bootstrap": True,
                "context_data_status": "SEM_AMOSTRA_CONTEXTUAL_CONFIAVEL",
                "context_block_reason": "",
            }
        return False, "sem_contexto_positivo_confiavel", {
            "context_score_final": round(score_final, 4),
            "context_loaded": True,
            "context_has_positive": False,
            "context_has_negative": False,
            "context_positive_reason": "",
            "context_negative_reason": "",
            "context_policy": "CONTEXTUAL_REAL",
            "context_bootstrap": False,
            "context_data_status": "AMOSTRA_CONTEXTUAL_CONFIAVEL_SEM_POSITIVO",
            "context_block_reason": "sem_contexto_positivo_confiavel",
        }
    if score_final < MIN_CONTEXT_SCORE:
        return False, f"context_score_baixo_{score_final:.1f}", {
            "context_score_final": round(score_final, 4),
            "context_loaded": True,
            "context_has_positive": has_positive,
            "context_has_negative": False,
            "context_positive_reason": ";".join(checks),
            "context_negative_reason": "",
            "context_policy": "CONTEXTUAL_REAL",
            "context_bootstrap": False,
            "context_data_status": "CONTEXTUAL_VALIDO",
            "context_block_reason": f"context_score_baixo_{score_final:.1f}",
        }
    return True, "ok", {
        "context_score_final": round(score_final, 4),
        "context_loaded": True,
        "context_has_positive": has_positive,
        "context_has_negative": False,
        "context_positive_reason": ";".join(checks),
        "context_negative_reason": "",
        "context_policy": "CONTEXTUAL_REAL",
        "context_bootstrap": False,
        "context_data_status": "CONTEXTUAL_VALIDO",
        "context_block_reason": "",
    }
