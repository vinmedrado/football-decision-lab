"""Módulo 8 — Calibração do modelo (Brier Score, Log Loss, ECE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .loader import ColumnMap
from .utils import get_logger

logger = get_logger(__name__)

_EPSILON = 1e-15


@dataclass(frozen=True)
class ResumoCalibracao:
    brier_score: float
    log_loss: float
    ece: float
    apostas_avaliadas: int


def _brier_score(prob: pd.Series, alvo: pd.Series) -> float:
    return float(np.mean((prob - alvo) ** 2))


def _log_loss(prob: pd.Series, alvo: pd.Series) -> float:
    prob_clip = prob.clip(_EPSILON, 1 - _EPSILON)
    return float(-np.mean(alvo * np.log(prob_clip) + (1 - alvo) * np.log(1 - prob_clip)))


def calcular_calibracao(
    resolvidas: pd.DataFrame, colunas: ColumnMap
) -> Optional[tuple[pd.DataFrame, ResumoCalibracao]]:
    """Compara a probabilidade prevista pelo modelo com a taxa real de
    acerto, em buckets de 5%, e calcula as métricas agregadas de
    calibração (Brier Score, Log Loss e ECE).

    Considera apenas apostas decididas (vitória ou derrota) com
    probabilidade prevista válida (não-nula) — pushes são excluídos por
    não terem um alvo binário claro, e linhas sem probabilidade válida
    são excluídas explicitamente ANTES de qualquer cálculo, de modo que
    ``apostas_avaliadas`` reflita exatamente o universo usado tanto nos
    buckets da curva de calibração quanto no Brier Score/Log Loss/ECE —
    sem descompasso entre o denominador do ECE e a soma de "Quantidade"
    nos buckets. Retorna ``None`` se não houver coluna de probabilidade
    prevista ou nenhuma linha com dado válido suficiente.
    """
    if not colunas.probabilidade:
        logger.warning("Probabilidade prevista ausente — análise de calibração ignorada.")
        return None

    candidatas = resolvidas[resolvidas["_vitoria"] | resolvidas["_derrota"]].copy()
    n_antes_filtro_prob = len(candidatas)
    base = candidatas[candidatas[colunas.probabilidade].notna()].copy()
    n_descartadas_prob_nula = n_antes_filtro_prob - len(base)
    if n_descartadas_prob_nula:
        logger.warning(
            "%d aposta(s) decidida(s) sem probabilidade prevista válida foram excluídas "
            "da calibração (Brier/Log Loss/ECE calculados só sobre linhas com dado válido).",
            n_descartadas_prob_nula,
        )

    if base.empty:
        logger.warning("Nenhuma aposta decidida com probabilidade válida para calibração.")
        return None

    base["_alvo"] = base["_vitoria"].astype(float)
    prob = base[colunas.probabilidade].clip(0, 1)

    faixas = np.arange(0, 1.01, 0.05)
    base["_bucket"] = pd.cut(prob, bins=faixas, include_lowest=True)

    curva = (
        base.groupby("_bucket", observed=True)
        .agg(
            Quantidade=("_alvo", "size"),
            Probabilidade_Prevista_Media=(colunas.probabilidade, "mean"),
            Taxa_Real=("_alvo", "mean"),
        )
        .reset_index()
        .rename(columns={"_bucket": "Faixa de Probabilidade"})
    )
    curva["Faixa de Probabilidade"] = curva["Faixa de Probabilidade"].astype(str)
    curva["Probabilidade Prevista (%)"] = curva["Probabilidade_Prevista_Media"] * 100
    curva["Taxa Real (%)"] = curva["Taxa_Real"] * 100
    curva["Erro (p.p.)"] = curva["Taxa Real (%)"] - curva["Probabilidade Prevista (%)"]
    curva = curva.drop(columns=["Probabilidade_Prevista_Media", "Taxa_Real"])

    # `total` é a soma efetiva das linhas que caíram em algum bucket — não
    # `len(base)` — garantindo que os pesos (Quantidade/total) do ECE
    # somem exatamente 1 mesmo se alguma linha ficar fora de todos os
    # buckets por arredondamento de ponto flutuante nas bordas do cut.
    total_em_buckets = int(curva["Quantidade"].sum())
    apostas_avaliadas = len(base)
    if total_em_buckets != apostas_avaliadas:
        logger.warning(
            "%d de %d apostas avaliadas não caíram em nenhum bucket de calibração "
            "(possível efeito de borda no arredondamento) — ECE ponderado apenas pelas %d incluídas.",
            apostas_avaliadas - total_em_buckets, apostas_avaliadas, total_em_buckets,
        )

    ece = float(
        (curva["Quantidade"] / total_em_buckets * curva["Erro (p.p.)"].abs() / 100).sum()
    ) if total_em_buckets else 0.0

    resumo = ResumoCalibracao(
        brier_score=round(_brier_score(prob, base["_alvo"]), 6),
        log_loss=round(_log_loss(prob, base["_alvo"]), 6),
        ece=round(ece, 6),
        apostas_avaliadas=apostas_avaliadas,
    )

    return curva, resumo
