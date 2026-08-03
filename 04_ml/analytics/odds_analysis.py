"""Módulo 3 — Desempenho por faixa de odd."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import ColumnMap
from .utils import get_logger, safe_divide

logger = get_logger(__name__)

_LIMITES = [-np.inf, 1.40, 1.60, 1.80, 2.00, 2.20, np.inf]

# Rótulos alinhados exatamente aos limites reais de cada intervalo gerado
# por pd.cut (que por padrão é fechado à direita e aberto à esquerda:
# (1.40, 1.60], etc.). Antes, rótulos como "1.41-1.60" davam a falsa
# impressão de que o limite inferior era exclusivo em 1.41 especificamente
# — o limite matemático real é "maior que 1.40", não "maior ou igual a 1.41".
_ROTULOS = ["≤1.40", ">1.40 e ≤1.60", ">1.60 e ≤1.80", ">1.80 e ≤2.00", ">2.00 e ≤2.20", ">2.20"]


def analisar_odds(resolvidas: pd.DataFrame, colunas: ColumnMap) -> pd.DataFrame:
    """Segmenta as apostas resolvidas nas faixas de odd fixas do projeto
    e calcula desempenho (quantidade, win rate, ROI, lucro) por faixa.

    Retorna um DataFrame vazio (com aviso no log) se não houver coluna
    de odd disponível no histórico.
    """
    if not colunas.odd:
        logger.warning("Coluna de odd não encontrada — análise de odds ignorada.")
        return pd.DataFrame()

    base = resolvidas.copy()
    base["Faixa de Odd"] = pd.cut(
        base[colunas.odd], bins=_LIMITES, labels=_ROTULOS, include_lowest=True
    )

    agrupado = base.groupby("Faixa de Odd", observed=False)

    saida = pd.DataFrame({
        "Faixa de Odd": _ROTULOS,
        "Quantidade": agrupado.size().reindex(_ROTULOS).values,
        "Vitórias": agrupado["_vitoria"].sum().reindex(_ROTULOS).values,
        "Derrotas": agrupado["_derrota"].sum().reindex(_ROTULOS).values,
        "Stake Total": agrupado[colunas.stake].sum().reindex(_ROTULOS).values,
        "Lucro": agrupado[colunas.lucro].sum().reindex(_ROTULOS).values,
        "Odd Média": agrupado[colunas.odd].mean().reindex(_ROTULOS).values,
    })

    decididas = saida["Vitórias"] + saida["Derrotas"]
    saida["Win Rate"] = safe_divide(saida["Vitórias"], decididas) * 100
    saida["ROI"] = safe_divide(saida["Lucro"], saida["Stake Total"]) * 100

    return saida
