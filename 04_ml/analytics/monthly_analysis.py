"""Módulo 2 — Desempenho mensal."""

from __future__ import annotations

import pandas as pd

from .loader import ColumnMap
from .market_analysis import AMOSTRA_MINIMA_RANKING_PADRAO
from .utils import safe_divide


def analisar_mensal(
    resolvidas: pd.DataFrame,
    colunas: ColumnMap,
    curva_banca: pd.DataFrame,
    amostra_minima: int = AMOSTRA_MINIMA_RANKING_PADRAO,
) -> pd.DataFrame:
    """Gera o resumo de desempenho por mês, incluindo o pior drawdown
    percentual observado dentro de cada mês e um "Ranking por ROI"
    consistente com o mesmo critério usado para mercados e ligas
    (Módulos 1 e 9) — meses com menos de ``amostra_minima`` apostas
    ficam com ranking nulo (não competem por "melhor mês"), mas
    permanecem na tabela.

    ``curva_banca`` deve ser o resultado de
    ``drawdown_analysis.calcular_curva_banca`` (mesma ordem/index de
    ``resolvidas``), usado para atribuir o drawdown do mês.
    """
    base = resolvidas.copy()
    base["_mes"] = base[colunas.data].dt.to_period("M").astype(str)

    agrupado = base.groupby("_mes", dropna=False, observed=True)

    saida = pd.DataFrame({
        "Mês": agrupado.size().index,
        "Quantidade": agrupado.size().values,
        "Lucro": agrupado[colunas.lucro].sum().values,
        "Vitórias": agrupado["_vitoria"].sum().values,
        "Derrotas": agrupado["_derrota"].sum().values,
        "Stake Total": agrupado[colunas.stake].sum().values,
    })

    decididas = saida["Vitórias"] + saida["Derrotas"]
    saida["Win Rate"] = safe_divide(saida["Vitórias"], decididas) * 100
    saida["ROI"] = safe_divide(saida["Lucro"], saida["Stake Total"]) * 100

    if colunas.odd:
        saida["Odd Média"] = agrupado[colunas.odd].mean().values

    if "_drawdown_pct" in curva_banca.columns:
        drawdown_mes = (
            curva_banca.assign(_mes=base["_mes"].values)
            .groupby("_mes", observed=True)["_drawdown_pct"]
            .min()
        )
        saida["Drawdown do Mês"] = saida["Mês"].map(drawdown_mes)

    elegivel = saida["Quantidade"] >= amostra_minima
    saida["Ranking por ROI"] = pd.NA
    saida.loc[elegivel, "Ranking por ROI"] = saida.loc[elegivel, "ROI"].rank(ascending=False, method="min")
    saida["Ranking por ROI"] = saida["Ranking por ROI"].astype("Int64")

    return saida.drop(columns=["Vitórias", "Derrotas", "Stake Total"]).reset_index(drop=True)
