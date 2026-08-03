"""Módulo 1 (e Módulo 9) — Desempenho por mercado e por liga.

As duas análises têm exatamente a mesma forma (agrupar apostas
resolvidas por uma coluna categórica e medir volume/ROI/lucro), então
compartilham a mesma rotina genérica ``_analisar_por_categoria``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .loader import ColumnMap
from .utils import safe_divide

# Quantidade mínima de apostas que uma categoria (mercado/liga/mês) precisa
# ter para ser elegível a "campeã"/"pior"/entrar no top N. Categorias
# abaixo do limite continuam em TODAS as tabelas/CSVs completos — apenas
# ficam de fora da eleição de destaques, para não distorcer rankings com
# amostras estatisticamente pequenas demais.
AMOSTRA_MINIMA_RANKING_PADRAO = 30


def _analisar_por_categoria(
    resolvidas: pd.DataFrame,
    coluna_categoria: str,
    nome_coluna_saida: str,
    colunas: ColumnMap,
    incluir_ranking: bool = True,
    amostra_minima: int = AMOSTRA_MINIMA_RANKING_PADRAO,
) -> pd.DataFrame:
    # dropna=False preserva categorias nulas (ex.: liga não informada em
    # algumas linhas) como um grupo próprio "NaN" em vez de descartar
    # silenciosamente essas apostas da tabela.
    agrupado = resolvidas.groupby(coluna_categoria, dropna=False, observed=True)

    saida = pd.DataFrame({
        nome_coluna_saida: agrupado.size().index,
        "Quantidade": agrupado.size().values,
        "Vitórias": agrupado["_vitoria"].sum().values,
        "Derrotas": agrupado["_derrota"].sum().values,
        "Push": agrupado["_push"].sum().values,
        "Stake Total": agrupado[colunas.stake].sum().values,
        "Lucro": agrupado[colunas.lucro].sum().values,
    })

    decididas = saida["Vitórias"] + saida["Derrotas"]
    saida["Win Rate"] = safe_divide(saida["Vitórias"], decididas) * 100
    saida["ROI"] = safe_divide(saida["Lucro"], saida["Stake Total"]) * 100

    if colunas.odd:
        saida["Odd Média"] = agrupado[colunas.odd].mean().values

    if colunas.probabilidade and colunas.odd:
        # Reindexado explicitamente por categoria (em vez de assumir que a
        # ordem de saída de um segundo .groupby().mean() bate
        # posicionalmente com `saida`), o que evita divergência de
        # tamanho/alinhamento quando a coluna de categoria tem valores
        # nulos (grupo NaN tratado de forma assimétrica entre groupbys).
        ev_unitario = resolvidas[colunas.probabilidade] * resolvidas[colunas.odd] - 1
        ev_por_categoria = (
            ev_unitario.groupby(resolvidas[coluna_categoria], dropna=False, observed=True).mean()
        )
        saida["EV Médio"] = saida[nome_coluna_saida].map(ev_por_categoria) * 100

    if incluir_ranking:
        elegivel = saida["Quantidade"] >= amostra_minima
        saida["Ranking por ROI"] = pd.NA
        saida["Ranking por Lucro"] = pd.NA
        saida.loc[elegivel, "Ranking por ROI"] = (
            saida.loc[elegivel, "ROI"].rank(ascending=False, method="min")
        )
        saida.loc[elegivel, "Ranking por Lucro"] = (
            saida.loc[elegivel, "Lucro"].rank(ascending=False, method="min")
        )
        saida["Ranking por ROI"] = saida["Ranking por ROI"].astype("Int64")
        saida["Ranking por Lucro"] = saida["Ranking por Lucro"].astype("Int64")
        # Elegíveis primeiro (ordenados por ROI), não-elegíveis ao final —
        # nenhuma linha é removida do DataFrame, apenas reordenada.
        saida = saida.sort_values(
            ["Ranking por ROI"], na_position="last"
        )

    return saida.reset_index(drop=True)


def analisar_mercados(
    resolvidas: pd.DataFrame, colunas: ColumnMap, amostra_minima: int = AMOSTRA_MINIMA_RANKING_PADRAO
) -> pd.DataFrame:
    """Gera o resumo de desempenho por mercado, incluindo ranking por
    ROI e por lucro.

    Colunas de saída: Mercado, Quantidade, Vitórias, Derrotas, Push,
    Win Rate, Stake Total, Lucro, ROI, Odd Média, EV Médio (se
    houver probabilidade), Ranking por ROI, Ranking por Lucro.
    Mercados com menos de ``amostra_minima`` apostas recebem ranking
    nulo (não competem por "campeão"), mas permanecem na tabela.
    """
    return _analisar_por_categoria(
        resolvidas, colunas.mercado, "Mercado", colunas, amostra_minima=amostra_minima
    )


def analisar_ligas(
    resolvidas: pd.DataFrame, colunas: ColumnMap, amostra_minima: int = AMOSTRA_MINIMA_RANKING_PADRAO
) -> Optional[pd.DataFrame]:
    """Gera o resumo de desempenho por liga (Módulo 9). Retorna
    ``None`` se o histórico não tiver coluna de liga. Mesmas regras de
    amostra mínima de ``analisar_mercados``.
    """
    if not colunas.liga:
        return None
    return _analisar_por_categoria(
        resolvidas, colunas.liga, "Liga", colunas, amostra_minima=amostra_minima
    )
