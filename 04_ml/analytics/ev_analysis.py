"""Módulo 7 — Expected Value (EV), condicionado a existir probabilidade prevista.

Universo considerado: TODAS as apostas resolvidas, incluindo pushes.
Isso é intencional — EV é uma medida ex-ante (probabilidade do modelo x
odd oferecida no momento da aposta), independente do resultado final
ter sido push, vitória ou derrota. Isso é diferente do universo usado
em ``calibration_analysis`` (que exclui pushes, pois calibração precisa
de um alvo binário observado). A constante ``UNIVERSO_EV`` documenta
essa escolha e é exposta junto do resultado para que quem consumir o
CSV/JSON saiba exatamente que população foi usada.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .loader import ColumnMap
from .utils import get_logger

logger = get_logger(__name__)

UNIVERSO_EV = "todas as apostas resolvidas (vitória, derrota e push)"


def calcular_ev(resolvidas: pd.DataFrame, colunas: ColumnMap) -> Optional[pd.DataFrame]:
    """Calcula EV unitário e financeiro aposta a aposta, e agrega por
    mercado e por mês.

    Retorna ``None`` (com aviso no log) se não houver coluna de
    probabilidade prevista e de odd disponíveis — nesse caso o EV não
    pode ser calculado.
    """
    if not (colunas.probabilidade and colunas.odd):
        logger.warning("Probabilidade prevista e/ou odd ausentes — análise de EV ignorada.")
        return None

    base = resolvidas.copy()
    base["_ev_unitario"] = base[colunas.probabilidade] * base[colunas.odd] - 1
    base["_ev_financeiro"] = base["_ev_unitario"] * base[colunas.stake]
    base["_mes"] = base[colunas.data].dt.to_period("M").astype(str)

    por_mercado = (
        base.groupby(colunas.mercado, dropna=False, observed=True)
        .agg(
            Quantidade=("_ev_unitario", "size"),
            EV_Medio=("_ev_unitario", "mean"),
            EV_Financeiro_Total=("_ev_financeiro", "sum"),
            Lucro_Real=(colunas.lucro, "sum"),
        )
        .reset_index()
        .rename(columns={colunas.mercado: "Mercado", "EV_Medio": "EV Médio", "EV_Financeiro_Total": "EV Financeiro Total", "Lucro_Real": "Lucro Real"})
    )
    por_mercado["Agrupamento"] = "Mercado"
    por_mercado = por_mercado.rename(columns={"Mercado": "Chave"})

    por_mes = (
        base.groupby("_mes", dropna=False, observed=True)
        .agg(
            Quantidade=("_ev_unitario", "size"),
            EV_Medio=("_ev_unitario", "mean"),
            EV_Financeiro_Total=("_ev_financeiro", "sum"),
            Lucro_Real=(colunas.lucro, "sum"),
        )
        .reset_index()
        .rename(columns={"_mes": "Chave", "EV_Medio": "EV Médio", "EV_Financeiro_Total": "EV Financeiro Total", "Lucro_Real": "Lucro Real"})
    )
    por_mes["Agrupamento"] = "Mês"

    saida = pd.concat([por_mercado, por_mes], ignore_index=True)
    saida["EV Médio"] = saida["EV Médio"] * 100  # exibido em %
    saida["Universo"] = UNIVERSO_EV
    colunas_ordem = ["Agrupamento", "Chave", "Quantidade", "EV Médio", "EV Financeiro Total", "Lucro Real", "Universo"]
    return saida[colunas_ordem]
