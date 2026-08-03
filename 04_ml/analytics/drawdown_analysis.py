"""Módulo 4 — Curva da banca e drawdown."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .loader import ColumnMap
from .utils import get_logger, round_or_none

logger = get_logger(__name__)

BANCA_INICIAL_PADRAO = 300.0

_COLUNA_DESEMPATE = "_ordem_original"


def _ordenar_cronologicamente(df: pd.DataFrame, colunas: ColumnMap) -> pd.DataFrame:
    """Ordena por data usando ``_ordem_original`` (a posição da aposta no
    CSV de origem) como critério de desempate determinístico.

    Isso garante que apostas do mesmo dia — sem informação de horário
    no histórico — preservem sempre a mesma sequência relativa em que
    apareceram no arquivo, independentemente de quantas vezes o
    DataFrame for reordenado ou de qual algoritmo de sort o pandas usar
    internamente. Não inventa nenhum horário: apenas evita que o
    "empate" entre datas idênticas seja resolvido de forma arbitrária.
    """
    chaves = [colunas.data]
    if _COLUNA_DESEMPATE in df.columns:
        chaves.append(_COLUNA_DESEMPATE)
    else:
        logger.warning(
            "Coluna de desempate '%s' ausente — ordenação de apostas no mesmo dia "
            "pode não ser determinística.", _COLUNA_DESEMPATE,
        )
    return df.sort_values(chaves, kind="mergesort").reset_index(drop=True)


def calcular_curva_banca(
    resolvidas: pd.DataFrame,
    colunas: ColumnMap,
    banca_inicial: float = BANCA_INICIAL_PADRAO,
) -> pd.DataFrame:
    """Reconstrói a curva da banca em ordem cronológica determinística e
    calcula, para cada aposta resolvida: novo pico, drawdown (valor e
    %) e dias em drawdown desde o último pico.

    Se existir coluna de banca real no histórico (``colunas.banca``)
    ela é usada como fonte da verdade; caso contrário a curva é
    reconstruída como ``banca_inicial + lucro acumulado``.

    Quando o pico acumulado (``_pico``) é menor ou igual a zero — banca
    zerada ou negativa desde o início —, o drawdown percentual não tem
    definição matemática estável (divisão por base não-positiva) e é
    registrado como ausente (``None`` na exportação). Esse cenário
    NÃO é escondido silenciosamente: a coluna booleana "Banca Não
    Positiva" sinaliza exatamente essas linhas e um aviso é emitido no
    log, por se tratar do cenário de risco mais crítico de uma
    auditoria de banca (ruína).
    """
    ordenado = _ordenar_cronologicamente(resolvidas, colunas).copy()

    if colunas.banca and ordenado[colunas.banca].notna().any():
        ordenado["Curva da Banca"] = ordenado[colunas.banca]
    else:
        ordenado["Curva da Banca"] = banca_inicial + ordenado[colunas.lucro].cumsum()

    # A banca inicial existe antes da primeira aposta e precisa participar do
    # cálculo do pico. Sem isso, uma derrota na primeira aposta era tratada
    # incorretamente como "novo pico" e o drawdown inicial desaparecia.
    pico_anterior = (
        ordenado["Curva da Banca"].cummax().shift(1).fillna(banca_inicial).clip(lower=banca_inicial)
    )

    ordenado["Novo Pico"] = ordenado["Curva da Banca"] > pico_anterior
    ordenado["_pico"] = ordenado["Curva da Banca"].cummax().clip(lower=banca_inicial)

    ordenado["Drawdown"] = ordenado["Curva da Banca"] - ordenado["_pico"]

    pico_positivo = ordenado["_pico"] > 0
    ordenado["Drawdown %"] = np.where(
        pico_positivo, ordenado["Drawdown"] / ordenado["_pico"] * 100, np.nan
    )
    ordenado["Banca Não Positiva"] = ~pico_positivo

    n_risco = int((~pico_positivo).sum())
    if n_risco:
        logger.warning(
            "%d linha(s) com pico de banca <= 0 detectada(s) — drawdown percentual "
            "indefinido nessas linhas (ver coluna 'Banca Não Positiva'). "
            "Risco crítico de ruína da banca.", n_risco,
        )

    ordenado["_data_ultimo_pico"] = (
        ordenado[colunas.data].where(ordenado["Novo Pico"]).ffill()
    )
    if not ordenado.empty:
        ordenado["_data_ultimo_pico"] = ordenado["_data_ultimo_pico"].fillna(
            ordenado[colunas.data].iloc[0]
        )
    ordenado["Tempo de Recuperação (dias)"] = (
        ordenado[colunas.data] - ordenado["_data_ultimo_pico"]
    ).dt.days

    return ordenado


def resumo_maior_drawdown(curva_banca: pd.DataFrame, colunas: ColumnMap) -> dict:
    """Extrai o pior episódio de drawdown da curva da banca: valor,
    percentual, datas de pico/vale e tempo de recuperação (se já
    recuperado).
    """
    if curva_banca.empty or curva_banca["Drawdown"].isna().all():
        return {
            "maior_drawdown_valor": None,
            "maior_drawdown_percentual": None,
            "data_pico": None,
            "data_vale": None,
            "data_recuperacao": None,
            "tempo_recuperacao_dias": None,
            "banca_nao_positiva_em_algum_ponto": bool(curva_banca.get("Banca Não Positiva", pd.Series(dtype=bool)).any()),
        }

    idx_min = curva_banca["Drawdown"].idxmin()
    vale = curva_banca.loc[idx_min]
    pico_valor = vale["_pico"]
    pico_data = vale["_data_ultimo_pico"]
    vale_data = vale[colunas.data]

    posteriores = curva_banca.loc[idx_min:]
    recuperados = posteriores[posteriores["Curva da Banca"] >= pico_valor]
    data_recuperacao: Optional[pd.Timestamp] = (
        recuperados[colunas.data].iloc[0] if not recuperados.empty else None
    )

    tempo_recuperacao = None
    if data_recuperacao is not None and pd.notna(pico_data):
        tempo_recuperacao = (data_recuperacao - pico_data).days

    return {
        "maior_drawdown_valor": round_or_none(vale["Drawdown"]),
        "maior_drawdown_percentual": round_or_none(vale["Drawdown %"], 4),
        "data_pico": str(pico_data.date()) if pd.notna(pico_data) else None,
        "data_vale": str(vale_data.date()) if pd.notna(vale_data) else None,
        "data_recuperacao": str(data_recuperacao.date()) if data_recuperacao is not None else None,
        "tempo_recuperacao_dias": tempo_recuperacao,
        "banca_nao_positiva_em_algum_ponto": bool(curva_banca["Banca Não Positiva"].any()),
    }
