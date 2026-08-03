"""Módulo 5 — Sequências (streaks) de vitórias e derrotas.

Pushes (resultado neutro, sem vitória nem derrota) são ignorados no
cálculo de sequências: uma aposta push não conta como vitória, não
conta como derrota, e — o ponto que motivou a correção deste módulo —
não interrompe uma sequência em andamento. Ela é tratada como se não
tivesse acontecido para fins de streak, preservando o comportamento
esperado por quem acompanha sequências de green/red na prática.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import ColumnMap

_COLUNA_DESEMPATE = "_ordem_original"


@dataclass(frozen=True)
class ResumoStreaks:
    maior_sequencia_vitorias: int
    maior_sequencia_derrotas: int
    sequencia_media_vitorias: float
    sequencia_media_derrotas: float


def _extrair_sequencias(flags: pd.Series) -> list[int]:
    """Dado um vetor booleano, retorna o tamanho de cada sequência
    consecutiva de ``True``."""
    sequencias: list[int] = []
    atual = 0
    for valor in flags:
        if valor:
            atual += 1
        elif atual:
            sequencias.append(atual)
            atual = 0
    if atual:
        sequencias.append(atual)
    return sequencias


def calcular_streaks(resolvidas: pd.DataFrame, colunas: ColumnMap) -> tuple[ResumoStreaks, pd.DataFrame]:
    """Calcula sequências de greens/reds em ordem cronológica
    determinística (mesmo critério de desempate usado no resto do
    pacote), ignorando apostas push por completo — elas não entram na
    sequência de vitórias nem na de derrotas, e não a interrompem.

    Retorna o resumo (maiores/médias sequências) e uma tabela de
    distribuição de tamanhos de sequência por tipo (vitória/derrota).
    """
    chaves = [colunas.data]
    if _COLUNA_DESEMPATE in resolvidas.columns:
        chaves.append(_COLUNA_DESEMPATE)
    ordenado = resolvidas.sort_values(chaves, kind="mergesort")

    decididas = ordenado[~ordenado["_push"]]

    seq_vitorias = _extrair_sequencias(decididas["_vitoria"].tolist())
    seq_derrotas = _extrair_sequencias(decididas["_derrota"].tolist())

    resumo = ResumoStreaks(
        maior_sequencia_vitorias=max(seq_vitorias, default=0),
        maior_sequencia_derrotas=max(seq_derrotas, default=0),
        sequencia_media_vitorias=round(float(np.mean(seq_vitorias)), 2) if seq_vitorias else 0.0,
        sequencia_media_derrotas=round(float(np.mean(seq_derrotas)), 2) if seq_derrotas else 0.0,
    )

    distribuicao = pd.concat([
        pd.DataFrame({"Tipo": "Vitórias", "Tamanho da Sequência": seq_vitorias}),
        pd.DataFrame({"Tipo": "Derrotas", "Tamanho da Sequência": seq_derrotas}),
    ], ignore_index=True)

    if not distribuicao.empty:
        distribuicao = (
            distribuicao.groupby(["Tipo", "Tamanho da Sequência"], observed=True)
            .size()
            .reset_index(name="Ocorrências")
            .sort_values(["Tipo", "Tamanho da Sequência"])
            .reset_index(drop=True)
        )

    return resumo, distribuicao
