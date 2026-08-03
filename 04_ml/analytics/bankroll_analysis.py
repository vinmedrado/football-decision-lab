"""Módulo 6 — Evolução completa da banca."""

from __future__ import annotations

import pandas as pd

from .loader import ColumnMap

# Contrato esperado com drawdown_analysis.calcular_curva_banca: essas
# colunas precisam existir em `curva_banca` para que este módulo consiga
# montar a tabela de evolução da banca. Validado explicitamente antes de
# qualquer indexação, para falhar com uma mensagem clara em vez de um
# KeyError genérico caso o contrato entre os dois módulos seja quebrado
# por uma mudança futura em drawdown_analysis.
COLUNAS_ESPERADAS_CURVA_BANCA = (
    "Curva da Banca", "Novo Pico", "Drawdown", "Drawdown %",
    "Tempo de Recuperação (dias)", "Banca Não Positiva",
)


def _validar_contrato_curva_banca(curva_banca: pd.DataFrame) -> None:
    faltando = [c for c in COLUNAS_ESPERADAS_CURVA_BANCA if c not in curva_banca.columns]
    if faltando:
        raise KeyError(
            "Contrato quebrado entre drawdown_analysis e bankroll_analysis: "
            f"coluna(s) esperada(s) ausente(s) em `curva_banca`: {faltando}. "
            "Verifique se drawdown_analysis.calcular_curva_banca foi alterado "
            "sem atualizar bankroll_analysis.COLUNAS_ESPERADAS_CURVA_BANCA."
        )


def montar_evolucao_banca(curva_banca: pd.DataFrame, colunas: ColumnMap) -> pd.DataFrame:
    """Formata a evolução da banca (já calculada em ``drawdown_analysis``)
    em uma tabela enxuta para exportação, aposta a aposta em ordem
    cronológica determinística.
    """
    _validar_contrato_curva_banca(curva_banca)

    colunas_saida = {
        colunas.data: "Data",
        colunas.mercado: "Mercado",
        colunas.resultado: "Resultado",
        colunas.stake: "Stake",
        colunas.lucro: "Lucro",
    }

    tabela = curva_banca[
        list(colunas_saida.keys())
        + list(COLUNAS_ESPERADAS_CURVA_BANCA)
    ]
    tabela = tabela.rename(columns=colunas_saida)
    return tabela.reset_index(drop=True)
