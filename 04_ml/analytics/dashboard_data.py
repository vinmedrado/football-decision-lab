"""Agregação final para consumo por dashboard (Streamlit / Power BI)."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def montar_dashboard_data(
    resumo_executivo: dict,
    mercados: pd.DataFrame,
    mensal: pd.DataFrame,
    odds: pd.DataFrame,
    curva_banca: pd.DataFrame,
    streaks_distribuicao: pd.DataFrame,
    ev: Optional[pd.DataFrame],
    calibracao_curva: Optional[pd.DataFrame],
    ligas: Optional[pd.DataFrame],
) -> dict:
    """Empacota todos os indicadores já calculados em um único objeto
    JSON-friendly (listas de dicts), pronto para ser consumido por um
    front-end de dashboard sem reprocessamento.
    """

    def registros(df: Optional[pd.DataFrame]) -> list:
        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    return {
        "resumo_executivo": resumo_executivo,
        "mercados": registros(mercados),
        "mensal": registros(mensal),
        "odds": registros(odds),
        "curva_banca": registros(
            curva_banca[[c for c in curva_banca.columns if not c.startswith("_")]]
            if curva_banca is not None and not curva_banca.empty
            else None
        ),
        "streaks_distribuicao": registros(streaks_distribuicao),
        "expected_value": registros(ev),
        "calibracao": registros(calibracao_curva),
        "ligas": registros(ligas),
    }
