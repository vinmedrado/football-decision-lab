"""Helpers de exportação de resultados (CSV/JSON/TXT)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def _sanitizar_valor(valor: Any) -> Any:
    """Converte um valor escalar problemático para JSON (NaN, Infinity,
    -Infinity, NaT, pd.NA, numpy scalars) em algo serializável — via
    ``None`` para os "sem valor" e tipos nativos do Python para os
    demais.
    """
    if valor is None:
        return None
    if isinstance(valor, float):
        return None if (math.isnan(valor) or math.isinf(valor)) else valor
    if isinstance(valor, np.floating):
        v = float(valor)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if valor is pd.NaT:
        return None
    if isinstance(valor, pd.Timestamp):
        return valor.isoformat()
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return valor


def sanitizar_para_json(objeto: Any) -> Any:
    """Percorre recursivamente dicts/listas substituindo NaN/Infinity/
    -Infinity/NaT/pd.NA por ``None``, garantindo que o resultado seja
    JSON padrão válido (compatível com ``JSON.parse``, Power BI, etc.),
    mesmo quando os DataFrames de origem contêm células vazias/NaN.
    """
    if isinstance(objeto, dict):
        return {str(chave): sanitizar_para_json(valor) for chave, valor in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [sanitizar_para_json(valor) for valor in objeto]
    return _sanitizar_valor(objeto)


def exportar_csv(df: Optional[pd.DataFrame], caminho: Path) -> None:
    """Exporta um DataFrame para CSV em UTF-8 com BOM (compatível com
    Excel em português). Não faz nada (apenas loga um aviso) se o
    DataFrame for ``None`` ou vazio.
    """
    if df is None or df.empty:
        logger.warning("Nada para exportar em %s (dados indisponíveis).", caminho.name)
        return
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    logger.info("Gerado: %s (%d linhas)", caminho.name, len(df))


def exportar_json(objeto: dict, caminho: Path) -> None:
    """Exporta um dicionário para JSON estritamente padrão (RFC 8259):
    NaN/Infinity/-Infinity são convertidos para ``null`` antes da
    serialização (``sanitizar_para_json``), e ``allow_nan=False`` é
    usado como rede de segurança — se algum valor não-serializável
    escapar da sanitização, a exportação falha ruidosamente em vez de
    gravar um JSON inválido silenciosamente.
    """
    objeto_limpo = sanitizar_para_json(objeto)
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(objeto_limpo, arquivo, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    logger.info("Gerado: %s", caminho.name)


def exportar_texto(linhas: list[str], caminho: Path) -> None:
    """Exporta uma lista de linhas como arquivo de texto simples."""
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Gerado: %s", caminho.name)
