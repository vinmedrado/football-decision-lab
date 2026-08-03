"""
Funções utilitárias compartilhadas pelo módulo de auditoria.

Contém apenas rotinas puras (sem I/O de arquivo) usadas pelos demais
módulos: conversão robusta de moeda/porcentagem/datas/números no
padrão brasileiro, validação pós-conversão, divisão segura e um
logger amigável padronizado.

Nenhuma função aqui lança exceção por causa de formato de dado
isoladamente — o princípio geral é: dado ruim vira NaN. A exceção só
é levantada explicitamente por ``validar_conversao`` quando a perda de
dados é grande o suficiente para comprometer a análise (ver docstring
da função).
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger amigável e silencioso por padrão (sem prints soltos)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_logger = get_logger(__name__)

_MOEDA_PATTERN = re.compile(r"[Rr]\$|\s")
_GRUPO_MILHAR_BR = re.compile(r"^\d{1,3}(\.\d{3})+$")


def _normalizar_numero_str(texto: str) -> Optional[str]:
    """
    Normaliza uma única string numérica para um formato que
    ``float()``/``pandas.to_numeric`` interpreta corretamente.

    Regras (nesta ordem):

    1. Contém ``,`` e ``.``: o separador decimal é o que aparece por
       último na string; o outro é separador de milhar e é removido.
       "1.234,56" -> "1234.56" | "1,234.56" -> "1234.56"
    2. Contém somente ``,``: vírgula é separador decimal (padrão BR).
       "1234,56" -> "1234.56"
    3. Contém somente ``.``: ambíguo entre decimal e milhar BR.
       Resolvido pela contagem de dígitos após o ÚLTIMO ponto — se o
       último grupo tiver exatamente 3 dígitos E a string inteira
       seguir o padrão de agrupamento de milhar BR (grupos de 3
       dígitos), o ponto é separador de milhar e é removido:
       "1.234" -> "1234" | "10.000" -> "10000" | "1.234.567" -> "1234567"
       Caso contrário, o ponto é decimal e a string é mantida:
       "1234.56" -> "1234.56" | "1234.5" -> "1234.5"
    4. Sem separador nenhum: devolvida sem alteração.
    """
    if texto is None:
        return None
    texto = texto.strip()
    if not texto:
        return None

    tem_virgula = "," in texto
    tem_ponto = "." in texto

    if tem_virgula and tem_ponto:
        if texto.rfind(",") > texto.rfind("."):
            # vírgula é a decimal; ponto(s) são milhar
            texto = texto.replace(".", "").replace(",", ".")
        else:
            # ponto é a decimal; vírgula(s) são milhar
            texto = texto.replace(",", "")
        return texto

    if tem_virgula:
        return texto.replace(".", "").replace(",", ".")

    if tem_ponto:
        if _GRUPO_MILHAR_BR.match(texto):
            return texto.replace(".", "")
        return texto

    return texto


def to_numeric_brl(series: pd.Series) -> pd.Series:
    """
    Converte uma série de texto para número, aceitando formatos mistos:

        "R$ 1.234,56"  -> 1234.56
        "1.234,56"     -> 1234.56
        "1234,56"      -> 1234.56
        "1234.56"      -> 1234.56
        "1.234"        -> 1234.0   (milhar BR, sem casas decimais)
        "2.500"        -> 2500.0
        "10.000"       -> 10000.0
        1234.56 (já numérico) -> 1234.56

    Nunca lança erro: valores não conversíveis viram NaN
    (via ``pandas.to_numeric(errors="coerce")``).
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    texto = series.astype(str).str.strip()
    texto = texto.str.replace(_MOEDA_PATTERN, "", regex=True)
    texto = texto.str.replace("%", "", regex=False)
    texto = texto.replace({"": np.nan, "nan": np.nan, "none": np.nan, "None": np.nan})

    normalizado = texto.map(lambda v: _normalizar_numero_str(v) if isinstance(v, str) else v)
    return pd.to_numeric(normalizado, errors="coerce")


def to_percentage_fraction(series: pd.Series, nome_coluna: str = "probabilidade") -> pd.Series:
    """
    Converte uma série de probabilidade/porcentagem para fração 0-1,
    linha a linha (nunca decide a escala da coluna inteira de uma vez):

    - Cada valor > 1 é assumido em escala 0-100 e dividido por 100.
    - Cada valor <= 1 é assumido já em fração 0-1 e mantido como está.

    Isso permite que colunas com escalas misturadas (ex.: a maioria em
    0-1 e uma linha digitada como "55" em vez de "0,55") sejam
    corrigidas individualmente, sem que um único valor fora de escala
    corrompa as demais linhas.

    Valores que, mesmo após a normalização por linha, ficam fora do
    intervalo [0, 1] (ex.: um "150" que não é nem fração nem
    percentual válido) são considerados inconsistentes: um aviso é
    registrado no log e o valor é convertido para NaN em vez de
    propagar um número estatisticamente inválido para EV/calibração.
    """
    numerica = to_numeric_brl(series)
    if numerica.dropna().empty:
        return numerica

    fracao = numerica.astype("float64")
    maior_que_1 = fracao > 1
    fracao.loc[maior_que_1] = fracao.loc[maior_que_1] / 100.0

    invalidos = fracao.notna() & ((fracao < 0) | (fracao > 1))
    n_invalidos = int(invalidos.sum())
    if n_invalidos:
        _logger.warning(
            "%d valor(es) da coluna '%s' fora do intervalo válido [0,1] mesmo após "
            "normalização de escala — tratados como dado inconsistente (NaN).",
            n_invalidos, nome_coluna,
        )
        fracao.loc[invalidos] = np.nan

    return fracao


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Converte para datetime sem lançar erro; formatos inválidos viram NaT."""
    convertido = pd.to_datetime(series, errors="coerce", dayfirst=False)
    # fallback para datas em formato dia/mês/ano quando a maioria falhou
    if convertido.isna().mean() > 0.5:
        alternativo = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if alternativo.notna().sum() > convertido.notna().sum():
            return alternativo
    return convertido


def safe_divide(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Divisão elemento-a-elemento que retorna NaN em vez de erro/inf quando denom=0."""
    denom = denominador.replace(0, np.nan)
    return numerador / denom


def safe_divide_scalar(numerador: float, denominador: float) -> Optional[float]:
    """Versão escalar de ``safe_divide`` — retorna None em vez de ZeroDivisionError."""
    if not denominador:
        return None
    return numerador / denominador


def round_or_none(valor: Optional[float], casas: int = 2) -> Optional[float]:
    """Arredonda um valor, preservando None/NaN em vez de lançar erro."""
    if valor is None:
        return None
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(valor), casas)


@dataclass(frozen=True)
class ResultadoValidacaoConversao:
    """Resultado da checagem de qualidade de uma conversão numérica/data."""

    coluna: str
    total_linhas: int
    validos_antes: int
    validos_depois: int
    percentual_perda: float


def validar_conversao(
    bruta: pd.Series,
    convertida: pd.Series,
    nome_coluna: str,
    limite_perda_pct: float = 20.0,
    obrigatoria: bool = True,
) -> ResultadoValidacaoConversao:
    """
    Compara quantos valores válidos existiam antes e depois de uma
    conversão (numérica ou de data) e reage à perda de dados:

    - Sempre registra no log quantos valores válidos viraram NaN/NaT
      (se houver perda), em nível ERROR quando a perda passa de
      ``limite_perda_pct``, ou WARNING caso contrário.
    - Se ``obrigatoria=True`` e a coluna ficou inteiramente inválida
      (havia dado, mas nada sobrou) OU a perda passou do limite,
      levanta ``ValueError`` — evita seguir a auditoria inteira com
      uma coluna essencial silenciosamente zerada.
    - Se ``obrigatoria=False`` (colunas opcionais como odd/probabilidade),
      nunca levanta erro: apenas loga, e os módulos condicionais que
      dependem dela decidem o que fazer com dados ausentes.
    """
    total = len(bruta)
    bruta_texto = bruta.astype(str).str.strip().str.lower()
    valido_bruto = bruta.notna() & ~bruta_texto.isin(["", "nan", "none", "nat"])
    validos_antes = int(valido_bruto.sum())
    validos_depois = int(convertida.notna().sum())
    perda = max(validos_antes - validos_depois, 0)
    percentual_perda = round((perda / validos_antes * 100), 2) if validos_antes else 0.0

    if perda > 0:
        nivel = _logger.error if percentual_perda >= limite_perda_pct else _logger.warning
        nivel(
            "Coluna '%s': %d de %d valor(es) válido(s) (%.2f%%) tornaram-se inválidos (NaN/NaT) "
            "após a conversão.",
            nome_coluna, perda, validos_antes, percentual_perda,
        )

    if obrigatoria and validos_antes > 0:
        if validos_depois == 0:
            raise ValueError(
                f"Coluna obrigatória '{nome_coluna}' ficou inteiramente inválida (NaN/NaT) "
                "após a conversão — abortando para não gerar auditoria com dado zerado."
            )
        if percentual_perda >= limite_perda_pct:
            raise ValueError(
                f"Coluna obrigatória '{nome_coluna}' perdeu {percentual_perda:.2f}% dos valores "
                f"válidos na conversão (limite: {limite_perda_pct}%) — abortando."
            )

    return ResultadoValidacaoConversao(nome_coluna, total, validos_antes, validos_depois, percentual_perda)
