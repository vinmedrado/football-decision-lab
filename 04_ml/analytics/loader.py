"""
Localização e carga do histórico de apostas.

Responsabilidades:
    1. Encontrar ``historico_apostas.csv`` automaticamente (sem caminho fixo).
    2. Mapear os nomes de coluna reais do CSV para os nomes canônicos que o
       resto do pacote usa (``data``, ``mercado``, ``odd``, ``stake``, ...),
       aceitando variações de grafia/idioma/caixa.
    3. Normalizar tipos (datas, números BR, flags de resultado) em um único
       DataFrame pronto para análise, validando a qualidade de cada conversão.
    4. Registrar a ordem original das linhas no arquivo (``_ordem_original``),
       usada por todos os módulos como critério de desempate em ordenações
       cronológicas — apostas do mesmo dia preservam a sequência em que
       apareceram no CSV, em vez de dependerem da (in)estabilidade do
       algoritmo de ordenação do pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import (
    get_logger,
    to_datetime_safe,
    to_numeric_brl,
    to_percentage_fraction,
    validar_conversao,
)

logger = get_logger(__name__)

NOME_ARQUIVO_HISTORICO = "historico_apostas.csv"

# Pastas preferenciais de busca, relativas a 04_ml/ (BASE_DIR).
_SUBPASTAS_PREFERIDAS = ("", "banca", "dados")

# Limite de perda (%) tolerado em colunas obrigatórias antes de abortar a auditoria.
LIMITE_PERDA_CRITICA_PCT = 20.0

_RESULTADOS_PENDENTES = {
    "pendente", "pending", "aberta", "open", "em aberto",
    "não resolvida", "nao resolvida", "sem resultado", "aguardando",
    "nan", "none", "",
}
_RESULTADOS_VITORIA = {
    "ganhou", "green", "win", "won", "vitoria", "vitória", "ganho", "gain", "g",
}
_RESULTADOS_DERROTA = {
    "perdeu", "red", "loss", "lost", "derrota", "perda", "l", "r",
}
_RESULTADOS_PUSH = {
    "push", "anulada", "anulado", "void", "reembolso", "cancelada", "cancelado",
}


@dataclass(frozen=True)
class ColumnMap:
    """Mapeamento de colunas canônicas -> nomes reais encontrados no CSV.

    Campos com valor ``None`` indicam que a coluna não existe no arquivo
    (aceitável para colunas opcionais como ``probabilidade``, ``liga`` e
    ``banca``, que ativam módulos condicionais).
    """

    data: str
    mercado: str
    resultado: str
    stake: str
    lucro: str
    odd: Optional[str] = None
    probabilidade: Optional[str] = None
    banca: Optional[str] = None
    liga: Optional[str] = None
    origem: Optional[str] = None

    def disponiveis(self) -> dict:
        """Retorna apenas os campos que foram efetivamente encontrados."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name)}


# Aliases aceitos por coluna canônica. Comparação é sempre case-insensitive
# e ignora espaços nas pontas, então "Market", "MERCADO", " mercado " etc.
# resolvem para o mesmo alias.
#
# IMPORTANTE: os aliases de "stake" foram deliberadamente restritos a nomes
# que se referem inequivocamente ao valor apostado. Termos genéricos como
# "valor" ou "unidade" foram removidos por risco de mapeamento falso em
# históricos de outros formatos (ex.: uma coluna "valor_liquido" não
# relacionada a stake poderia ser capturada incorretamente).
_ALIASES = {
    "data": ["data", "date", "data_aposta", "dt", "match_date"],
    "mercado": ["mercado", "market", "mercados", "markets", "tipo_aposta", "bet_market"],
    "resultado": ["resultado", "status", "result", "result_status"],
    "stake": [
        "stake", "valor_apostado", "valor aposta", "aposta", "bet_amount", "stake_valor",
    ],
    "lucro": ["lucro", "profit", "pnl", "resultado_financeiro", "ganho_perda", "lucro_liquido"],
    "odd": ["odd", "odds", "cotacao", "cotação", "price"],
    "probabilidade": [
        "probabilidade", "probability", "prob_modelo", "prob", "predicted_probability",
    ],
    "banca": ["banca", "banca_apos", "bankroll", "bankroll_after", "saldo"],
    "liga": ["liga", "league", "competicao", "competição", "campeonato"],
    "origem": ["origem", "origin", "source", "data_source"],
}

_OBRIGATORIAS = ("data", "mercado", "resultado", "stake", "lucro")


def localizar_historico(base_dir: Path) -> Path:
    """Procura ``historico_apostas.csv`` em locais conhecidos e, na falta
    deles, em qualquer subdiretório de ``base_dir`` (recursivamente).

    Levanta ``FileNotFoundError`` com mensagem clara apenas se realmente
    não encontrar o arquivo em lugar nenhum.
    """
    candidatos = [base_dir / sub / NOME_ARQUIVO_HISTORICO for sub in _SUBPASTAS_PREFERIDAS]
    for caminho in candidatos:
        if caminho.exists():
            logger.info("Histórico localizado em %s", caminho)
            return caminho

    encontrados = sorted(base_dir.rglob(NOME_ARQUIVO_HISTORICO))
    if encontrados:
        logger.info("Histórico localizado (busca recursiva) em %s", encontrados[0])
        return encontrados[0]

    raise FileNotFoundError(
        f"Não foi possível localizar '{NOME_ARQUIVO_HISTORICO}' em {base_dir} "
        f"nem em suas subpastas ({', '.join(_SUBPASTAS_PREFERIDAS)} ou outras)."
    )


def _resolver_coluna(colunas_reais: dict, aliases: list) -> Optional[str]:
    for alias in aliases:
        achado = colunas_reais.get(alias.strip().lower())
        if achado is not None:
            return achado
    return None


def mapear_colunas(df: pd.DataFrame) -> ColumnMap:
    """Identifica automaticamente as colunas relevantes do DataFrame,
    tolerando variações de nome/idioma/caixa (``market``/``Mercado``/etc.).
    """
    colunas_reais = {str(c).strip().lower(): c for c in df.columns}

    resolvidas = {
        canonico: _resolver_coluna(colunas_reais, aliases)
        for canonico, aliases in _ALIASES.items()
    }

    faltando = [c for c in _OBRIGATORIAS if not resolvidas.get(c)]
    if faltando:
        raise KeyError(
            "Colunas obrigatórias não encontradas no histórico: "
            f"{faltando}. Colunas disponíveis: {list(df.columns)}"
        )

    return ColumnMap(**resolvidas)


def _classificar_resultado(serie_resultado: pd.Series) -> pd.DataFrame:
    normalizado = serie_resultado.astype(str).str.strip().str.lower()
    vitoria = normalizado.isin(_RESULTADOS_VITORIA)
    derrota = normalizado.isin(_RESULTADOS_DERROTA)
    push = normalizado.isin(_RESULTADOS_PUSH)
    pendente = normalizado.isin(_RESULTADOS_PENDENTES)
    return pd.DataFrame({
        "_resolvida": vitoria | derrota | push,
        "_vitoria": vitoria,
        "_derrota": derrota,
        "_push": push,
        "_status_desconhecido": ~(vitoria | derrota | push | pendente),
    })


def carregar_dados(base_dir: Path) -> tuple[pd.DataFrame, ColumnMap, Path]:
    """Ponto de entrada do loader: localiza o CSV, mapeia colunas e devolve
    um DataFrame já normalizado (datas, números, flags de resultado).

    Cada conversão numérica/data passa por ``validar_conversao``: colunas
    obrigatórias (data/stake/lucro) abortam a auditoria se ficarem
    inteiramente inválidas ou perderem uma fração grande de valores
    válidos; colunas opcionais (odd/probabilidade/banca) apenas geram
    aviso no log.

    Retorna ``(dados, colunas, caminho_arquivo)``.
    """
    caminho = localizar_historico(base_dir)
    df = pd.read_csv(caminho, low_memory=False, encoding="utf-8-sig")
    logger.info("Registros carregados: %d", len(df))

    colunas = mapear_colunas(df)
    dados = df.copy()

    # Ordem original do arquivo — usada como critério de desempate
    # determinístico em toda ordenação cronológica feita pelos demais
    # módulos (evita que apostas do mesmo dia troquem de posição entre
    # execuções ou entre módulos diferentes).
    dados["_ordem_original"] = np.arange(len(dados))

    dados[colunas.data] = to_datetime_safe(dados[colunas.data])
    validar_conversao(df[colunas.data], dados[colunas.data], "data", LIMITE_PERDA_CRITICA_PCT, obrigatoria=True)

    dados[colunas.stake] = to_numeric_brl(dados[colunas.stake])
    validar_conversao(df[colunas.stake], dados[colunas.stake], "stake", LIMITE_PERDA_CRITICA_PCT, obrigatoria=True)

    dados[colunas.lucro] = to_numeric_brl(dados[colunas.lucro])
    validar_conversao(df[colunas.lucro], dados[colunas.lucro], "lucro", LIMITE_PERDA_CRITICA_PCT, obrigatoria=True)

    if colunas.odd:
        dados[colunas.odd] = to_numeric_brl(dados[colunas.odd])
        validar_conversao(df[colunas.odd], dados[colunas.odd], "odd", obrigatoria=False)
    if colunas.probabilidade:
        dados[colunas.probabilidade] = to_percentage_fraction(dados[colunas.probabilidade], "probabilidade")
        validar_conversao(df[colunas.probabilidade], dados[colunas.probabilidade], "probabilidade", obrigatoria=False)
    if colunas.banca:
        dados[colunas.banca] = to_numeric_brl(dados[colunas.banca])
        validar_conversao(df[colunas.banca], dados[colunas.banca], "banca", obrigatoria=False)

    classificacao = _classificar_resultado(dados[colunas.resultado])
    desconhecidos = classificacao["_status_desconhecido"]
    if desconhecidos.any():
        exemplos = sorted(
            dados.loc[desconhecidos, colunas.resultado].astype(str).str.strip().unique().tolist()
        )[:10]
        raise ValueError(
            f"{int(desconhecidos.sum())} resultado(s) com status desconhecido: {exemplos}. "
            "A auditoria foi abortada para não misturar registros inválidos com apostas liquidadas."
        )
    dados = pd.concat([dados, classificacao], axis=1)

    return dados, colunas, caminho
