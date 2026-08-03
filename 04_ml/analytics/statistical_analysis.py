"""Incerteza estatística das métricas financeiras observadas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import ColumnMap
from .utils import round_or_none

BOOTSTRAP_SEED = 20260723
BOOTSTRAP_REPETITIONS = 10_000


def calcular_incerteza_roi(
    resolvidas: pd.DataFrame,
    colunas: ColumnMap,
    repeticoes: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Estima IC95% do ROI por bootstrap agrupado por dia.

    O dia, e não a aposta individual, é reamostrado porque apostas do mesmo dia
    podem compartilhar ligas, jogos, condições de mercado e versões do modelo.
    A semente fixa deixa o relatório integralmente reproduzível.
    """
    base = resolvidas[[colunas.data, colunas.lucro, colunas.stake]].dropna().copy()
    dias = (
        base.groupby(colunas.data, observed=True)
        .agg(lucro=(colunas.lucro, "sum"), stake=(colunas.stake, "sum"))
        .reset_index(drop=True)
    )
    dias = dias[dias["stake"] > 0]
    if dias.empty:
        return {
            "metodo": "bootstrap_por_dia",
            "dias_avaliados": 0,
            "repeticoes": repeticoes,
            "roi_ic95_inferior_percentual": None,
            "roi_ic95_superior_percentual": None,
            "probabilidade_roi_nao_positivo_percentual": None,
        }

    valores = dias[["lucro", "stake"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(valores), size=(repeticoes, len(valores)))
    amostras = valores[indices]
    stake = amostras[:, :, 1].sum(axis=1)
    rois = np.divide(
        amostras[:, :, 0].sum(axis=1),
        stake,
        out=np.full(repeticoes, np.nan),
        where=stake > 0,
    )
    validos = rois[np.isfinite(rois)]
    if not len(validos):
        inferior = superior = prob_nao_positivo = None
    else:
        inferior, superior = np.quantile(validos, [0.025, 0.975])
        prob_nao_positivo = float(np.mean(validos <= 0))

    ordenado = resolvidas.sort_values(
        [colunas.data] + (["_ordem_original"] if "_ordem_original" in resolvidas.columns else []),
        kind="mergesort",
    )
    meio = len(ordenado) // 2

    def roi_parte(parte: pd.DataFrame):
        stake_parte = float(parte[colunas.stake].sum())
        return (
            float(parte[colunas.lucro].sum()) / stake_parte * 100
            if stake_parte
            else None
        )

    return {
        "metodo": "bootstrap_por_dia",
        "semente": seed,
        "dias_avaliados": int(len(dias)),
        "repeticoes": repeticoes,
        "roi_ic95_inferior_percentual": round_or_none(
            inferior * 100 if inferior is not None else None, 4
        ),
        "roi_ic95_superior_percentual": round_or_none(
            superior * 100 if superior is not None else None, 4
        ),
        "probabilidade_roi_nao_positivo_percentual": round_or_none(
            prob_nao_positivo * 100 if prob_nao_positivo is not None else None, 4
        ),
        "roi_primeira_metade_percentual": round_or_none(roi_parte(ordenado.iloc[:meio]), 4),
        "roi_segunda_metade_percentual": round_or_none(roi_parte(ordenado.iloc[meio:]), 4),
    }


def comparar_calibracao_com_baseline(
    resolvidas: pd.DataFrame, colunas: ColumnMap
) -> dict | None:
    """Compara probabilidade do modelo com a constante da taxa observada."""
    if not colunas.probabilidade:
        return None
    base = resolvidas[resolvidas["_vitoria"] | resolvidas["_derrota"]].copy()
    base = base[base[colunas.probabilidade].notna()]
    if base.empty:
        return None

    alvo = base["_vitoria"].astype(float)
    prob = base[colunas.probabilidade].clip(1e-15, 1 - 1e-15)
    taxa = float(alvo.mean())
    baseline = pd.Series(taxa, index=alvo.index).clip(1e-15, 1 - 1e-15)

    def brier(p: pd.Series) -> float:
        return float(((p - alvo) ** 2).mean())

    def logloss(p: pd.Series) -> float:
        return float(-(alvo * np.log(p) + (1 - alvo) * np.log(1 - p)).mean())

    brier_modelo = brier(prob)
    brier_baseline = brier(baseline)
    logloss_modelo = logloss(prob)
    logloss_baseline = logloss(baseline)
    return {
        "probabilidade_media_modelo_percentual": round_or_none(float(prob.mean()) * 100, 4),
        "taxa_observada_percentual": round_or_none(taxa * 100, 4),
        "brier_modelo": round_or_none(brier_modelo, 6),
        "brier_baseline": round_or_none(brier_baseline, 6),
        "log_loss_modelo": round_or_none(logloss_modelo, 6),
        "log_loss_baseline": round_or_none(logloss_baseline, 6),
        "modelo_supera_baseline_brier": bool(brier_modelo < brier_baseline),
        "modelo_supera_baseline_log_loss": bool(logloss_modelo < logloss_baseline),
    }
