"""Módulo 10 — Relatório executivo (resumo geral + texto formatado)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .calibration_analysis import ResumoCalibracao
from .drawdown_analysis import resumo_maior_drawdown
from .ev_analysis import UNIVERSO_EV
from .loader import ColumnMap
from .streak_analysis import ResumoStreaks
from .statistical_analysis import calcular_incerteza_roi, comparar_calibracao_com_baseline
from .utils import round_or_none, safe_divide_scalar

# Critério ÚNICO e explícito usado para eleger "campeão"/"pior" tanto em
# mercados/ligas (Módulos 1 e 9) quanto em meses (Módulo 2): ROI, entre as
# categorias que atingiram a amostra mínima configurada em
# market_analysis.AMOSTRA_MINIMA_RANKING_PADRAO. Antes, mercado usava ROI
# e mês usava Lucro — critérios diferentes sem nenhuma indicação disso no
# relatório. Agora ambos usam a coluna "Ranking por ROI" já calculada
# pelos respectivos módulos (que já respeita a amostra mínima).
CRITERIO_DESTAQUE = "ROI (entre categorias com amostra mínima suficiente)"

# Documentação explícita do tratamento de "push" em cada métrica, exposta
# no relatório para que a diferença de universo entre módulos não fique
# implícita no código-fonte.
TRATAMENTO_PUSH = {
    "roi_e_lucro": "push incluído no stake total e no lucro (capital ficou em risco)",
    "win_rate": "push excluído do denominador (apenas vitória/derrota contam)",
    "calibracao": "push excluído (não há alvo binário definido para push)",
    "expected_value": UNIVERSO_EV,
    "streaks": "push ignorado — não conta como vitória/derrota e não interrompe a sequência",
}


def _linha_qualificada(tabela: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas as linhas elegíveis a destaque (ranking não nulo,
    ou seja, que atingiram a amostra mínima configurada)."""
    if tabela is None or tabela.empty or "Ranking por ROI" not in tabela.columns:
        return tabela.iloc[0:0] if tabela is not None else pd.DataFrame()
    return tabela[tabela["Ranking por ROI"].notna()].sort_values("Ranking por ROI")


def montar_resumo_executivo(
    dados: pd.DataFrame,
    resolvidas: pd.DataFrame,
    colunas: ColumnMap,
    curva_banca: pd.DataFrame,
    streaks: ResumoStreaks,
    mercados: pd.DataFrame,
    ligas: Optional[pd.DataFrame],
    mensal: pd.DataFrame,
    calibracao: Optional[ResumoCalibracao],
    caminho_arquivo: str,
) -> dict:
    """Consolida todas as métricas do topo do funil em um único
    dicionário — usado tanto para ``00_resumo_executivo.json`` quanto
    como base do texto do relatório executivo.

    "Campeão"/"pior"/"top N" de mercado, liga e mês usam todos o mesmo
    critério (ROI, restrito a categorias com amostra mínima — ver
    ``CRITERIO_DESTAQUE``), documentado explicitamente no resultado.
    """
    vitorias = int(resolvidas["_vitoria"].sum())
    derrotas = int(resolvidas["_derrota"].sum())
    push = int(resolvidas["_push"].sum())
    total_decididas = vitorias + derrotas

    stake_total = float(resolvidas[colunas.stake].sum())
    lucro_total = float(resolvidas[colunas.lucro].sum())

    drawdown_info = resumo_maior_drawdown(curva_banca, colunas)

    mercados_qualificados = _linha_qualificada(mercados)
    mercado_campeao = mercados_qualificados.iloc[0].to_dict() if not mercados_qualificados.empty else None
    mercado_pior = (
        mercados_qualificados.iloc[-1].to_dict()
        if len(mercados_qualificados) >= 2
        else None
    )

    mensal_qualificado = _linha_qualificada(mensal)
    melhor_mes = mensal_qualificado.iloc[0].to_dict() if not mensal_qualificado.empty else None
    pior_mes = mensal_qualificado.iloc[-1].to_dict() if not mensal_qualificado.empty else None

    ligas_qualificadas = _linha_qualificada(ligas) if ligas is not None else None
    origens = (
        dados[colunas.origem].fillna("nao_informada").astype(str).str.strip().value_counts().to_dict()
        if colunas.origem
        else {"nao_informada": int(len(dados))}
    )
    origens_normalizadas = [str(origem).lower() for origem in origens]
    flags_simulado = [
        any(token in origem for token in ("simulad", "backfill", "paper"))
        for origem in origens_normalizadas
    ]
    escopo_dados = (
        "NAO_INFORMADO"
        if origens_normalizadas == ["nao_informada"]
        else "SIMULADO"
        if flags_simulado and all(flags_simulado)
        else "MISTO"
        if any(flags_simulado)
        else "REAL"
    )
    incerteza_roi = calcular_incerteza_roi(resolvidas, colunas)
    comparacao_calibracao = comparar_calibracao_com_baseline(resolvidas, colunas)

    resumo = {
        "arquivo_analisado": caminho_arquivo,
        "escopo_dados": escopo_dados,
        "origens_dados": {str(k): int(v) for k, v in origens.items()},
        "evidencia_lucro_real": escopo_dados == "REAL",
        "registros_totais": int(len(dados)),
        "apostas_resolvidas": int(len(resolvidas)),
        "apostas_pendentes": int((~dados["_resolvida"]).sum()),
        "vitorias": vitorias,
        "derrotas": derrotas,
        "push": push,
        "win_rate_percentual": round_or_none(
            safe_divide_scalar(vitorias, total_decididas) * 100 if total_decididas else None, 4
        ),
        "stake_total": round_or_none(stake_total),
        "lucro_total": round_or_none(lucro_total),
        "roi_percentual_sobre_stake": round_or_none(
            safe_divide_scalar(lucro_total, stake_total) * 100 if stake_total else None, 4
        ),
        "odd_media": round_or_none(float(resolvidas[colunas.odd].mean()), 4) if colunas.odd else None,
        "maior_sequencia_vitorias": streaks.maior_sequencia_vitorias,
        "maior_sequencia_derrotas": streaks.maior_sequencia_derrotas,
        "sequencia_media_vitorias": streaks.sequencia_media_vitorias,
        "sequencia_media_derrotas": streaks.sequencia_media_derrotas,
        **{f"drawdown_{k}": v for k, v in drawdown_info.items()},
        "criterio_destaque_categorias": CRITERIO_DESTAQUE,
        "tratamento_push_por_metrica": TRATAMENTO_PUSH,
        **incerteza_roi,
        "mercado_campeao": mercado_campeao.get("Mercado") if mercado_campeao else None,
        "mercado_pior": mercado_pior.get("Mercado") if mercado_pior else None,
        "melhor_mes": melhor_mes.get("Mês") if melhor_mes else None,
        "pior_mes": pior_mes.get("Mês") if pior_mes else None,
        "top_10_mercados": mercados_qualificados.head(10)["Mercado"].tolist() if not mercados_qualificados.empty else [],
        "top_10_ligas": (
            ligas_qualificadas.head(10)["Liga"].tolist()
            if ligas_qualificadas is not None and not ligas_qualificadas.empty
            else []
        ),
    }

    if calibracao is not None:
        resumo["calibracao_brier_score"] = calibracao.brier_score
        resumo["calibracao_log_loss"] = calibracao.log_loss
        resumo["calibracao_ece"] = calibracao.ece
        resumo["calibracao_apostas_avaliadas"] = calibracao.apostas_avaliadas
    if comparacao_calibracao is not None:
        resumo.update(comparacao_calibracao)

    return resumo


def montar_texto_relatorio(resumo: dict, pasta_saida: str) -> list[str]:
    """Formata o resumo executivo como texto simples e legível para
    ``RELATORIO_EXECUTIVO.txt``.
    """

    def fmt(valor, sufixo: str = "") -> str:
        return "N/D" if valor is None else f"{valor}{sufixo}"

    linhas = [
        "FOOTBALL LAB — AUDITORIA ESTATÍSTICA PROFISSIONAL",
        "=" * 52,
        f"Arquivo analisado: {resumo['arquivo_analisado']}",
        f"Escopo dos dados:   {resumo.get('escopo_dados', 'N/D')}",
        f"Origens:            {resumo.get('origens_dados', {})}",
        (
            "AVISO: lucro e ROI abaixo pertencem a simulação/backfill; "
            "não constituem evidência de lucro real."
            if resumo.get("escopo_dados") == "SIMULADO"
            else "Escopo contém dados reais ou mistos; valide a origem por registro."
        ),
        "",
        "RESUMO GERAL",
        "-" * 52,
        f"Registros totais:       {resumo['registros_totais']}",
        f"Apostas liquidadas:     {resumo['apostas_resolvidas']}",
        f"Apostas pendentes:      {resumo['apostas_pendentes']}",
        f"Vitórias:               {resumo['vitorias']}",
        f"Derrotas:               {resumo['derrotas']}",
        f"Push:                   {resumo['push']}",
        f"Win rate:               {fmt(resumo['win_rate_percentual'], '%')}",
        f"Stake total:            R$ {fmt(resumo['stake_total'])}",
        f"Lucro total:            R$ {fmt(resumo['lucro_total'])}",
        f"ROI sobre stake:        {fmt(resumo['roi_percentual_sobre_stake'], '%')}",
        f"Odd média:              {fmt(resumo['odd_media'])}",
        f"IC95% ROI (por dia):    {fmt(resumo.get('roi_ic95_inferior_percentual'), '%')} a "
        f"{fmt(resumo.get('roi_ic95_superior_percentual'), '%')}",
        f"P(ROI <= 0):            {fmt(resumo.get('probabilidade_roi_nao_positivo_percentual'), '%')}",
        f"ROI primeira metade:    {fmt(resumo.get('roi_primeira_metade_percentual'), '%')}",
        f"ROI segunda metade:     {fmt(resumo.get('roi_segunda_metade_percentual'), '%')}",
        "",
        "SEQUÊNCIAS",
        "-" * 52,
        f"Maior sequência de vitórias: {resumo['maior_sequencia_vitorias']}",
        f"Maior sequência de derrotas: {resumo['maior_sequencia_derrotas']}",
        f"Sequência média de vitórias: {resumo['sequencia_media_vitorias']}",
        f"Sequência média de derrotas: {resumo['sequencia_media_derrotas']}",
        "",
        "DRAWDOWN",
        "-" * 52,
        f"Maior drawdown (valor):  R$ {fmt(resumo.get('drawdown_maior_drawdown_valor'))}",
        f"Maior drawdown (%):      {fmt(resumo.get('drawdown_maior_drawdown_percentual'), '%')}",
        f"Data do pico:            {fmt(resumo.get('drawdown_data_pico'))}",
        f"Data do vale:            {fmt(resumo.get('drawdown_data_vale'))}",
        f"Data de recuperação:     {fmt(resumo.get('drawdown_data_recuperacao'))}",
        f"Tempo de recuperação:    {fmt(resumo.get('drawdown_tempo_recuperacao_dias'), ' dias')}",
        f"Banca não-positiva em algum ponto: {fmt(resumo.get('drawdown_banca_nao_positiva_em_algum_ponto'))}",
        "",
        f"DESTAQUES (critério: {resumo['criterio_destaque_categorias']})",
        "-" * 52,
        f"Mercado campeão: {fmt(resumo['mercado_campeao'])}",
        f"Mercado pior:    {fmt(resumo['mercado_pior'])}",
        f"Melhor mês:      {fmt(resumo['melhor_mes'])}",
        f"Pior mês:        {fmt(resumo['pior_mes'])}",
        "",
        "TOP 10 MERCADOS: " + (", ".join(resumo["top_10_mercados"]) or "N/D (nenhuma categoria atingiu a amostra mínima)"),
        "TOP 10 LIGAS: " + (", ".join(resumo["top_10_ligas"]) or "N/D (nenhuma categoria atingiu a amostra mínima)"),
        "",
        "TRATAMENTO DE PUSH POR MÉTRICA",
        "-" * 52,
    ] + [f"- {chave}: {valor}" for chave, valor in resumo["tratamento_push_por_metrica"].items()]

    if "calibracao_brier_score" in resumo:
        linhas += [
            "",
            "CALIBRAÇÃO DO MODELO",
            "-" * 52,
            f"Brier Score: {resumo['calibracao_brier_score']}",
            f"Log Loss:    {resumo['calibracao_log_loss']}",
            f"ECE:         {resumo['calibracao_ece']}",
            f"Apostas avaliadas: {resumo['calibracao_apostas_avaliadas']}",
            f"Brier baseline: {fmt(resumo.get('brier_baseline'))}",
            f"Log Loss baseline: {fmt(resumo.get('log_loss_baseline'))}",
            f"Supera baseline (Brier): {fmt(resumo.get('modelo_supera_baseline_brier'))}",
            f"Supera baseline (Log Loss): {fmt(resumo.get('modelo_supera_baseline_log_loss'))}",
        ]

    linhas += ["", f"Relatórios completos salvos em: {pasta_saida}"]
    return linhas
