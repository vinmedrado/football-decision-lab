"""
Ponto de entrada único da auditoria estatística.

Executa:

    python 04_ml/analytics/run_audit.py

Localiza o histórico automaticamente, roda todos os módulos de
análise e grava CSVs/JSON/TXT em ``04_ml/audit_output/`` — sem
nenhuma configuração manual e sem alterar qualquer outro arquivo do
projeto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Permite `python 04_ml/analytics/run_audit.py` (execução direta, fora de
# um pacote) e também `python -m 04_ml.analytics.run_audit`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analytics.loader import carregar_dados  # type: ignore
    from analytics import (  # type: ignore
        bankroll_analysis,
        calibration_analysis,
        dashboard_data,
        drawdown_analysis,
        ev_analysis,
        export,
        market_analysis,
        monthly_analysis,
        odds_analysis,
        report,
        streak_analysis,
    )
    from analytics.utils import get_logger  # type: ignore
else:
    from .loader import carregar_dados
    from . import (
        bankroll_analysis,
        calibration_analysis,
        dashboard_data,
        drawdown_analysis,
        ev_analysis,
        export,
        market_analysis,
        monthly_analysis,
        odds_analysis,
        report,
        streak_analysis,
    )
    from .utils import get_logger

logger = get_logger("run_audit")

BASE_DIR = Path(__file__).resolve().parents[1]  # 04_ml/
OUTPUT_DIR = BASE_DIR / "audit_output"

# Amostra mínima (nº de apostas) para uma categoria (mercado/liga/mês)
# concorrer a "campeã"/"pior"/top N. Único ponto de configuração,
# repassado a todos os módulos que calculam ranking.
AMOSTRA_MINIMA_RANKING = market_analysis.AMOSTRA_MINIMA_RANKING_PADRAO


def _ordenar(df, colunas):
    chaves = [colunas.data]
    if "_ordem_original" in df.columns:
        chaves.append("_ordem_original")
    return df.sort_values(chaves, kind="mergesort").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Iniciando auditoria estatística — Football Lab (Fase 13)")

    dados, colunas, caminho = carregar_dados(BASE_DIR)

    resolvidas = _ordenar(dados[dados["_resolvida"]], colunas)
    pendentes = _ordenar(dados[~dados["_resolvida"]], colunas)
    if resolvidas.empty:
        raise ValueError("O histórico não possui nenhuma aposta resolvida para analisar.")

    logger.info(
        "Resolvidas: %d | Pendentes: %d | Colunas identificadas: %s",
        len(resolvidas), int((~dados["_resolvida"]).sum()), colunas.disponiveis(),
    )

    # Módulo 1 — Mercados
    mercados = market_analysis.analisar_mercados(resolvidas, colunas, amostra_minima=AMOSTRA_MINIMA_RANKING)
    export.exportar_csv(mercados, OUTPUT_DIR / "01_market_analysis.csv")

    # Módulo 4 — Drawdown / curva da banca (calculado cedo: alimenta o mensal e o bankroll)
    curva_banca = drawdown_analysis.calcular_curva_banca(resolvidas, colunas)
    tabela_banca = bankroll_analysis.montar_evolucao_banca(curva_banca, colunas)
    export.exportar_csv(tabela_banca, OUTPUT_DIR / "04_drawdown.csv")
    curva_banca_helper = curva_banca.assign(_drawdown_pct=curva_banca["Drawdown %"])

    # Módulo 2 — Mensal
    mensal = monthly_analysis.analisar_mensal(
        resolvidas, colunas, curva_banca_helper, amostra_minima=AMOSTRA_MINIMA_RANKING
    )
    export.exportar_csv(mensal, OUTPUT_DIR / "02_month_analysis.csv")

    # Módulo 3 — Odds
    odds = odds_analysis.analisar_odds(resolvidas, colunas)
    export.exportar_csv(odds, OUTPUT_DIR / "03_odds_analysis.csv")

    # Módulo 5 — Streaks
    resumo_streaks, streaks_distribuicao = streak_analysis.calcular_streaks(resolvidas, colunas)
    export.exportar_csv(streaks_distribuicao, OUTPUT_DIR / "05_streaks.csv")

    # Módulo 6 — Bankroll (evolução completa; mesma tabela enxuta do módulo 4)
    export.exportar_csv(tabela_banca, OUTPUT_DIR / "06_bankroll.csv")

    # Módulo 7 — Expected Value (condicional)
    ev = ev_analysis.calcular_ev(resolvidas, colunas)
    export.exportar_csv(ev, OUTPUT_DIR / "07_ev_analysis.csv")

    # Módulo 8 — Calibração (condicional)
    resultado_calibracao = calibration_analysis.calcular_calibracao(resolvidas, colunas)
    calibracao_curva, resumo_calibracao = (
        resultado_calibracao if resultado_calibracao is not None else (None, None)
    )
    export.exportar_csv(calibracao_curva, OUTPUT_DIR / "08_calibration.csv")

    # Módulo 9 — Ligas (condicional)
    ligas = market_analysis.analisar_ligas(resolvidas, colunas, amostra_minima=AMOSTRA_MINIMA_RANKING)
    export.exportar_csv(ligas, OUTPUT_DIR / "09_leagues.csv")

    # Pendências ficam visíveis para reconciliação, com todos os campos de
    # origem disponíveis e sem colunas auxiliares internas.
    pendentes_export = pendentes[[c for c in pendentes.columns if not c.startswith("_")]].copy()
    if not pendentes_export.empty:
        hoje = pd.Timestamp.now().normalize()
        datas_pendentes = pd.to_datetime(pendentes_export[colunas.data], errors="coerce")
        pendentes_export["Dias Pendente"] = (hoje - datas_pendentes).dt.days.clip(lower=0)
        base_match_col = next(
            (c for c in pendentes_export.columns if str(c).strip().lower() == "base_match"),
            None,
        )
        if base_match_col:
            sem_match = ~pendentes_export[base_match_col].astype(str).str.strip().str.lower().isin(
                {"true", "1", "sim", "yes"}
            )
        else:
            sem_match = pd.Series(False, index=pendentes_export.index)
        vencida = datas_pendentes < hoje
        pendentes_export["Pendência Vencida"] = vencida
        pendentes_export["Motivo Pendência"] = "AGUARDANDO_DATA_OU_SETTLEMENT"
        pendentes_export.loc[vencida, "Motivo Pendência"] = "SETTLEMENT_ATRASADO"
        pendentes_export.loc[sem_match, "Motivo Pendência"] = "SEM_MATCH_BASE_OFICIAL"
    export.exportar_csv(pendentes_export, OUTPUT_DIR / "10_pending_records.csv")

    # Módulo 10 — Relatório executivo
    resumo_executivo = report.montar_resumo_executivo(
        dados, resolvidas, colunas, curva_banca, resumo_streaks,
        mercados, ligas, mensal, resumo_calibracao, str(caminho),
    )
    export.exportar_json(resumo_executivo, OUTPUT_DIR / "00_resumo_executivo.json")
    export.exportar_texto(
        report.montar_texto_relatorio(resumo_executivo, str(OUTPUT_DIR)),
        OUTPUT_DIR / "RELATORIO_EXECUTIVO.txt",
    )

    # Dashboard consolidado (Streamlit / Power BI)
    dashboard = dashboard_data.montar_dashboard_data(
        resumo_executivo, mercados, mensal, odds, tabela_banca,
        streaks_distribuicao, ev, calibracao_curva, ligas,
    )
    export.exportar_json(dashboard, OUTPUT_DIR / "dashboard_data.json")

    logger.info("Auditoria concluída com sucesso. Saída em: %s", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:  # noqa: BLE001 — ponto de entrada precisa reportar qualquer falha
        logger.error("Falha na auditoria: %s", erro)
        raise
