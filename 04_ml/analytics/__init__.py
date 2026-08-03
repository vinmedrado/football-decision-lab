"""
Football Lab — Módulo de Auditoria Estatística (Fase 13)
==========================================================

Pacote de análise **somente-leitura** sobre o histórico de apostas já
liquidado pelo sistema. Nenhuma função aqui altera modelos, pipelines,
backtests, banco de dados ou scripts existentes — o pacote apenas lê
``historico_apostas.csv`` e produz relatórios (CSV/JSON/TXT) dentro de
``04_ml/audit_output/``.

Uso:

    python 04_ml/analytics/run_audit.py

Estrutura:

    loader.py               -> localização e carga do histórico + mapeamento de colunas
    utils.py                -> conversões numéricas/datas robustas, logging, helpers
    market_analysis.py      -> desempenho por mercado
    monthly_analysis.py     -> desempenho mensal
    odds_analysis.py        -> desempenho por faixa de odd
    drawdown_analysis.py    -> curva de banca e drawdown
    streak_analysis.py      -> sequências de greens/reds
    bankroll_analysis.py    -> evolução completa da banca
    ev_analysis.py          -> expected value (quando há probabilidade prevista)
    calibration_analysis.py -> calibração do modelo (Brier, Log Loss, ECE)
    statistical_analysis.py -> incerteza do ROI e comparação com baseline
    report.py               -> relatório executivo (.txt + .json)
    dashboard_data.py       -> agregação para dashboard (Streamlit / Power BI)
    export.py               -> helpers de exportação (CSV/JSON)
    run_audit.py             -> orquestrador (ponto de entrada único)
"""

__version__ = "13.0.0"
