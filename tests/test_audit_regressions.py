from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "04_ml"
for path in (ROOT, ML_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytics.drawdown_analysis import calcular_curva_banca  # noqa: E402
from analytics.loader import ColumnMap, carregar_dados  # noqa: E402
from analytics.report import montar_resumo_executivo  # noqa: E402
from analytics.streak_analysis import ResumoStreaks  # noqa: E402
from core import performance_provider  # noqa: E402
from utils.data_quality import run_quality_checks  # noqa: E402


def _history_rows() -> list[dict]:
    return [
        {
            "data": "2026-01-01",
            "jogo": "A x B",
            "liga": "L1",
            "mercado": "TG_FT_O25",
            "resultado": "ganhou",
            "odd": 2.0,
            "valor_apostado": 5.0,
            "lucro": 5.0,
            "banca_apos": 305.0,
            "origem": "backfill_simulado",
        },
        {
            "data": "2026-01-02",
            "jogo": "C x D",
            "liga": "L1",
            "mercado": "TG_FT_O25",
            "resultado": "ganhou",
            "odd": 2.0,
            "valor_apostado": 5.0,
            "lucro": 5.0,
            "banca_apos": 310.0,
            "origem": "backfill_simulado",
        },
        {
            "data": "2026-01-03",
            "jogo": "E x F",
            "liga": "L1",
            "mercado": "TG_FT_O25",
            "resultado": "perdeu",
            "odd": 2.0,
            "valor_apostado": 5.0,
            "lucro": -5.0,
            "banca_apos": 305.0,
            "origem": "backfill_simulado",
        },
    ]


def test_performance_provider_uses_valor_apostado_as_stake(tmp_path, monkeypatch):
    history = tmp_path / "historico_apostas.csv"
    pd.DataFrame(_history_rows()).to_csv(history, index=False)
    monkeypatch.setattr(performance_provider, "HISTORICO_PATH", history)

    metrics, alerts = performance_provider._read_history_performance()

    assert alerts == []
    assert metrics["TG_FT_O25"]["stake"] == 15.0
    assert metrics["TG_FT_O25"]["profit"] == 5.0
    assert metrics["TG_FT_O25"]["roi"] == pytest.approx(1 / 3)
    assert metrics["TG_FT_O25"]["data_scope"] == "SIMULATED"


def test_data_quality_accepts_current_market_codes():
    report = run_quality_checks(pd.DataFrame(_history_rows()))
    checks = {issue["check"] for issue in report["issues"]}
    assert "invalid_mercados" not in checks


def test_loader_rejects_unknown_result_status(tmp_path):
    bank = tmp_path / "banca"
    bank.mkdir()
    row = _history_rows()[0]
    row["resultado"] = "talvez"
    pd.DataFrame([row]).to_csv(bank / "historico_apostas.csv", index=False)

    with pytest.raises(ValueError, match="status desconhecido"):
        carregar_dados(tmp_path)


def test_drawdown_includes_initial_bank_before_first_loss():
    df = pd.DataFrame(
        {
            "data": pd.to_datetime(["2026-01-01"]),
            "mercado": ["TG_FT_O25"],
            "resultado": ["perdeu"],
            "stake": [5.0],
            "lucro": [-5.0],
            "_ordem_original": [0],
        }
    )
    columns = ColumnMap(
        data="data",
        mercado="mercado",
        resultado="resultado",
        stake="stake",
        lucro="lucro",
    )

    curve = calcular_curva_banca(df, columns, banca_inicial=300.0)

    assert curve.loc[0, "Curva da Banca"] == 295.0
    assert curve.loc[0, "Novo Pico"] == False  # noqa: E712
    assert curve.loc[0, "Drawdown"] == -5.0
    assert curve.loc[0, "Drawdown %"] == pytest.approx(-5 / 300 * 100)


def test_executive_summary_labels_simulated_data():
    raw = pd.DataFrame(_history_rows())
    raw["data"] = pd.to_datetime(raw["data"])
    raw["_ordem_original"] = range(len(raw))
    raw["_vitoria"] = raw["resultado"].eq("ganhou")
    raw["_derrota"] = raw["resultado"].eq("perdeu")
    raw["_push"] = False
    raw["_resolvida"] = True
    columns = ColumnMap(
        data="data",
        mercado="mercado",
        resultado="resultado",
        stake="valor_apostado",
        lucro="lucro",
        odd="odd",
        banca="banca_apos",
        liga="liga",
        origem="origem",
    )
    curve = calcular_curva_banca(raw, columns, banca_inicial=300.0)
    market = pd.DataFrame(
        [{"Mercado": "TG_FT_O25", "Ranking por ROI": 1, "Quantidade": 3}]
    )
    month = pd.DataFrame(
        [{"Mês": "2026-01", "Ranking por ROI": 1, "Quantidade": 3}]
    )

    summary = montar_resumo_executivo(
        raw,
        raw,
        columns,
        curve,
        ResumoStreaks(2, 1, 2.0, 1.0),
        market,
        None,
        month,
        None,
        "test.csv",
    )

    assert summary["escopo_dados"] == "SIMULADO"
    assert summary["evidencia_lucro_real"] is False
    assert summary["origens_dados"] == {"backfill_simulado": 3}
