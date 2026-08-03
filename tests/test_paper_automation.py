from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_fill_blocked_and_paper_fill_allowed():
    responsible = load_module("responsible_test", "04_ml/11_responsible_mode.py")
    real_allowed, _ = responsible.is_auto_fill_allowed()
    paper_allowed, _ = responsible.is_paper_fill_allowed()
    assert real_allowed is False
    assert paper_allowed is True


def test_frozen_paper_bundle_hashes_are_valid():
    manager = load_module("paper_manager_test", "04_ml/paper_model_manager.py")
    manifest_path = manager.validate()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["real_bets_allowed"] is False
    assert manifest["cycle_id"] == "PAPER_2026_07_TG_FT_O25_V1"


def test_schedule_does_not_backfill_jobs_before_not_before():
    orchestrator = load_module("orchestrator_test", "automation_orchestrator.py")
    config = json.loads(orchestrator.CONFIG_PATH.read_text(encoding="utf-8"))
    now = datetime(2026, 7, 23, 13, 0, tzinfo=ZoneInfo("America/Sao_Paulo"))
    assert orchestrator.due_jobs(config, {"jobs": {}}, now) == []


def test_interval_schedule_rounds_to_current_slot():
    orchestrator = load_module("orchestrator_interval_test", "automation_orchestrator.py")
    now = datetime(2026, 7, 24, 13, 37, tzinfo=ZoneInfo("America/Sao_Paulo"))
    slots = orchestrator.slots_for_job(
        "paper_scan",
        {"frequency": "interval", "every_minutes": 15},
        now,
    )
    assert slots[0][0] == "2026-07-24@13:30"
    assert slots[0][1].minute == 30


def test_dynamic_settlement_only_selects_match_after_expected_end():
    settlement = load_module("dynamic_settlement_test", "04_ml/dynamic_settlement.py")
    config = {
        "cycle_id": "CYCLE",
        "settlement_first_attempt_minutes_after_kickoff": 130,
    }
    history = pd.DataFrame([
        {
            "data": "2026-07-24", "home": "A", "away": "B", "mercado": "TG_FT_O25",
            "resultado": "pendente", "origem": "paper_forward", "paper_cycle_id": "CYCLE",
            "kickoff_at": "2026-07-24T10:00:00",
        },
        {
            "data": "2026-07-24", "home": "C", "away": "D", "mercado": "TG_FT_O25",
            "resultado": "pendente", "origem": "paper_forward", "paper_cycle_id": "CYCLE",
            "kickoff_at": "2026-07-24T12:00:00",
        },
    ])
    now = datetime(2026, 7, 24, 12, 15, tzinfo=ZoneInfo("America/Sao_Paulo"))
    due = settlement.pending_paper(history, config, now, {"bets": {}})
    assert due.index.tolist() == [0]


def test_paper_metrics_are_isolated_and_reproducible():
    monitor = load_module("paper_monitor_test", "04_ml/paper_monitor.py")
    frame = pd.DataFrame([
        {"resultado": "ganhou", "valor_apostado": 5, "lucro": 4, "confianca": 0.6},
        {"resultado": "perdeu", "valor_apostado": 5, "lucro": -5, "confianca": 0.7},
        {"resultado": "pendente", "valor_apostado": 5, "lucro": 0, "confianca": 0.8},
    ])
    result = monitor.metrics(frame, 250.0)
    assert result["bets"] == 3
    assert result["settled"] == 2
    assert result["pending"] == 1
    assert result["profit"] == -1
    assert result["roi"] == -0.1
    assert result["brier"] == 0.325


def test_dashboard_does_not_promote_legacy_predictions_to_official_signals():
    dashboard = load_module("dashboard_signal_source_test", "web_dashboard_lux/app.py")
    result = dashboard.previsoes("2026-07-23")
    assert result["origem"] == "legado_pre_ciclo"
    assert result["oficial"] is False
    assert result["sinais"] == 0
    assert result["sinais_no_arquivo"] == 2
    assert result["top"] == []
    assert result["linhas"] == []


def test_dashboard_recognizes_only_active_cycle_directory_as_official():
    dashboard = load_module("dashboard_active_cycle_source_test", "web_dashboard_lux/app.py")
    official_path = dashboard.active_paper_predictions_dir() / "previsoes_2099-01-01.csv"
    source = dashboard.prediction_source(official_path)
    assert source["origem"] == "paper_oficial"
    assert source["oficial"] is True
    assert source["ciclo_id"] == "PAPER_2026_07_TG_FT_O25_V1"


def test_float_encoded_boolean_flags_are_recognized():
    registrar = load_module("paper_registrar_float_bool_test", "04_ml/06_registrar_paper.py")
    predictor = load_module("paper_predict_float_bool_test", "04_ml/paper_predict.py")
    dashboard = load_module("dashboard_float_bool_test", "web_dashboard_lux/app.py")
    flags = pd.Series([1.0, 0.0, "1.00", "true", "false"])
    expected = [True, False, True, True, False]
    assert registrar.bool_mask(flags).tolist() == expected
    assert predictor.bool_mask(flags).tolist() == expected
    assert [dashboard.as_bool(value) for value in flags] == expected


def test_recovery_requires_capture_before_kickoff():
    registrar = load_module("paper_registrar_prospective_test", "04_ml/06_registrar_paper.py")
    timezone = ZoneInfo("America/Sao_Paulo")
    valid = pd.Series({
        "paper_generated_at": "2026-07-24T11:45:00-03:00",
        "kickoff_at": "2026-07-24T13:00:00",
    })
    invalid = pd.Series({
        "paper_generated_at": "2026-07-24T13:01:00-03:00",
        "kickoff_at": "2026-07-24T13:00:00",
    })
    assert registrar.captured_before_kickoff(valid, timezone) is True
    assert registrar.captured_before_kickoff(invalid, timezone) is False


def test_paper_scan_refreshes_monitor_and_nightly_check_validates_frozen_bundle():
    orchestrator = load_module("orchestrator_paper_monitor_test", "automation_orchestrator.py")
    now = datetime(2026, 7, 27, 15, 0, tzinfo=ZoneInfo("America/Sao_Paulo"))
    scan_commands, _ = orchestrator.job_commands("paper_scan", now)
    nightly_commands, _ = orchestrator.job_commands("paper_monitor", now)
    assert Path(scan_commands[-1][1]).name == "paper_monitor.py"
    assert Path(nightly_commands[-1][1]).name == "paper_model_manager.py"
    assert nightly_commands[-1][2] == "validate"
