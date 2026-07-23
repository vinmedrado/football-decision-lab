from datetime import datetime

import pytest

from paper_contract import PaperPolicy, PaperPrediction


def test_capture_inside_window_is_valid() -> None:
    policy = PaperPolicy(cycle_id="PAPER_EXAMPLE_V1")
    prediction = PaperPrediction(
        prediction_id="demo-001",
        captured_at=datetime(2026, 7, 24, 18, 0),
        kickoff=datetime(2026, 7, 24, 19, 0),
        market="TG_FT_O25",
        probability=0.64,
        odd=1.92,
        model_sha256="a" * 64,
    )

    prediction.validate(policy)


def test_capture_after_kickoff_is_rejected() -> None:
    policy = PaperPolicy(cycle_id="PAPER_EXAMPLE_V1")
    prediction = PaperPrediction(
        prediction_id="demo-002",
        captured_at=datetime(2026, 7, 24, 19, 1),
        kickoff=datetime(2026, 7, 24, 19, 0),
        market="TG_FT_O25",
        probability=0.64,
        odd=1.92,
        model_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="fora da janela"):
        prediction.validate(policy)


def test_real_bets_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="PAPER_ONLY"):
        PaperPolicy(cycle_id="PAPER_EXAMPLE_V1", allow_real_bets=True)
