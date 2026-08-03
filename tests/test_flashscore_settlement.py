from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "04_ml"
for path in (ROOT, ML_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.flashscore_settlement import (  # noqa: E402
    find_local_match_id,
    parse_flashscore_html,
    validate_match_teams,
)


def _page(
    *,
    match_id: str = "SrA26GOH",
    home: str = "Sheffield Utd",
    away: str = "Oxford Utd",
    score: str = "3-1",
    status: str = "Finished",
    played_at: str = "03.02.2026 21:45",
    halves: str = "<h4>1st Half: <b>1-1</b></h4><h4>2nd Half: <b>2-0</b></h4>",
    extra_details: str = "",
) -> str:
    padding = "x" * 600
    return f"""
    <html><body id="detail" data-match-id='{match_id}'>
    <h3><a>{home}</a> - <a>{away}</a></h3>
    <div class="detail"><b>{score}</b> (1-1,2-0)</div>
    <div class="detail">{status}</div>
    {extra_details}
    <div class="detail">{played_at}</div>
    {halves}
    {padding}
    </body></html>
    """


def test_parse_finished_match_and_regulation_score():
    match = parse_flashscore_html(_page(), "SrA26GOH")

    assert match.home == "Sheffield Utd"
    assert match.away == "Oxford Utd"
    assert match.status == "Finished"
    assert (match.ft_home, match.ft_away) == (3, 1)
    assert match.halftime_score == (1, 1)
    assert match.regulation_score == (3, 1)
    assert match.is_finished is True


def test_extra_time_uses_first_and_second_half_for_regulation():
    page = _page(
        score="4-3",
        status="After Extra Time",
        halves="<h4>1st Half: <b>1-0</b></h4><h4>2nd Half: <b>1-2</b></h4>",
    )

    match = parse_flashscore_html(page, "SrA26GOH")

    assert (match.ft_home, match.ft_away) == (4, 3)
    assert match.regulation_score == (2, 2)


def test_postponed_match_is_not_finished():
    page = _page(score="-", status="Postponed", halves="")

    match = parse_flashscore_html(page, "SrA26GOH")

    assert match.status == "Postponed"
    assert match.is_finished is False
    assert match.regulation_score is None


def test_full_time_with_penalties_keeps_regulation_score_and_date():
    page = _page(
        score="1-1",
        status="Full time",
        played_at="05.04.2026 07:00",
        halves="<h4>1st Half: <b>1-0</b></h4><h4>2nd Half: <b>0-1</b></h4>",
        extra_details='<div class="detail">Penalties: 8-7</div>',
    )

    match = parse_flashscore_html(page, "SrA26GOH")

    assert match.is_finished is True
    assert match.played_at == "05.04.2026 07:00"
    assert match.regulation_score == (1, 1)


def test_team_validation_accepts_common_united_abbreviation():
    match = parse_flashscore_html(_page(home="Sheffield United"), "SrA26GOH")

    accepted, confidence = validate_match_teams("Sheffield Utd", "Oxford Utd", match)

    assert accepted is True
    assert confidence == 1.0


def test_find_local_id_from_original_daily_file(tmp_path):
    daily = tmp_path / "jogos_do_dia_2026-01-04.csv"
    pd.DataFrame(
        [
            {
                "Id": "4KuMjeBs",
                "Date": "2026-01-04",
                "Home": "Cambridge Utd",
                "Away": "Grimsby",
            },
            {
                "Id": "other",
                "Date": "2026-01-04",
                "Home": "Another",
                "Away": "Match",
            },
        ]
    ).to_csv(daily, index=False)

    found = find_local_match_id(
        tmp_path,
        "2026-01-04",
        "Cambridge United",
        "Grimsby",
    )

    assert found is not None
    assert found.match_id == "4KuMjeBs"
    assert found.confidence == 1.0


def test_cli_apply_uses_cache_and_preserves_original_date(tmp_path, monkeypatch):
    history_path = tmp_path / "historico_apostas.csv"
    raw_dir = tmp_path / "daily"
    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "preview.csv"
    raw_dir.mkdir()
    cache_dir.mkdir()

    pd.DataFrame(
        [
            {
                "data": "2026-01-04",
                "liga": "ENGLAND 2",
                "jogo": "Sheffield Utd x Oxford Utd",
                "home": "Sheffield Utd",
                "away": "Oxford Utd",
                "mercado": "TG_FT_O25",
                "odd": 1.83,
                "valor_apostado": 5.0,
                "resultado": "pendente",
                "lucro": 0.0,
                "banca_apos": 250.0,
                "origem": "backfill_simulado",
            }
        ]
    ).to_csv(history_path, index=False)
    pd.DataFrame(
        [
            {
                "Id": "SrA26GOH",
                "Date": "2026-01-04",
                "Home": "Sheffield Utd",
                "Away": "Oxford Utd",
            }
        ]
    ).to_csv(raw_dir / "jogos_do_dia_2026-01-04.csv", index=False)
    (cache_dir / "SrA26GOH.json").write_text(
        json.dumps(
            {
                "match_id": "SrA26GOH",
                "source_url": "https://www.flashscore.mobi/match/SrA26GOH/",
                "home": "Sheffield Utd",
                "away": "Oxford Utd",
                "status": "Finished",
                "played_at": "03.02.2026 21:45",
                "ft_home": 3,
                "ft_away": 1,
                "ht_home": 1,
                "ht_away": 1,
                "second_home": 2,
                "second_away": 0,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    script_path = ML_DIR / "05_settle_flashscore.py"
    spec = importlib.util.spec_from_file_location("settle_flashscore_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--apply",
            "--skip-rebuild-bank",
            "--history",
            str(history_path),
            "--raw-daily-dir",
            str(raw_dir),
            "--cache-dir",
            str(cache_dir),
            "--report",
            str(report_path),
        ],
    )

    assert module.main() == 0

    updated = pd.read_csv(history_path)
    assert updated.loc[0, "data"] == "2026-01-04"
    assert updated.loc[0, "data_realizacao"] == "2026-02-03"
    assert updated.loc[0, "resultado"] == "ganhou"
    assert updated.loc[0, "lucro"] == 4.15
    assert updated.loc[0, "settlement_match_id"] == "SrA26GOH"
    assert report_path.exists()
    assert len(list(tmp_path.glob("historico_apostas.backup_flashscore_*.csv"))) == 1
