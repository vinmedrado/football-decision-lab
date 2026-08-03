#!/usr/bin/env python3
"""Liquida entradas paper quando o jogo já deveria ter terminado."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "config" / "paper_mode.json"
HISTORY_PATH = BASE_DIR / "banca" / "historico_apostas.csv"
STATE_PATH = BASE_DIR / "reports" / "dynamic_settlement_state.json"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def pending_paper(history: pd.DataFrame, config: dict, now: datetime, state: dict) -> pd.DataFrame:
    if history.empty or "kickoff_at" not in history.columns or "origem" not in history.columns:
        return pd.DataFrame()
    result = history["resultado"].fillna("").astype(str).str.strip().str.lower()
    origin = history["origem"].fillna("").astype(str).str.strip().str.lower()
    mask = result.eq("pendente") & origin.eq("paper_forward")
    if "paper_cycle_id" in history.columns:
        mask &= history["paper_cycle_id"].fillna("").astype(str).eq(str(config["cycle_id"]))
    frame = history.loc[mask].copy()
    if frame.empty:
        return frame
    kickoff = pd.to_datetime(frame["kickoff_at"], errors="coerce")
    timezone = now.tzinfo
    if getattr(kickoff.dt, "tz", None) is None:
        kickoff = kickoff.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        kickoff = kickoff.dt.tz_convert(timezone)
    frame["_kickoff"] = kickoff
    frame["_bet_key"] = frame.apply(
        lambda row: "|".join([
            str(config["cycle_id"]),
            str(row.get("data", "")).strip().lower(),
            str(row.get("home", "")).strip().lower(),
            str(row.get("away", "")).strip().lower(),
            str(row.get("mercado", "")).strip().lower(),
        ]),
        axis=1,
    )
    first_delay = int(config["settlement_first_attempt_minutes_after_kickoff"])
    frame["_first_due"] = frame["_kickoff"] + pd.to_timedelta(first_delay, unit="m")

    def next_due(row) -> datetime:
        stored = (state.get("bets", {}).get(row["_bet_key"], {}) or {}).get("next_attempt_at")
        if stored:
            try:
                return datetime.fromisoformat(stored).astimezone(timezone)
            except (TypeError, ValueError):
                pass
        return row["_first_due"]

    frame["_next_due"] = frame.apply(next_due, axis=1)
    return frame[frame["_kickoff"].notna() & frame["_next_due"].le(now)].copy()


def run(command: list[str]) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT_DIR).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_json(CONFIG_PATH, {})
    timezone = ZoneInfo(str(config.get("timezone", "America/Sao_Paulo")))
    now = datetime.fromisoformat(args.now).astimezone(timezone) if args.now else datetime.now(timezone)
    state = load_json(STATE_PATH, {"bets": {}})
    state.setdefault("bets", {})
    state["last_check_at"] = now.isoformat(timespec="seconds")

    if not HISTORY_PATH.exists():
        print("Histórico ainda não existe; nada para liquidar.")
        atomic_json(state, STATE_PATH)
        return 0
    before = pd.read_csv(HISTORY_PATH, low_memory=False)
    due = pending_paper(before, config, now, state)
    if due.empty:
        print("Nenhuma entrada paper atingiu o horário de settlement.")
        atomic_json(state, STATE_PATH)
        return 0

    print(f"Entradas paper prontas para settlement: {len(due)}")
    command_errors: list[str] = []
    for match_date in sorted(due["data"].fillna("").astype(str).unique()):
        if match_date:
            code = run([
                sys.executable, str(BASE_DIR / "05_settle_historico.py"),
                "--date", match_date, "--skip-post-update",
            ])
            if code != 0:
                command_errors.append(f"base:{match_date}:exit_{code}")

    after_base = pd.read_csv(HISTORY_PATH, low_memory=False)
    remaining_indices = [
        int(index) for index in due.index
        if index in after_base.index
        and str(after_base.at[index, "resultado"]).strip().lower() == "pendente"
    ]
    if remaining_indices:
        code = run([
            sys.executable, str(BASE_DIR / "05_settle_flashscore.py"),
            "--apply", "--refresh-cache", "--skip-rebuild-bank",
            "--history-indices", ",".join(map(str, remaining_indices)),
        ])
        if code != 0:
            command_errors.append(f"flashscore:exit_{code}")

    after = pd.read_csv(HISTORY_PATH, low_memory=False)
    resolved = 0
    retry_delays = [int(value) for value in config.get("settlement_retry_minutes", [30, 60, 180, 360])]
    for index, row in due.iterrows():
        key = row["_bet_key"]
        entry = state["bets"].setdefault(key, {})
        attempts = int(entry.get("attempts", 0)) + 1
        entry.update({
            "history_index": int(index),
            "attempts": attempts,
            "last_attempt_at": now.isoformat(timespec="seconds"),
        })
        current_result = (
            str(after.at[index, "resultado"]).strip().lower()
            if index in after.index and "resultado" in after.columns else "pendente"
        )
        if current_result in {"ganhou", "perdeu"}:
            entry.update({"status": "resolved", "resolved_at": now.isoformat(timespec="seconds"), "next_attempt_at": None})
            resolved += 1
        else:
            delay = retry_delays[min(attempts - 1, len(retry_delays) - 1)]
            entry.update({
                "status": "pending_retry",
                "next_attempt_at": (now + timedelta(minutes=delay)).isoformat(timespec="seconds"),
            })
    state["last_result"] = {
        "due": int(len(due)),
        "resolved": resolved,
        "pending_retry": int(len(due) - resolved),
        "command_errors": command_errors,
    }
    atomic_json(state, STATE_PATH)

    if resolved:
        run([sys.executable, str(BASE_DIR / "04_banca.py"), "--rebuild-bank"])
        run([sys.executable, str(BASE_DIR / "paper_monitor.py")])
    print(f"Resolvidas: {resolved} | aguardando nova tentativa: {len(due) - resolved}")
    if command_errors:
        print("Falhas de comando: " + ", ".join(command_errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
