#!/usr/bin/env python3
"""Controlador único e idempotente das rotinas do Football Decision Lab."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "04_ml" / "config" / "automation_schedule.json"
STATE_PATH = ROOT / "04_ml" / "reports" / "automation_state.json"
LOCK_PATH = ROOT / "04_ml" / "reports" / "automation.lock"
LOG_DIR = ROOT / "logs" / "automation"


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


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class ProcessLock:
    def __init__(self, stale_minutes: int):
        self.stale_minutes = stale_minutes
        self.acquired = False

    def __enter__(self):
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOCK_PATH.exists():
            existing = load_json(LOCK_PATH, {})
            existing_pid = int(existing.get("pid") or 0)
            age_minutes = (time.time() - LOCK_PATH.stat().st_mtime) / 60
            if pid_alive(existing_pid) or age_minutes < self.stale_minutes:
                raise RuntimeError(f"Outro controlador está ativo (PID {existing_pid}).")
            LOCK_PATH.unlink()
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat()}, stream)
        self.acquired = True
        return self

    def __exit__(self, *_args):
        if self.acquired and LOCK_PATH.exists():
            current = load_json(LOCK_PATH, {})
            if int(current.get("pid") or 0) == os.getpid():
                LOCK_PATH.unlink()


def parse_clock(day: date, clock: str, timezone: ZoneInfo) -> datetime:
    hour, minute = [int(value) for value in clock.split(":", 1)]
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone)


def latest_weekday(now: datetime, weekday: int, clock: str) -> datetime:
    target_date = now.date() - timedelta(days=(now.weekday() - int(weekday)) % 7)
    candidate = parse_clock(target_date, clock, now.tzinfo)
    if candidate > now:
        candidate -= timedelta(days=7)
    return candidate


def latest_monthly(now: datetime, day: int, clock: str) -> datetime:
    safe_day = min(int(day), 28)
    candidate = parse_clock(date(now.year, now.month, safe_day), clock, now.tzinfo)
    if candidate <= now:
        return candidate
    previous_last = date(now.year, now.month, 1) - timedelta(days=1)
    return parse_clock(date(previous_last.year, previous_last.month, safe_day), clock, now.tzinfo)


def slots_for_job(name: str, cfg: dict, now: datetime) -> list[tuple[str, datetime]]:
    frequency = cfg.get("frequency")
    clocks = cfg.get("at", [])
    if isinstance(clocks, str):
        clocks = [clocks]
    slots = []
    if frequency == "interval":
        every_minutes = max(1, int(cfg.get("every_minutes", 15)))
        midnight = datetime(now.year, now.month, now.day, tzinfo=now.tzinfo)
        elapsed_minutes = int((now - midnight).total_seconds() // 60)
        slot_minutes = (elapsed_minutes // every_minutes) * every_minutes
        scheduled = midnight + timedelta(minutes=slot_minutes)
        slots.append((scheduled.strftime("%Y-%m-%d@%H:%M"), scheduled))
    elif frequency == "daily":
        for clock in clocks:
            scheduled = parse_clock(now.date(), clock, now.tzinfo)
            slots.append((f"{now.date().isoformat()}@{clock}", scheduled))
    elif frequency == "weekly":
        scheduled = latest_weekday(now, int(cfg["weekday"]), str(clocks[0]))
        slots.append((f"{scheduled.date().isoformat()}@{clocks[0]}", scheduled))
    elif frequency == "monthly":
        scheduled = latest_monthly(now, int(cfg["day"]), str(clocks[0]))
        slots.append((f"{scheduled:%Y-%m}@{clocks[0]}", scheduled))
    return slots


def job_commands(name: str, now: datetime) -> tuple[list[list[str]], dict[str, str]]:
    py = sys.executable
    today = now.date().isoformat()
    env: dict[str, str] = {}
    if name == "daily_refresh":
        return [
            [py, str(ROOT / "01_scripts" / "01_fetch_futpython_daily.py"), "--date", today],
        ], env
    if name == "paper_scan":
        return [
            [py, str(ROOT / "04_ml" / "paper_predict.py"), "--date", today],
            [py, str(ROOT / "04_ml" / "06_registrar_paper.py"), "--date", today],
            [py, str(ROOT / "04_ml" / "paper_monitor.py")],
        ], env
    if name == "dynamic_settlement":
        return [[py, str(ROOT / "04_ml" / "dynamic_settlement.py")]], env
    if name == "settlement":
        return [
            [py, str(ROOT / "04_ml" / "05_settle_historico.py"), "--skip-post-update"],
            [py, str(ROOT / "04_ml" / "05_settle_flashscore.py"), "--apply", "--refresh-cache", "--skip-rebuild-bank"],
            [py, str(ROOT / "04_ml" / "04_banca.py"), "--rebuild-bank"],
            [py, str(ROOT / "04_ml" / "paper_monitor.py")],
        ], env
    if name == "paper_monitor":
        return [
            [py, str(ROOT / "04_ml" / "paper_monitor.py")],
            [py, str(ROOT / "04_ml" / "paper_alerts.py")],
            [py, str(ROOT / "04_ml" / "07_banca_dashboard.py")],
            [py, str(ROOT / "04_ml" / "11_observability_report.py")],
            [py, str(ROOT / "04_ml" / "paper_model_manager.py"), "validate"],
        ], env
    if name == "base_refresh":
        return [[
            py, str(ROOT / "01_scripts" / "run_pipeline.py"),
            "--fetch", "--fetch-all", "--incremental",
        ]], env
    if name == "challenger_review":
        run_id = now.strftime("%Y-%m")
        return [[
            py, str(ROOT / "04_ml" / "challenger_runner.py"), "--run-id", run_id,
        ]], env
    raise KeyError(name)


def execute(name: str, slot: str, commands: list[list[str]], extra_env: dict[str, str], state: dict, now: datetime) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{now:%Y-%m-%d}_{name}_{slot.replace(':', '-')}.log"
    job_state = state.setdefault("jobs", {}).setdefault(name, {})
    slot_state = job_state.setdefault("slots", {}).setdefault(slot, {})
    slot_state.update({
        "status": "running",
        "attempts": int(slot_state.get("attempts", 0)) + 1,
        "started_at": datetime.now(now.tzinfo).isoformat(timespec="seconds"),
        "log": str(log_path.relative_to(ROOT)),
        "error": None,
    })
    state.update({"running": True, "current_job": name, "current_slot": slot, "heartbeat_at": datetime.now(now.tzinfo).isoformat(timespec="seconds")})
    atomic_json(state, STATE_PATH)
    environment = os.environ.copy()
    environment.update(extra_env)
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                log.write(f"\n$ {' '.join(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    text=True,
                )
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"{Path(command[1]).name} terminou com código {return_code}")
        slot_state.update({"status": "success", "finished_at": datetime.now(now.tzinfo).isoformat(timespec="seconds")})
        return True
    except Exception as exc:
        slot_state.update({
            "status": "failed",
            "finished_at": datetime.now(now.tzinfo).isoformat(timespec="seconds"),
            "error": str(exc),
        })
        return False
    finally:
        state.update({"running": False, "current_job": None, "current_slot": None, "heartbeat_at": datetime.now(now.tzinfo).isoformat(timespec="seconds")})
        atomic_json(state, STATE_PATH)


def due_jobs(config: dict, state: dict, now: datetime) -> list[tuple[str, str, datetime]]:
    retry_minutes = int(config.get("retry_after_minutes", 45))
    due = []
    for name, cfg in config.get("jobs", {}).items():
        if not cfg.get("enabled"):
            continue
        not_before = cfg.get("not_before")
        if not_before:
            try:
                if now.date() < date.fromisoformat(str(not_before)):
                    continue
            except ValueError:
                continue
        for slot, scheduled in slots_for_job(name, cfg, now):
            if scheduled > now:
                continue
            slot_state = state.get("jobs", {}).get(name, {}).get("slots", {}).get(slot, {})
            if slot_state.get("status") == "success":
                continue
            max_attempts = int(cfg.get("max_attempts_per_day", cfg.get("max_attempts_per_slot", 1)))
            if int(slot_state.get("attempts", 0)) >= max_attempts:
                continue
            finished = slot_state.get("finished_at")
            if finished:
                try:
                    if datetime.fromisoformat(finished) + timedelta(minutes=retry_minutes) > now:
                        continue
                except ValueError:
                    pass
            due.append((name, slot, scheduled))
    return sorted(due, key=lambda item: item[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-due", action="store_true")
    parser.add_argument("--job", choices=[
        "daily_refresh", "paper_scan", "dynamic_settlement", "settlement",
        "paper_monitor", "base_refresh", "challenger_review",
    ])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json(CONFIG_PATH, {})
    timezone = ZoneInfo(str(config.get("timezone", "America/Sao_Paulo")))
    now = datetime.now(timezone)
    state = load_json(STATE_PATH, {"jobs": {}, "running": False})

    if args.status or (not args.run_due and not args.job):
        print(json.dumps({"now": now.isoformat(), "due": [(a, b, c.isoformat()) for a, b, c in due_jobs(config, state, now)], "state": state}, indent=2, ensure_ascii=False))
        return 0

    requested = []
    if args.job:
        requested = [(args.job, f"manual@{now:%Y-%m-%dT%H-%M-%S}", now)]
    elif args.run_due:
        requested = due_jobs(config, state, now)
    if args.dry_run:
        print(json.dumps([
            {"job": name, "slot": slot, "scheduled_at": scheduled.isoformat(), "commands": job_commands(name, now)[0]}
            for name, slot, scheduled in requested
        ], indent=2, ensure_ascii=False))
        return 0
    if not config.get("enabled"):
        print("Automação desativada em automation_schedule.json.")
        return 0
    if not requested:
        state["heartbeat_at"] = now.isoformat(timespec="seconds")
        atomic_json(state, STATE_PATH)
        print("Nenhuma rotina pendente.")
        return 0

    success = True
    try:
        with ProcessLock(int(config.get("lock_stale_after_minutes", 240))):
            for name, slot, _scheduled in requested:
                commands, extra_env = job_commands(name, now)
                success = execute(name, slot, commands, extra_env, state, now) and success
    except RuntimeError as exc:
        print(str(exc))
        return 0
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
