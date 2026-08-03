#!/usr/bin/env python3
"""Executa um script Python sem console, preservando stdout/stderr em arquivo."""
from __future__ import annotations

import os
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "headless"
MAX_LOG_BYTES = 2 * 1024 * 1024


def safe_target(raw: str) -> Path:
    target = Path(raw).resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("O script solicitado está fora do projeto.") from exc
    if not target.is_file() or target.suffix.lower() != ".py":
        raise SystemExit(f"Script não encontrado: {target}")
    return target


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("Uso: run_headless.py <nome-log> <script.py> [argumentos]")

    log_name = "".join(char for char in sys.argv[1] if char.isalnum() or char in {"-", "_"}) or "process"
    target = safe_target(sys.argv[2])
    target_args = sys.argv[3:]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    if log_path.exists() and log_path.stat().st_size > MAX_LOG_BYTES:
        backup = log_path.with_suffix(".previous.log")
        if backup.exists():
            backup.unlink()
        os.replace(log_path, backup)

    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            print(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] Iniciando {target.name}")
            os.chdir(ROOT)
            sys.argv = [str(target), *target_args]
            runpy.run_path(str(target), run_name="__main__")
            return 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if code:
                print(f"Encerrado com código {code}: {exc}")
            return code
        except BaseException:
            traceback.print_exc(file=stream)
            return 1
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
