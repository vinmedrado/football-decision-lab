#!/usr/bin/env python3
"""Executa backtest e treino mensal em diretórios versionados, sem promover modelo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "04_ml"
BACKTEST_RESULTS = ROOT / "03_backtest" / "results"


def run(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(command[1]).name} terminou com código {result.returncode}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_within(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if child_resolved == parent_resolved or parent_resolved not in child_resolved.parents:
        raise RuntimeError(f"Caminho fora do diretório versionado: {child_resolved}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y-%m"))
    args = parser.parse_args()
    run_root = ML_DIR / "challengers" / args.run_id
    dataset_dir = run_root / "datasets"
    model_dir = run_root / "models"
    backtest_snapshot = run_root / "backtest_results"
    report_path = run_root / "challenger_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "SUCCESS":
            print(f"Challenger {args.run_id} já concluído; nada foi sobrescrito.")
            return 0

    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "ML_BACKTEST_RESULTS_DIR": str(backtest_snapshot),
        "ML_DATASET_OUTPUT_DIR": str(dataset_dir),
        "ML_DATASET_INPUT_DIR": str(dataset_dir),
        "ML_MODEL_OUTPUT_DIR": str(model_dir),
    })
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        run([sys.executable, str(ROOT / "03_backtest" / "runner.py")], env)
        if backtest_snapshot.exists():
            ensure_within(backtest_snapshot, run_root)
            shutil.rmtree(backtest_snapshot)
        shutil.copytree(BACKTEST_RESULTS, backtest_snapshot)
        run([sys.executable, str(ML_DIR / "01_dataset_builder.py")], env)
        run([sys.executable, str(ML_DIR / "02_train_model.py")], env)
        summary = model_dir / "resumo_modelos.pkl"
        report = {
            "run_id": args.run_id,
            "status": "SUCCESS",
            "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "backtest_snapshot": str(backtest_snapshot.relative_to(ROOT)),
            "dataset_dir": str(dataset_dir.relative_to(ROOT)),
            "model_dir": str(model_dir.relative_to(ROOT)),
            "model_summary_sha256": hash_file(summary) if summary.exists() else None,
            "auto_promoted": False,
            "note": "Artefatos desafiantes isolados. Promoção exige revisão e novo ciclo paper.",
        }
    except Exception as exc:
        report = {
            "run_id": args.run_id,
            "status": "FAILED",
            "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": str(exc),
            "auto_promoted": False,
        }
        run_root.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    run_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = ML_DIR / "reports" / "latest_challenger.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
