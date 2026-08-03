#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

FETCH_SCRIPT = ROOT / "01_scripts" / "01_fetch_futpython_daily.py"
PREDICT_SCRIPT = ROOT / "04_ml" / "03_predict.py"


def run_cmd(cmd):
    print("\n" + "=" * 80)
    print("$ " + " ".join(map(str, cmd)))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def daterange(start, end):
    atual = start
    while atual <= end:
        yield atual
        atual += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD")
    parser.add_argument("--skip-fetch", action="store_true", help="Não baixa daily, só roda predict")
    parser.add_argument("--modo-auditoria", action="store_true", help="Gera previsões históricas sem operação real e com apostar=False")
    parser.add_argument(
        "--usar-flashscore-2025",
        action="store_true",
        help="Permite usar o histórico FlashScore 2025 no fetch durante backfill/auditoria histórica.",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if end < start:
        raise SystemExit("❌ Data final menor que data inicial.")

    total = 0
    ok = 0
    falhas = []

    print("\n🔁 BACKFILL DAILY FOOTBALL LAB")
    print(f"Período: {start} até {end}")

    for data in daterange(start, end):
        data_str = data.isoformat()
        total += 1

        print("\n" + "#" * 80)
        print(f"📅 Processando {data_str}")
        print("#" * 80)

        if not args.skip_fetch:
            fetch_cmd = [PYTHON, str(FETCH_SCRIPT), "--date", data_str]
            if args.usar_flashscore_2025:
                fetch_cmd.append("--usar-flashscore-2025")
            fetch_ok = run_cmd(fetch_cmd)
            if not fetch_ok:
                falhas.append((data_str, "fetch"))
                continue

        predict_cmd = [PYTHON, str(PREDICT_SCRIPT), "--date", data_str]
        if args.modo_auditoria:
            predict_cmd.append("--modo-auditoria")
        predict_ok = run_cmd(predict_cmd)
        if not predict_ok:
            falhas.append((data_str, "predict"))
            continue

        ok += 1

    print("\n" + "=" * 80)
    print("✅ BACKFILL FINALIZADO")
    print(f"Dias processados com sucesso: {ok}/{total}")

    if falhas:
        print("\n❌ Falhas:")
        for data_str, etapa in falhas:
            print(f" - {data_str}: {etapa}")
    print("=" * 80)


if __name__ == "__main__":
    main()