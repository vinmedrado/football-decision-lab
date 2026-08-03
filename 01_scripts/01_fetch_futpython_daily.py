from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from futpython_client import FutPythonClient, save_text


DAILY_DIR = Path(os.getenv("FUTPYTHON_DAILY_DIR", "data/raw/futpython/daily"))
FLASH_2025_DIR = DAILY_DIR / "FlashScore_2025"

CABECALHO_MINIMO = "Date,Time,League,Round,Home,Away\n"


def _tem_csv_valido(content: str | None) -> bool:
    if not content or not str(content).strip():
        return False

    primeira_linha = str(content).splitlines()[0].strip().lower()

    colunas_obrigatorias = ["date", "time", "league", "home", "away"]

    return all(coluna in primeira_linha for coluna in colunas_obrigatorias)


def _arquivo_flashscore_2025(data: str) -> Path:
    return FLASH_2025_DIR / f"Jogos_do_Dia_FlashScore_{data}.csv"


def _usar_flashscore_2025(data: str, destino: Path) -> bool:
    origem = _arquivo_flashscore_2025(data)

    if not origem.exists():
        return False

    if origem.stat().st_size == 0:
        return False

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origem, destino)

    print(f"Jogos do dia carregados do histórico FlashScore 2025: {origem}")
    print(f"Arquivo salvo em formato padrão diário: {destino}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa jogos do dia da FutPython em CSV.")
    parser.add_argument("--date", required=True, help="Data no formato YYYY-MM-DD")
    parser.add_argument("--output-dir", default=str(DAILY_DIR))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--usar-flashscore-2025",
        action="store_true",
        help="Usa o histórico FlashScore 2025 como fallback explícito para backfill/auditoria histórica.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    path = output_dir / f"jogos_do_dia_{args.date}.csv"

    client = FutPythonClient(timeout=args.timeout)
    content = client.jogos_do_dia_csv(args.date)

    if _tem_csv_valido(content):
        save_text(path, content)
        print(f"Jogos do dia salvos via API FutPython em: {path}")
        return

    if args.usar_flashscore_2025 and args.date.startswith("2025-"):
        if _usar_flashscore_2025(args.date, path):
            return

    save_text(path, CABECALHO_MINIMO)
    print(f"Nenhum jogo encontrado para {args.date}. CSV vazio com cabeçalho salvo: {path}")


if __name__ == "__main__":
    main()
