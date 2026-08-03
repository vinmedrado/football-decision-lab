from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CAMINHO_DADOS = Path(os.getenv("FUTPYTHON_RAW_DIR", "data/raw/futpython/leagues"))
EXTENSOES = {".csv", ".xlsx", ".xls"}


def ler_amostra(arquivo: Path) -> pd.DataFrame:
    if arquivo.suffix.lower() == ".csv":
        return pd.read_csv(arquivo, nrows=5)
    return pd.read_excel(arquivo, nrows=5)


def validar_arquivos(caminho_base: Path):
    total_encontrados = 0
    total_lidos = 0
    arquivos_com_erro = []

    if not caminho_base.exists():
        return 0, 0, [(str(caminho_base), "Pasta de dados ainda não existe. Baixe dados com 01_fetch_futpython_ligas.py")]

    for arquivo in caminho_base.rglob("*"):
        if arquivo.suffix.lower() in EXTENSOES:
            total_encontrados += 1
            try:
                ler_amostra(arquivo)
                total_lidos += 1
            except Exception as erro:
                arquivos_com_erro.append((arquivo.name, str(erro)))

    return total_encontrados, total_lidos, arquivos_com_erro


def main():
    total_encontrados, total_lidos, arquivos_com_erro = validar_arquivos(CAMINHO_DADOS)

    print("\nResumo da validação")
    print("-------------------")
    print(f"Pasta analisada: {CAMINHO_DADOS}")
    print(f"Arquivos encontrados: {total_encontrados}")
    print(f"Arquivos lidos com sucesso: {total_lidos}")
    print(f"Arquivos com erro: {len(arquivos_com_erro)}")

    if arquivos_com_erro:
        print("\nArquivos com erro:")
        for nome, erro in arquivos_com_erro:
            print(f"- {nome}: {erro}")

    if total_encontrados == 0:
        print("\nNenhum arquivo encontrado. Use --fetch no pipeline ou rode 01_fetch_futpython_ligas.py.")
    elif total_encontrados == total_lidos:
        print("\nTodos os arquivos foram lidos com sucesso.")
    else:
        print("\nArquivos com erro.")


if __name__ == "__main__":
    main()
