from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

try:
    from dotenv import load_dotenv
except Exception:  # python-dotenv é opcional
    load_dotenv = None


if load_dotenv:
    load_dotenv()


DEFAULT_BASE_URL = "https://futpythontrader.com.br/api"


class FutPythonClient:
    """Cliente simples para a API FutPython Trader.

    Endpoints usados:
    - /jogos-do-dia?date=YYYY-MM-DD&format=csv&api_key=...
    - /download/PAIS/LIGA/TEMPORADA?api_key=...
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("FUTPYTHON_API_KEY")
        self.base_url = (base_url or os.getenv("FUTPYTHON_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def _require_key(self) -> str:
        if not self.api_key or self.api_key.strip() in {"", "SUA_API_KEY", "your_api_key_here"}:
            raise RuntimeError(
                "FUTPYTHON_API_KEY não configurada. Crie um .env na raiz com: "
                "FUTPYTHON_API_KEY=sua_chave_real"
            )
        return self.api_key.strip()

    @staticmethod
    def _q(value: object) -> str:
        # safe='' codifica espaço, acento e barra para URL funcionar com nomes de ligas/países.
        return quote(str(value).strip(), safe="")

    def build_download_url(self, country: str, liga: str, season: str) -> str:
        return f"{self.base_url}/download/{self._q(country)}/{self._q(liga)}/{self._q(season)}"

    def download_liga_csv(self, country: str, liga: str, season: str) -> str:
        url = self.build_download_url(country, liga, season)
        response = requests.get(url, params={"api_key": self._require_key()}, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def jogos_do_dia_csv(self, date: str) -> str:
        url = f"{self.base_url}/jogos-do-dia"
        params = {"date": date, "format": "csv", "api_key": self._require_key()}
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.text


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
