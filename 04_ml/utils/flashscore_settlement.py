#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coleta e validação de resultados no Flashscore para settlement.

O módulo usa apenas páginas públicas da versão mobile do Flashscore. O ID da
partida vem dos CSVs diários já salvos pelo projeto; não há busca por nomes na
internet nem uso de endpoints privados.
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests


FLASHSCORE_MOBILE_URL = "https://www.flashscore.mobi/match/{match_id}/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def normalize_team(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    replacements = {
        "united": "utd",
        "association": "",
        "football club": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b(fc|cf|sc|afc|ac|club|football|futebol)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def team_similarity(left: Any, right: Any) -> float:
    a = normalize_team(left)
    b = normalize_team(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        if shorter >= 4:
            return max(0.82, shorter / longer)
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class LocalMatchId:
    match_id: str
    source_file: str
    source_home: str
    source_away: str
    confidence: float


@dataclass(frozen=True)
class FlashscoreMatch:
    match_id: str
    source_url: str
    home: str
    away: str
    status: str
    played_at: str
    ft_home: int | None
    ft_away: int | None
    ht_home: int | None
    ht_away: int | None
    second_home: int | None
    second_away: int | None
    fetched_at: str

    @property
    def is_finished(self) -> bool:
        normalized = self.status.strip().lower()
        return normalized in {
            "finished",
            "full time",
            "after extra time",
            "after penalties",
        }

    @property
    def regulation_score(self) -> tuple[int, int] | None:
        if None not in (self.ht_home, self.ht_away, self.second_home, self.second_away):
            return (
                int(self.ht_home) + int(self.second_home),
                int(self.ht_away) + int(self.second_away),
            )
        if self.status.strip().lower() == "finished" and None not in (self.ft_home, self.ft_away):
            return int(self.ft_home), int(self.ft_away)
        return None

    @property
    def halftime_score(self) -> tuple[int, int] | None:
        if None in (self.ht_home, self.ht_away):
            return None
        return int(self.ht_home), int(self.ht_away)


def _clean_html_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _score_from_match(match: re.Match[str] | None) -> tuple[int, int] | None:
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_flashscore_html(page: str, match_id: str) -> FlashscoreMatch:
    """Extrai times, status, data e placares da página mobile de uma partida."""
    if not page or len(page) < 500:
        raise ValueError("Página vazia ou incompleta.")

    body_id = re.search(r"<body[^>]+data-match-id=['\"]([^'\"]+)['\"]", page, re.I)
    if body_id and body_id.group(1).strip() != str(match_id).strip():
        raise ValueError("O ID retornado pelo Flashscore difere do ID solicitado.")

    teams_match = re.search(
        r"<h3[^>]*>\s*<a[^>]*>(.*?)</a>\s*-\s*<a[^>]*>(.*?)</a>\s*</h3>",
        page,
        re.I | re.S,
    )
    if not teams_match:
        raise ValueError("Não foi possível identificar os dois times.")
    home = _clean_html_text(teams_match.group(1))
    away = _clean_html_text(teams_match.group(2))

    header_end_match = re.search(r"id=['\"]detail-tabs['\"]", page[teams_match.end() :], re.I)
    header_end = (
        teams_match.end() + header_end_match.start()
        if header_end_match
        else min(len(page), teams_match.end() + 5000)
    )
    header = page[teams_match.end() : header_end]
    details = re.findall(
        r"<div[^>]*class=['\"][^'\"]*\bdetail\b[^'\"]*['\"][^>]*>(.*?)</div>",
        header,
        re.I | re.S,
    )
    cleaned_details = [_clean_html_text(item) for item in details]
    ft_home: int | None = None
    ft_away: int | None = None
    status = ""
    played_at = ""
    score_position: int | None = None
    for position, detail in enumerate(cleaned_details):
        score = re.match(r"^(\d+)\s*-\s*(\d+)", detail)
        if score:
            ft_home, ft_away = int(score.group(1)), int(score.group(2))
            score_position = position
            break

    status_values = {
        "finished",
        "full time",
        "after extra time",
        "after penalties",
        "postponed",
        "cancelled",
        "canceled",
        "abandoned",
        "interrupted",
        "delayed",
        "awarded",
        "walkover",
        "scheduled",
        "not started",
        "half time",
        "break",
    }
    status_start = (score_position + 1) if score_position is not None else 0
    for detail in cleaned_details[status_start:]:
        if detail.strip().lower() in status_values:
            status = detail.strip()
            break

    date_pattern = re.compile(r"\b\d{2}\.\d{2}\.\d{4}(?:\s+\d{1,2}:\d{2})?\b")
    for detail in cleaned_details:
        date_match = date_pattern.search(detail)
        if date_match:
            played_at = date_match.group(0)
            break

    if not status and played_at and ft_home is None and ft_away is None:
        status = "Scheduled"

    if not status:
        raise ValueError("Não foi possível identificar o status da partida.")

    first_half = _score_from_match(
        re.search(
            r"<h4[^>]*>\s*1st\s+Half:\s*<b[^>]*>\s*(\d+)\s*-\s*(\d+)\s*</b>",
            page,
            re.I | re.S,
        )
    )
    second_half = _score_from_match(
        re.search(
            r"<h4[^>]*>\s*2nd\s+Half:\s*<b[^>]*>\s*(\d+)\s*-\s*(\d+)\s*</b>",
            page,
            re.I | re.S,
        )
    )

    return FlashscoreMatch(
        match_id=str(match_id).strip(),
        source_url=FLASHSCORE_MOBILE_URL.format(match_id=str(match_id).strip()),
        home=home,
        away=away,
        status=status,
        played_at=played_at,
        ft_home=ft_home,
        ft_away=ft_away,
        ht_home=first_half[0] if first_half else None,
        ht_away=first_half[1] if first_half else None,
        second_home=second_half[0] if second_half else None,
        second_away=second_half[1] if second_half else None,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def validate_match_teams(
    expected_home: Any,
    expected_away: Any,
    match: FlashscoreMatch,
    minimum_similarity: float = 0.72,
) -> tuple[bool, float]:
    home_score = team_similarity(expected_home, match.home)
    away_score = team_similarity(expected_away, match.away)
    confidence = round((home_score + away_score) / 2, 4)
    return home_score >= minimum_similarity and away_score >= minimum_similarity, confidence


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def find_local_match_id(
    raw_daily_dir: Path,
    original_date: Any,
    expected_home: Any,
    expected_away: Any,
    minimum_similarity: float = 0.72,
) -> LocalMatchId | None:
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", str(original_date or ""))
    if not date_match or not raw_daily_dir.exists():
        return None

    candidates = sorted(raw_daily_dir.glob(f"*{date_match.group(0)}*.csv"))
    possible: list[LocalMatchId] = []
    for path in candidates:
        try:
            frame = _read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if frame.empty:
            continue

        columns = {str(column).strip().lower(): column for column in frame.columns}
        id_column = next(
            (columns[name] for name in ("id", "flashscore_id", "match_id", "event_id") if name in columns),
            None,
        )
        home_column = next(
            (columns[name] for name in ("home", "home_std", "mandante", "home_team") if name in columns),
            None,
        )
        away_column = next(
            (columns[name] for name in ("away", "away_std", "visitante", "away_team") if name in columns),
            None,
        )
        if id_column is None or home_column is None or away_column is None:
            continue

        for _, row in frame.iterrows():
            match_id = str(row.get(id_column, "") or "").strip()
            source_home = str(row.get(home_column, "") or "").strip()
            source_away = str(row.get(away_column, "") or "").strip()
            if not match_id or match_id.lower() in {"nan", "none"}:
                continue
            home_score = team_similarity(expected_home, source_home)
            away_score = team_similarity(expected_away, source_away)
            confidence = (home_score + away_score) / 2
            if home_score >= minimum_similarity and away_score >= minimum_similarity:
                possible.append(
                    LocalMatchId(
                        match_id=match_id,
                        source_file=str(path),
                        source_home=source_home,
                        source_away=source_away,
                        confidence=round(confidence, 4),
                    )
                )

    if not possible:
        return None
    possible.sort(key=lambda item: item.confidence, reverse=True)
    if len(possible) > 1 and possible[0].confidence == possible[1].confidence:
        distinct_ids = {item.match_id for item in possible if item.confidence == possible[0].confidence}
        if len(distinct_ids) > 1:
            return None
    return possible[0]


class FlashscoreClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout: float = 20.0,
        max_retries: int = 2,
        nonfinal_cache_hours: float = 6.0,
        min_request_interval: float = 1.25,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.nonfinal_cache_hours = max(0.0, float(nonfinal_cache_hours))
        self.min_request_interval = max(0.0, float(min_request_interval))
        self._last_request_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.8",
            }
        )

    def _cache_path(self, match_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(match_id))
        return self.cache_dir / f"{safe_id}.json"

    def _read_cache(self, match_id: str) -> FlashscoreMatch | None:
        path = self._cache_path(match_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            match = FlashscoreMatch(**payload)
            fetched = datetime.fromisoformat(match.fetched_at.replace("Z", "+00:00"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

        if match.is_finished:
            if match.played_at and not re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", match.played_at):
                return None
            return match
        age_hours = (datetime.now(timezone.utc) - fetched.astimezone(timezone.utc)).total_seconds() / 3600
        return match if age_hours <= self.nonfinal_cache_hours else None

    def _write_cache(self, match: FlashscoreMatch) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(match.match_id)
        temporary = path.with_suffix(f".{time.time_ns()}.tmp")
        temporary.write_text(
            json.dumps(asdict(match), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def get_match(self, match_id: str, refresh: bool = False) -> tuple[FlashscoreMatch, bool]:
        if not refresh:
            cached = self._read_cache(match_id)
            if cached is not None:
                return cached, True

        url = FLASHSCORE_MOBILE_URL.format(match_id=str(match_id).strip())
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self._last_request_at is not None and self.min_request_interval > 0:
                    elapsed = time.monotonic() - self._last_request_at
                    if elapsed < self.min_request_interval:
                        time.sleep(self.min_request_interval - elapsed)
                self._last_request_at = time.monotonic()
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                match = parse_flashscore_html(response.text, str(match_id))
                self._write_cache(match)
                return match, False
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"Falha ao consultar {match_id}: {last_error}")
