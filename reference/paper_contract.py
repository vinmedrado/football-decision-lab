"""Contratos públicos de referência do ciclo de paper trading.

Este módulo ilustra regras de domínio sem acessar dados, modelos ou credenciais
do ambiente operacional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class Mode(str, Enum):
    PAPER_ONLY = "PAPER_ONLY"


@dataclass(frozen=True)
class PaperPolicy:
    cycle_id: str
    stake_fixed: float = 5.0
    max_bets_per_day: int = 10
    capture_min_minutes: int = 30
    capture_max_minutes: int = 90
    settlement_after_minutes: int = 130
    settlement_retries_minutes: tuple[int, ...] = (30, 60, 180, 360)
    mode: Mode = Mode.PAPER_ONLY
    allow_real_bets: bool = False

    def __post_init__(self) -> None:
        if self.allow_real_bets:
            raise ValueError("A política pública aceita somente PAPER_ONLY.")
        if self.capture_min_minutes >= self.capture_max_minutes:
            raise ValueError("A janela de captura é inválida.")
        if self.stake_fixed <= 0 or self.max_bets_per_day <= 0:
            raise ValueError("Stake e limite diário devem ser positivos.")

    def capture_window(self, kickoff: datetime) -> tuple[datetime, datetime]:
        """Retorna início e fim da janela válida de captura pré-jogo."""
        return (
            kickoff - timedelta(minutes=self.capture_max_minutes),
            kickoff - timedelta(minutes=self.capture_min_minutes),
        )

    def settlement_schedule(self, kickoff: datetime) -> tuple[datetime, ...]:
        """Gera tentativas progressivas sem alterar a previsão original."""
        first = kickoff + timedelta(minutes=self.settlement_after_minutes)
        attempts = [first]
        current = first
        for retry in self.settlement_retries_minutes:
            current += timedelta(minutes=retry)
            attempts.append(current)
        return tuple(attempts)


@dataclass(frozen=True)
class PaperPrediction:
    prediction_id: str
    captured_at: datetime
    kickoff: datetime
    market: str
    probability: float
    odd: float
    model_sha256: str

    def validate(self, policy: PaperPolicy) -> None:
        window_start, window_end = policy.capture_window(self.kickoff)
        if not window_start <= self.captured_at <= window_end:
            raise ValueError("Previsão fora da janela pré-jogo.")
        if not 0 <= self.probability <= 1:
            raise ValueError("Probabilidade inválida.")
        if self.odd <= 1:
            raise ValueError("Odd inválida.")
        if len(self.model_sha256) != 64:
            raise ValueError("Hash do modelo inválido.")
