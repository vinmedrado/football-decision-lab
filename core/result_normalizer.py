#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize settlement/result values to WIN, LOSS or PENDING."""
from __future__ import annotations

import unicodedata
from typing import Any

WIN_VALUES = {"ganhou", "ganha", "win", "green", "acertou", "acerto", "vitoria", "vitória", "won", "1", "true"}
LOSS_VALUES = {"perdeu", "perdida", "loss", "red", "errou", "erro", "derrota", "lost", "0", "false"}
PENDING_VALUES = {"pendente", "pending", "aguardando", "aberta", "open", "em_aberto", "nan", "none", ""}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(" ", "_")


def normalize_result(value: Any) -> str:
    cleaned = _clean(value)
    if cleaned in WIN_VALUES:
        return "WIN"
    if cleaned in LOSS_VALUES:
        return "LOSS"
    if cleaned in PENDING_VALUES:
        return "PENDING"
    return "PENDING"


if __name__ == "__main__":
    samples = ["ganhou", "perdeu", "pendente", "win", "red", "aguardando"]
    for item in samples:
        print(f"{item} -> {normalize_result(item)}")
