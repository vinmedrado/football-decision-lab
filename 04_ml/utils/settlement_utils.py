#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Funções únicas de settlement para todos os scripts de banca/auditoria."""

from __future__ import annotations

import math


def _has_nan(*values) -> bool:
    for value in values:
        try:
            if math.isnan(float(value)):
                return True
        except Exception:
            return True
    return False


def resolver_mercado(mercado, gh_ft, ga_ft, gh_ht=0, ga_ht=0):
    mercado = str(mercado).upper().strip()

    try:
        gh_ft = float(gh_ft)
        ga_ft = float(ga_ft)
        gh_ht = float(gh_ht)
        ga_ht = float(ga_ht)
    except Exception:
        return None

    if _has_nan(gh_ft, ga_ft, gh_ht, ga_ht):
        return None

    tg_ft = gh_ft + ga_ft
    tg_ht = gh_ht + ga_ht

    if mercado == "DC_12":
        return gh_ft != ga_ft

    if mercado == "BTTS_Y":
        return gh_ft > 0 and ga_ft > 0
    if mercado == "BTTS_N":
        return not (gh_ft > 0 and ga_ft > 0)

    if mercado in {"R_FT_H", "1X2_FT_1"}:
        return gh_ft > ga_ft
    if mercado in {"R_FT_D", "1X2_FT_X"}:
        return gh_ft == ga_ft
    if mercado in {"R_FT_A", "1X2_FT_2"}:
        return gh_ft < ga_ft

    if mercado in {"R_HT_H", "1X2_HT_1"}:
        return gh_ht > ga_ht
    if mercado in {"R_HT_D", "1X2_HT_X"}:
        return gh_ht == ga_ht
    if mercado in {"R_HT_A", "1X2_HT_2"}:
        return gh_ht < ga_ht

    if mercado.startswith("TG_HT_U"):
        return _resolve_total_line(mercado, "TG_HT_U", tg_ht, under=True)
    if mercado.startswith("TG_HT_O"):
        return _resolve_total_line(mercado, "TG_HT_O", tg_ht, under=False)
    if mercado.startswith("TG_FT_U"):
        return _resolve_total_line(mercado, "TG_FT_U", tg_ft, under=True)
    if mercado.startswith("TG_FT_O"):
        return _resolve_total_line(mercado, "TG_FT_O", tg_ft, under=False)

    return None


def _resolve_total_line(mercado: str, prefix: str, total: float, under: bool):
    try:
        linha = float(mercado.replace(prefix, "")) / 10
    except Exception:
        return None

    if _has_nan(total, linha):
        return None

    return total < linha if under else total > linha


def resolver_resultado_mercado(mercado, event, home_score, away_score, ht_home=0, ht_away=0):
    mercado_norm = str(mercado).upper().strip()
    event_norm = str(event).strip()

    direto = resolver_mercado(mercado_norm, home_score, away_score, ht_home, ht_away)
    if direto is not None:
        return direto

    try:
        hs = float(home_score)
        aw = float(away_score)
        hh = float(ht_home)
        ha = float(ht_away)
    except Exception:
        return None

    if _has_nan(hs, aw, hh, ha):
        return None

    if mercado_norm in {"TG_FT", "TG_HT"}:
        total = (hs + aw) if mercado_norm == "TG_FT" else (hh + ha)
        try:
            partes = event_norm.lower().replace(" ", "_").split("_")
            direcao = partes[0]
            linha = float(partes[-1])
        except Exception:
            return None

        if _has_nan(total, linha):
            return None

        if direcao == "over":
            return total > linha
        if direcao == "under":
            return total < linha
        return None

    if mercado_norm == "1X2_FT":
        resultado = "1" if hs > aw else ("X" if hs == aw else "2")
        return resultado == event_norm.upper()

    if mercado_norm == "1X2_HT":
        resultado = "1" if hh > ha else ("X" if hh == ha else "2")
        return resultado == event_norm.upper()

    if mercado_norm == "BTTS_FT":
        esperado = event_norm.upper() in {"Y", "YES", "SIM", "BTTS_Y"}
        realizado = hs >= 1 and aw >= 1
        return realizado if esperado else not realizado

    if mercado_norm == "BTTS_HT":
        esperado = event_norm.upper() in {"Y", "YES", "SIM", "BTTS_Y"}
        realizado = hh >= 1 and ha >= 1
        return realizado if esperado else not realizado

    return None