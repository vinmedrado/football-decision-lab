#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera painel executivo SYSTEM HEALTH para a FASE 12."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "04_ml"
REPORTS_DIR = ML_DIR / "reports"
DASHBOARD_PATH = REPORTS_DIR / "painel_saude_operacional.html"


def read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default if default is not None else {}


def badge(status: str) -> str:
    s = str(status or "DESCONHECIDA").upper()
    if s in {"OK", "PASS", "HEALTHY"}:
        icon = "🟢"
    elif s in {"ALERTA", "WARN", "DESCONHECIDA"}:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon} {s}"


def main() -> None:
    operational = read_json(REPORTS_DIR / "status_operacional.json", {})
    exposure = read_json(REPORTS_DIR / "guard_exposicao.json", {})
    mercado = read_json(REPORTS_DIR / "relatorio_governanca_mercados.json", {})

    rows = [
        ("MODEL STATUS", operational.get("model_registry", "DESCONHECIDA")),
        ("CALIBRATION STATUS", operational.get("guard_calibracao", "DESCONHECIDA")),
        ("SIMULATION STATUS", operational.get("guard_simulacao", "DESCONHECIDA")),
        ("DRIFT STATUS", operational.get("saude_features", "DESCONHECIDA")),
        ("EXPOSURE STATUS", exposure.get("status", operational.get("exposure_status", "DESCONHECIDA"))),
        ("MARKET GOVERNANCE STATUS", mercado.get("status", operational.get("mercado_governance_status", "DESCONHECIDA"))),
        ("OPERATIONAL STATUS", operational.get("status", "DESCONHECIDA")),
    ]

    mercados_bloqueados = operational.get("mercados_bloqueados") or mercado.get("mercados_bloqueados") or []
    mercados_permitidos = operational.get("mercados_permitidos") or mercado.get("mercados_permitidos") or []
    alertas = operational.get("alertas") or mercado.get("alertas") or []

    html_rows = "\n".join(f"<tr><td>{name}</td><td>{badge(status)}</td></tr>" for name, status in rows)
    html = f"""<!doctype html>
<html lang=\"pt-BR\">
<head>
<meta charset=\"utf-8\" />
<title>Football Lab — Saúde do Sistema</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; background: #f7f7f7; color: #222; }}
.card {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ text-align: left; border-bottom: 1px solid #ddd; padding: 10px; }}
th {{ background: #efefef; }}
code {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Football Lab — SYSTEM HEALTH</h1>
<div class=\"card\">
<p><strong>Gerado em:</strong> {datetime.now().isoformat(timespec='seconds')}</p>
<p><strong>Status operacional:</strong> {badge(operational.get('status', 'DESCONHECIDA'))}</p>
<p><strong>Motivo:</strong> <code>{operational.get('motivo')}</code></p>
<p><strong>Allow predictions:</strong> <code>{operational.get('permitir_previsoes')}</code></p>
</div>
<div class=\"card\">
<h2>Indicadores executivos</h2>
<table><thead><tr><th>Componente</th><th>Status</th></tr></thead><tbody>{html_rows}</tbody></table>
</div>
<div class=\"card\">
<h2>Model Registry</h2>
<p><strong>Active:</strong> <code>{operational.get('modelo_ativo')}</code></p>
<p><strong>Champion:</strong> <code>{operational.get('modelo_campeao')}</code></p>
<p><strong>Baseline:</strong> <code>{operational.get('modelo_base')}</code></p>
</div>
<div class=\"card\">
<h2>Mercados bloqueados</h2>
<p>{', '.join(map(str, mercados_bloqueados)) if mercados_bloqueados else 'Nenhum mercado bloqueado.'}</p>
<h2>Mercados permitidos</h2>
<p>{', '.join(map(str, mercados_permitidos)) if mercados_permitidos else 'Nenhum mercado permitido ou amostra insuficiente.'}</p>
</div>
<div class=\"card\">
<h2>Alertas</h2>
<ul>{''.join(f'<li>{w}</li>' for w in alertas) if alertas else '<li>Nenhum warning.</li>'}</ul>
</div>
</body>
</html>"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"Painel salvo em: {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
