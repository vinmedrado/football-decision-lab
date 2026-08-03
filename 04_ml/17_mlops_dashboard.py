#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Painel HTML MLOps analítico."""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
DASHBOARD_DIR = BASE_DIR / "dashboard"
OUTPUT = DASHBOARD_DIR / "mlops_dashboard.html"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def card(title: str, value: Any, detail: str = "") -> str:
    return f"<div class='card'><div class='label'>{esc(title)}</div><div class='value'>{esc(value)}</div><div class='detail'>{esc(detail)}</div></div>"


def main() -> int:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    registry = read_json(REPORTS_DIR / "relatorio_registro_modelos.json", {})
    compare = read_json(REPORTS_DIR / "comparacao_modelo_base.json", {})
    features = read_json(REPORTS_DIR / "relatorio_saude_features.json", {})
    guard = read_json(REPORTS_DIR / "estado_guard_simulacao.json", {})

    models = registry.get("models", []) if isinstance(registry, dict) else []
    rows = "".join(
        f"<tr><td>{esc(m.get('mercado'))}</td><td>{esc(m.get('model_name'))}</td><td>{esc(m.get('status'))}</td><td>{esc(m.get('brier'))}</td><td>{esc(m.get('auc'))}</td><td>{esc(m.get('modified_at'))}</td></tr>"
        for m in models[:100]
    ) or "<tr><td colspan='6'>Nenhum modelo encontrado no pacote atual.</td></tr>"

    checks = compare.get("checks", []) if isinstance(compare, dict) else []
    checks_html = "".join(f"<li><b>{esc(c.get('status'))}</b> — {esc(c.get('check'))}: {esc(c.get('detail'))}</li>" for c in checks) or "<li>Sem checks disponíveis.</li>"

    html_doc = f"""<!doctype html>
<html lang='pt-BR'>
<head>
<meta charset='utf-8'>
<title>Football Lab — MLOps Painel</title>
<style>
body{{font-family:Arial, sans-serif;margin:24px;background:#f6f7f9;color:#1f2937}}
.container{{max-width:1180px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:18px 0}}
.card{{background:white;border-radius:14px;padding:16px;box-shadow:0 1px 8px rgba(0,0,0,.08)}}
.label{{font-size:12px;text-transform:uppercase;color:#6b7280}}
.value{{font-size:26px;font-weight:700;margin-top:6px}}
.detail{{font-size:12px;color:#6b7280;margin-top:6px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:14px;overflow:hidden;box-shadow:0 1px 8px rgba(0,0,0,.08)}}
th,td{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;font-size:13px}}
th{{background:#111827;color:white}}
.section{{margin-top:24px}}
.note{{font-size:13px;color:#6b7280}}
</style>
</head>
<body><div class='container'>
<h1>Football Lab — MLOps Painel</h1>
<p class='note'>Gerado em {esc(datetime.now().isoformat(timespec='seconds'))}. Painel analítico: não executa apostas nem recomenda entradas.</p>
<div class='grid'>
{card('Modelos publicáveis', registry.get('deployable_models', 0) if isinstance(registry, dict) else 0, 'Presentes no resumo ativo')}
{card('Status baseline', compare.get('status', 'NO_DATA') if isinstance(compare, dict) else 'NO_DATA', 'Comparação Brier/drift')}
{card('Brier baseline', compare.get('baseline_brier') if isinstance(compare, dict) else None, 'Menor é melhor')}
{card('Brier atual', compare.get('current_brier') if isinstance(compare, dict) else None, 'Via drift_summary')}
{card('Datasets avaliados', features.get('files_found', 0) if isinstance(features, dict) else 0, 'Saúde das features')}
{card('Guard analítico', guard.get('status', 'NO_DATA') if isinstance(guard, dict) else 'NO_DATA', 'Estado da fase 7')}
</div>
<div class='section'>
<h2>Checks de baseline</h2>
<ul>{checks_html}</ul>
</div>
<div class='section'>
<h2>Model Registry</h2>
<table><thead><tr><th>Mercado</th><th>Modelo</th><th>Status</th><th>Brier</th><th>AUC</th><th>Modificado em</th></tr></thead><tbody>{rows}</tbody></table>
</div>
</div></body></html>"""
    OUTPUT.write_text(html_doc, encoding="utf-8")
    print(f"Painel MLOps gerado em: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
