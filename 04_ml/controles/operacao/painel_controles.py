#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera painel executivo dos controles operacionais."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controles.operacao.controle_operacional import evaluate_operational_guard
from core.project_paths import REPORTS_DIR, now_iso, read_json


def badge(status):
    s = str(status or 'DESCONHECIDA').upper()
    if s in {'OK','HEALTHY','EXCELLENT'}:
        return f'<span class="ok">🟢 {html.escape(s)}</span>'
    if s in {'ALERTA','DESCONHECIDA'}:
        return f'<span class="warn">🟡 {html.escape(s)}</span>'
    return f'<span class="block">🔴 {html.escape(s)}</span>'


def main():
    status = evaluate_operational_guard()
    env = read_json(REPORTS_DIR / 'status_ambiente.json', {}) or {}
    diag = read_json(REPORTS_DIR / 'diagnostico_governanca.json', {}) or {}
    health = read_json(REPORTS_DIR / 'saude_operacional.json', {}) or {}
    breakdown = read_json(REPORTS_DIR / 'detalhamento_pontuacao_saude.json', {}) or {}
    roi_summary = read_json(REPORTS_DIR / 'resumo_consistencia_roi.json', {}) or {}
    explain = read_json(REPORTS_DIR / 'explicabilidade_governanca.json', {}) or {}
    env_explain = read_json(REPORTS_DIR / 'explicabilidade_ambiente.json', {}) or {}
    executive = read_json(REPORTS_DIR / 'relatorio_executivo_governanca.json', {}) or {}
    recovery = read_json(REPORTS_DIR / 'relatorio_recuperacao_calibracao.json', {}) or {}
    best_cal = read_json(REPORTS_DIR / 'melhor_calibrador_por_mercado.json', {}) or {}
    mercado = read_json(REPORTS_DIR / 'relatorio_governanca_mercados.json', {}) or {}
    perf = read_json(REPORTS_DIR / 'diagnostico_fonte_performance.json', {}) or {}
    rows = [
        ('Status Operacional', status.get('status')),
        ('Operational Health', f"{health.get('score', 0)} — {health.get('label', 'DESCONHECIDA')}"),
        ('Health Breakdown', f"{breakdown.get('final_score', health.get('score', 0))} — {breakdown.get('label', health.get('label', 'DESCONHECIDA'))}"),
        ('Status do Ambiente', env.get('status')),
        ('Environment Explainability', env_explain.get('status')),
        ('Diagnóstico de Governança', diag.get('mercado_governance')),
        ('Governança Explainability', explain.get('severity')),
        ('ROI Consistency', roi_summary.get('status')),
        ('Fonte de Performance', mercado.get('performance_source', perf.get('source'))),
        ('Status da Governança de Mercados', mercado.get('status')),
        ('Status da Calibração', status.get('guard_calibracao')),
        ('Status da Simulação', status.get('guard_simulacao')),
        ('Status da Deriva', status.get('saude_features')),
        ('Status de Exposição', status.get('exposure_status')),
        ('Model Registry', status.get('model_registry')),
        ('Calibration Recovery', recovery.get('status')),
    ]
    alertas = status.get('alertas', []) + diag.get('alertas', []) + env_explain.get('issues', [])
    blocked = status.get('mercados_bloqueados', [])
    allowed = status.get('mercados_permitidos', [])
    body_rows = '\n'.join(f'<tr><td>{html.escape(k)}</td><td>{badge(v)}</td></tr>' for k,v in rows)
    deduction_rows = ''.join(
        f"<tr><td>{html.escape(str(d.get('motivo')))}</td><td>{html.escape(str(d.get('points')))}</td><td>{html.escape(str(d.get('source', '')))}</td></tr>"
        for d in breakdown.get('deductions', [])
    ) or '<tr><td colspan="3">Sem deduções</td></tr>'
    explanation_html = (
        f"<p><b>Motivo:</b> {html.escape(str(explain.get('motivo', 'OK')))}</p>"
        f"<p><b>Severidade:</b> {html.escape(str(explain.get('severity', 'OK')))}</p>"
        f"<p>{html.escape(str(explain.get('explanation', 'Sem bloqueios ativos.')))}</p>"
        f"<p><b>Ação recomendada:</b> {html.escape(str(explain.get('acao_recomendada', 'Manter monitoramento.')))}</p>"
    )
    roi_html = (
        f"<p><b>Status:</b> {html.escape(str(roi_summary.get('status', 'DESCONHECIDA')))}</p>"
        f"<p><b>Mercados checados:</b> {html.escape(str(roi_summary.get('checked_mercados', 0)))}</p>"
        f"<p><b>Alertas:</b> {html.escape(str(roi_summary.get('alertas', 0)))} | "
        f"<b>Critical:</b> {html.escape(str(roi_summary.get('critical', 0)))}</p>"
    )
    recovery_html = (
        f"<p><b>Status:</b> {html.escape(str(recovery.get('status', 'DESCONHECIDA')))}</p>"
        f"<p><b>Erro antes:</b> {html.escape(str((recovery.get('global_before') or {}).get('calibration_error')))}</p>"
        f"<p><b>Erro depois:</b> {html.escape(str((recovery.get('global_after') or {}).get('calibration_error')))}</p>"
        f"<p><b>Brier antes:</b> {html.escape(str((recovery.get('global_before') or {}).get('brier')))}</p>"
        f"<p><b>Brier depois:</b> {html.escape(str((recovery.get('global_after') or {}).get('brier')))}</p>"
        f"<p><b>Mercados recuperados:</b> {html.escape(str(recovery.get('mercados_recovered', 0)))}</p>"
        f"<p><b>Mercados ainda bloqueados/falhos:</b> {html.escape(str(recovery.get('mercados_failed', 0)))}</p>"
        f"<p><b>Melhores calibradores:</b> {html.escape(', '.join([str(m)+':'+str(v.get('best_method')) for m, v in best_cal.items()]) or 'nenhum')}</p>"
    )
    exec_html = ''.join(
        f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>'
        for k, v in executive.items() if k != 'alertas'
    )
    html_doc = f"""<!doctype html>
<html lang='pt-br'><head><meta charset='utf-8'><title>Football Lab — Governança Painel</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#111827;color:#e5e7eb}}.card{{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:18px;margin-bottom:16px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #374151;padding:10px;text-align:left}}.ok{{color:#22c55e}}.warn{{color:#facc15}}.block{{color:#ef4444}}code{{background:#0b1220;padding:2px 6px;border-radius:6px}}ul{{line-height:1.6}}
</style></head><body>
<h1>Football Lab — Executive Saúde do Sistema</h1>
<p>Gerado em <code>{now_iso()}</code></p>
<div class='card'><h2>Saúde do Sistema</h2><table><tbody>{body_rows}</tbody></table></div>
<div class='card'><h2>Health Detalhamento da Pontuação</h2><table><thead><tr><th>Reason</th><th>Points</th><th>Source</th></tr></thead><tbody>{deduction_rows}</tbody></table></div>
<div class='card'><h2>Governança Explainability</h2>{explanation_html}</div>
<div class='card'><h2>Calibration Recovery</h2>{recovery_html}</div>
<div class='card'><h2>ROI Consistency</h2>{roi_html}</div>
<div class='card'><h2>Executive Report</h2><table><tbody>{exec_html}</tbody></table></div>
<div class='card'><h2>Markets</h2><p><b>Bloqueados:</b> {html.escape(', '.join(blocked) or 'nenhum')}</p><p><b>Permitidos:</b> {html.escape(', '.join(allowed) or 'nenhum')}</p></div>
<div class='card'><h2>Alertas</h2><ul>{''.join('<li>'+html.escape(str(w))+'</li>' for w in alertas) or '<li>Sem alertas</li>'}</ul></div>
</body></html>"""
    out = REPORTS_DIR / 'painel_governanca.html'
    out.write_text(html_doc, encoding='utf-8')
    print(out)

if __name__ == '__main__':
    main()
