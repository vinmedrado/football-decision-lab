#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 21 — Governança de Odds.

Camada analítica em português para medir a performance real por faixa de odds.
Este script não altera modelos, não altera banca, não remove guards e não libera
apostas. Todos os artefatos são de observabilidade/simulação.
"""
from __future__ import annotations

import csv
import html
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import REPORTS_DIR, ROOT_DIR, now_iso, write_json  # noqa: E402
from core.result_normalizer import normalize_result  # noqa: E402

CAMINHOS_HISTORICO = [
    ROOT_DIR / "04_ml" / "banca" / "historico_apostas.csv",
    ROOT_DIR / "04_ml" / "historico_apostas.csv",
]

RELATORIO_GOVERNANCA = REPORTS_DIR / "governanca_odds.json"
RANKING_ODDS = REPORTS_DIR / "classificacao_odds.json"
AUDITORIA_ODDS = REPORTS_DIR / "auditoria_odds.json"
FAIXAS_ODDS_REPROVADAS = REPORTS_DIR / "faixas_odds_reprovadas.json"
CICLO_VIDA_ODDS = REPORTS_DIR / "ciclo_vida_odds.json"
MATRIZ_ODDS_CONFIANCA = REPORTS_DIR / "matriz_odds_confianca.json"
MATRIZ_ODDS_MERCADO = REPORTS_DIR / "matriz_odds_mercado.json"
DASHBOARD_ODDS = REPORTS_DIR / "painel_governanca_odds.html"
RELATORIO_FASE = ROOT_DIR / "FASE21_GOVERNANCA_ODDS_REPORT.md"

AMOSTRA_MINIMA = 100

FAIXAS_ODDS = [
    (1.00, 1.31, "1.00_1.30", "1.00 – 1.30"),
    (1.31, 1.51, "1.31_1.50", "1.31 – 1.50"),
    (1.51, 1.71, "1.51_1.70", "1.51 – 1.70"),
    (1.71, 2.01, "1.71_2.00", "1.71 – 2.00"),
    (2.01, 2.51, "2.01_2.50", "2.01 – 2.50"),
    (2.51, 3.01, "2.51_3.00", "2.51 – 3.00"),
    (3.01, float("inf"), "3.01_mais", "3.01+"),
]

FAIXAS_CONFIANCA = [
    (0.50, 0.60, "50_60", "50% – 60%"),
    (0.60, 0.70, "60_70", "60% – 70%"),
    (0.70, 0.80, "70_80", "70% – 80%"),
    (0.80, 0.90, "80_90", "80% – 90%"),
    (0.90, 1.01, "90_100", "90% – 100%"),
]

FLAGS_SEGURAS = {
    "modo_simulacao": True,
    "aposta_real_habilitada": False,
    "recomendacoes_habilitadas": False,
    "modo_seguro": True,
    "escopo": "analise_educacional_e_simulacao",
}


def _float(valor: Any, padrao: float = 0.0) -> float:
    if valor is None:
        return padrao
    try:
        if isinstance(valor, str):
            texto = valor.strip().replace("R$", "").replace("%", "")
            if "," in texto and "." in texto:
                texto = texto.replace(".", "").replace(",", ".")
            elif "," in texto:
                texto = texto.replace(",", ".")
            valor = texto
        saida = float(valor)
        if math.isnan(saida) or math.isinf(saida):
            return padrao
        return saida
    except Exception:
        return padrao


def _encontrar_historico() -> Path | None:
    for caminho in CAMINHOS_HISTORICO:
        if caminho.exists():
            return caminho
    return None


def _ler_historico() -> Tuple[List[Dict[str, Any]], str]:
    caminho = _encontrar_historico()
    if caminho is None:
        return [], "NAO_ENCONTRADO"
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with caminho.open("r", encoding=encoding, newline="") as arquivo:
                return list(csv.DictReader(arquivo)), str(caminho.relative_to(ROOT_DIR))
        except Exception:
            continue
    return [], str(caminho.relative_to(ROOT_DIR))


def _faixa_odd(odd: float) -> Tuple[str, str] | Tuple[None, None]:
    for minimo, maximo, codigo, rotulo in FAIXAS_ODDS:
        if minimo <= odd < maximo:
            return codigo, rotulo
    return None, None


def _faixa_confianca(probabilidade: float) -> Tuple[str, str] | Tuple[None, None]:
    for minimo, maximo, codigo, rotulo in FAIXAS_CONFIANCA:
        if minimo <= probabilidade < maximo or (codigo == "90_100" and minimo <= probabilidade <= maximo):
            return codigo, rotulo
    return None, None


def _lucro_e_stake(linha: Dict[str, Any], resultado: str, odd: float) -> Tuple[float, float]:
    stake = _float(linha.get("valor_apostado") or linha.get("stake") or linha.get("valor"), 1.0)
    lucro_informado = linha.get("lucro") or linha.get("profit") or linha.get("pnl")
    if lucro_informado not in (None, ""):
        return _float(lucro_informado), max(stake, 0.0)
    if resultado == "WIN":
        return stake * max(odd - 1.0, 0.0), max(stake, 0.0)
    if resultado == "LOSS":
        return -stake, max(stake, 0.0)
    return 0.0, max(stake, 0.0)


def _nova_agregacao() -> Dict[str, Any]:
    return {
        "quantidade_apostas": 0,
        "vitorias": 0,
        "derrotas": 0,
        "stake_total": 0.0,
        "lucro_real": 0.0,
        "odd_soma": 0.0,
        "lucros_sequenciais": [],
    }


def _drawdown(lucros: List[float], stake_total: float) -> float:
    acumulado = 0.0
    pico = 0.0
    pior = 0.0
    for lucro in lucros:
        acumulado += lucro
        pico = max(pico, acumulado)
        pior = max(pior, pico - acumulado)
    denominador = stake_total if stake_total else 1.0
    return pior / denominador


def _classificar(roi: float, quantidade: int) -> str:
    if quantidade <= 0:
        return "SEM_AMOSTRA"
    if quantidade < AMOSTRA_MINIMA:
        return "OBSERVACAO" if roi > 0 else "SEM_AMOSTRA"
    if roi > 0:
        return "ATIVA"
    if roi < 0:
        return "REPROVADA"
    return "OBSERVACAO"


def _motivo_status(status: str, roi: float, quantidade: int) -> str:
    if status == "ATIVA":
        return "ROI real positivo com amostra mínima atendida."
    if status == "OBSERVACAO":
        return "Faixa positiva, mas ainda sem amostra mínima para decisão definitiva."
    if status == "REPROVADA":
        return "ROI real negativo com amostra mínima atendida."
    if quantidade <= 0:
        return "Sem apostas liquidadas nesta faixa."
    return "Amostra insuficiente para classificação operacional."


def _finalizar_agregacao(agg: Dict[str, Any], rotulo: str) -> Dict[str, Any]:
    qtd = int(agg["quantidade_apostas"])
    stake = _float(agg["stake_total"])
    lucro = _float(agg["lucro_real"])
    vitorias = int(agg["vitorias"])
    taxa_acerto = vitorias / qtd if qtd else 0.0
    roi = lucro / stake if stake else 0.0
    odd_media = _float(agg["odd_soma"]) / qtd if qtd else 0.0
    status = _classificar(roi, qtd)
    return {
        "faixa": rotulo,
        "status": status,
        "quantidade_apostas": qtd,
        "taxa_acerto": round(taxa_acerto, 6),
        "roi_real": round(roi, 6),
        "lucro_real": round(lucro, 2),
        "stake_total": round(stake, 2),
        "drawdown": round(_drawdown(agg["lucros_sequenciais"], stake), 6),
        "odd_media": round(odd_media, 6),
        "elegivel_para_operacao": bool(status == "ATIVA"),
        "motivo_status": _motivo_status(status, roi, qtd),
    }


def _analisar_faixas(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    agregacoes = {codigo: _nova_agregacao() for _, _, codigo, _ in FAIXAS_ODDS}
    rotulos = {codigo: rotulo for _, _, codigo, rotulo in FAIXAS_ODDS}
    ignoradas = 0
    liquidadas = 0
    for linha in linhas:
        resultado = normalize_result(linha.get("resultado") or linha.get("result") or linha.get("status"))
        if resultado not in {"WIN", "LOSS"}:
            ignoradas += 1
            continue
        odd = _float(linha.get("odd") or linha.get("odds") or linha.get("cotacao"), 0.0)
        codigo, _ = _faixa_odd(odd)
        if not codigo:
            ignoradas += 1
            continue
        lucro, stake = _lucro_e_stake(linha, resultado, odd)
        agg = agregacoes[codigo]
        agg["quantidade_apostas"] += 1
        agg["vitorias"] += 1 if resultado == "WIN" else 0
        agg["derrotas"] += 1 if resultado == "LOSS" else 0
        agg["stake_total"] += stake
        agg["lucro_real"] += lucro
        agg["odd_soma"] += odd
        agg["lucros_sequenciais"].append(lucro)
        liquidadas += 1
    faixas = {codigo: _finalizar_agregacao(agg, rotulos[codigo]) for codigo, agg in agregacoes.items()}
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "fonte": "historico_apostas.csv",
        "linhas_liquidadas_avaliadas": liquidadas,
        "linhas_ignoradas": ignoradas,
        "amostra_minima": AMOSTRA_MINIMA,
        "faixas": faixas,
    }


def _classificacao(relatorio: Dict[str, Any]) -> Dict[str, Any]:
    faixas = list((relatorio.get("faixas") or {}).values())
    def ordenado(chave: str, reverso: bool = True) -> List[Dict[str, Any]]:
        return sorted(faixas, key=lambda x: (_float(x.get(chave)), int(x.get("quantidade_apostas") or 0)), reverse=reverso)
    def estabilidade(item: Dict[str, Any]) -> float:
        return _float(item.get("roi_real")) - _float(item.get("drawdown"))
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "melhor_roi": ordenado("roi_real"),
        "maior_lucro": ordenado("lucro_real"),
        "melhor_estabilidade": sorted(faixas, key=lambda x: (estabilidade(x), int(x.get("quantidade_apostas") or 0)), reverse=True),
        "menor_drawdown": sorted(faixas, key=lambda x: (_float(x.get("drawdown")), -int(x.get("quantidade_apostas") or 0))),
    }


def _faixas_reprovadas(relatorio: Dict[str, Any]) -> Dict[str, Any]:
    saida = {}
    for codigo, item in (relatorio.get("faixas") or {}).items():
        if item.get("status") == "REPROVADA":
            saida[codigo] = {
                "faixa": item.get("faixa"),
                "motivo": "ROI_NEGATIVO",
                "roi_real": item.get("roi_real"),
                "lucro_real": item.get("lucro_real"),
                "taxa_acerto": item.get("taxa_acerto"),
                "drawdown": item.get("drawdown"),
                "quantidade_apostas": item.get("quantidade_apostas"),
                "elegivel_para_operacao": False,
            }
    return {"gerado_em": now_iso(), **FLAGS_SEGURAS, "faixas_reprovadas": saida}


def _ciclo_vida(relatorio: Dict[str, Any]) -> Dict[str, Any]:
    faixas = {}
    contagem = defaultdict(int)
    for codigo, item in (relatorio.get("faixas") or {}).items():
        status = item.get("status") or "SEM_AMOSTRA"
        contagem[status] += 1
        faixas[codigo] = {
            "faixa": item.get("faixa"),
            "status": status,
            "motivo": item.get("motivo_status"),
            "roi_real": item.get("roi_real"),
            "lucro_real": item.get("lucro_real"),
            "quantidade_apostas": item.get("quantidade_apostas"),
            "taxa_acerto": item.get("taxa_acerto"),
            "drawdown": item.get("drawdown"),
            "elegivel_para_operacao": item.get("elegivel_para_operacao", False),
        }
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "status_permitidos": ["ATIVA", "OBSERVACAO", "REPROVADA", "SEM_AMOSTRA"],
        "resumo": dict(contagem),
        "faixas": faixas,
    }


def _agregar_matriz(linhas: List[Dict[str, Any]], dimensoes: Tuple[str, str]) -> Dict[str, Dict[str, Any]]:
    agregacoes: Dict[str, Dict[str, Any]] = defaultdict(_nova_agregacao)
    rotulos: Dict[str, Dict[str, str]] = {}
    for linha in linhas:
        resultado = normalize_result(linha.get("resultado") or linha.get("result") or linha.get("status"))
        if resultado not in {"WIN", "LOSS"}:
            continue
        odd = _float(linha.get("odd") or linha.get("odds") or linha.get("cotacao"), 0.0)
        cod_odd, rot_odd = _faixa_odd(odd)
        if not cod_odd:
            continue
        if dimensoes[1] == "confianca":
            prob = _float(linha.get("probabilidade_calibrada") or linha.get("prob_calibrada") or linha.get("confianca") or linha.get("prob_modelo") or linha.get("prob"), 0.0)
            if prob > 1.0:
                prob /= 100.0
            cod_dim, rot_dim = _faixa_confianca(prob)
            if not cod_dim:
                continue
            chave = f"{cod_odd}__{cod_dim}"
            rotulos[chave] = {"faixa_odds": rot_odd or cod_odd, "faixa_confianca": rot_dim or cod_dim}
        else:
            mercado = str(linha.get("mercado") or linha.get("mercado") or linha.get("event") or "DESCONHECIDO").strip() or "DESCONHECIDO"
            chave = f"{cod_odd}__{mercado}"
            rotulos[chave] = {"faixa_odds": rot_odd or cod_odd, "mercado": mercado}
        lucro, stake = _lucro_e_stake(linha, resultado, odd)
        agg = agregacoes[chave]
        agg["quantidade_apostas"] += 1
        agg["vitorias"] += 1 if resultado == "WIN" else 0
        agg["derrotas"] += 1 if resultado == "LOSS" else 0
        agg["stake_total"] += stake
        agg["lucro_real"] += lucro
        agg["odd_soma"] += odd
        agg["lucros_sequenciais"].append(lucro)
    saida = {}
    for chave, agg in agregacoes.items():
        item = _finalizar_agregacao(agg, rotulos[chave]["faixa_odds"])
        item.update(rotulos[chave])
        saida[chave] = item
    return saida


def _matriz_odds_confianca(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "descricao": "Matriz de resultado real por combinação de faixa de odds e faixa de confiança.",
        "matriz": _agregar_matriz(linhas, ("odds", "confianca")),
    }


def _matriz_odds_mercado(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "descricao": "Matriz de resultado real por combinação de faixa de odds e mercado.",
        "matriz": _agregar_matriz(linhas, ("odds", "mercado")),
    }


def _dashboard(relatorio: Dict[str, Any], classificacao: Dict[str, Any], reprovadas: Dict[str, Any]) -> None:
    faixas = relatorio.get("faixas") or {}
    linhas = []
    for item in faixas.values():
        linhas.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('faixa')))}</td>"
            f"<td>{html.escape(str(item.get('status')))}</td>"
            f"<td>{item.get('quantidade_apostas')}</td>"
            f"<td>{item.get('roi_real')}</td>"
            f"<td>{item.get('lucro_real')}</td>"
            f"<td>{item.get('taxa_acerto')}</td>"
            f"<td>{item.get('drawdown')}</td>"
            f"<td>{item.get('odd_media')}</td>"
            "</tr>"
        )
    melhores = classificacao.get("melhor_roi", [])[:5]
    piores = sorted(faixas.values(), key=lambda x: (_float(x.get("roi_real")), -int(x.get("quantidade_apostas") or 0)))[:5]
    lista_melhores = "".join(f"<li><strong>{html.escape(str(x.get('faixa')))}</strong> — ROI {x.get('roi_real')} | Lucro {x.get('lucro_real')} | Status {x.get('status')}</li>" for x in melhores)
    lista_piores = "".join(f"<li><strong>{html.escape(str(x.get('faixa')))}</strong> — ROI {x.get('roi_real')} | Lucro {x.get('lucro_real')} | Status {x.get('status')}</li>" for x in piores)
    html_text = f"""<!doctype html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\">
  <title>Governança de Odds</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e5e7eb; margin: 24px; }}
    h1, h2 {{ color: #f8fafc; }}
    .card {{ background: #111827; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #374151; text-align: left; }}
    th {{ color: #93c5fd; }}
    .seguro {{ color: #86efac; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Governança de Odds</h1>
  <div class=\"card\">
    <p><strong>Gerado em:</strong> {html.escape(str(relatorio.get('gerado_em')))}</p>
    <p class=\"seguro\">Modo simulação: ativo · Aposta real: desabilitada · Recomendações: desabilitadas</p>
    <p>Este painel é apenas analítico e não libera operação real.</p>
  </div>
  <div class=\"card\">
    <h2>Melhores faixas de odds</h2>
    <ul>{lista_melhores}</ul>
  </div>
  <div class=\"card\">
    <h2>Piores faixas de odds</h2>
    <ul>{lista_piores}</ul>
  </div>
  <div class=\"card\">
    <h2>Resumo por faixa</h2>
    <table>
      <thead><tr><th>Faixa</th><th>Status</th><th>Apostas</th><th>ROI real</th><th>Lucro real</th><th>Taxa de acerto</th><th>Drawdown</th><th>Odd média</th></tr></thead>
      <tbody>{''.join(linhas)}</tbody>
    </table>
  </div>
  <div class=\"card\">
    <h2>Faixas reprovadas</h2>
    <pre>{html.escape(str(reprovadas.get('faixas_reprovadas', {})))}</pre>
  </div>
</body>
</html>
"""
    DASHBOARD_ODDS.write_text(html_text, encoding="utf-8")


def _atualizar_dashboard_executivo() -> None:
    candidatos = [
        REPORTS_DIR / "liga_painel_governanca.html",
        REPORTS_DIR / "painel_governanca_confianca.html",
        REPORTS_DIR / "painel_governanca.html",
    ]
    bloco = """
<section class=\"card\">
  <h2>GOVERNANÇA DE ODDS</h2>
  <p>Relatórios em português gerados em <code>04_ml/reports/governanca_odds.json</code>.</p>
  <p>Modo simulação ativo. Aposta real e recomendações permanecem desabilitadas.</p>
</section>
"""
    for dashboard in candidatos:
        if not dashboard.exists():
            continue
        try:
            texto = dashboard.read_text(encoding="utf-8")
            if "GOVERNANÇA DE ODDS" not in texto:
                texto = texto.replace("</body>", bloco + "\n</body>") if "</body>" in texto else texto + bloco
                dashboard.write_text(texto, encoding="utf-8")
        except Exception:
            continue


def _relatorio_markdown(relatorio: Dict[str, Any]) -> None:
    contagem = defaultdict(int)
    for item in (relatorio.get("faixas") or {}).values():
        contagem[item.get("status", "SEM_AMOSTRA")] += 1
    texto = f"""# FASE 21 — Governança de Odds

Pacote gerado em modo seguro/simulação.

## Artefatos criados

- `04_ml/27_governanca_odds.py`
- `04_ml/reports/governanca_odds.json`
- `04_ml/reports/classificacao_odds.json`
- `04_ml/reports/auditoria_odds.json`
- `04_ml/reports/faixas_odds_reprovadas.json`
- `04_ml/reports/ciclo_vida_odds.json`
- `04_ml/reports/matriz_odds_confianca.json`
- `04_ml/reports/matriz_odds_mercado.json`
- `04_ml/reports/painel_governanca_odds.html`

## Segurança

- `modo_simulacao = true`
- `aposta_real_habilitada = false`
- `recomendacoes_habilitadas = false`
- Nenhum guard foi removido.
- Nenhuma aposta foi liberada.
- O campo `elegivel_para_operacao` é apenas analítico.

## Resumo

- Faixas ATIVA: {contagem.get('ATIVA', 0)}
- Faixas OBSERVACAO: {contagem.get('OBSERVACAO', 0)}
- Faixas REPROVADA: {contagem.get('REPROVADA', 0)}
- Faixas SEM_AMOSTRA: {contagem.get('SEM_AMOSTRA', 0)}
"""
    RELATORIO_FASE.write_text(texto, encoding="utf-8")


def executar() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    linhas, fonte = _ler_historico()
    relatorio = _analisar_faixas(linhas)
    relatorio["fonte"] = fonte
    classificacao = _classificacao(relatorio)
    auditoria = {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "descricao": "Auditoria operacional por faixa de odds.",
        "faixas": relatorio.get("faixas") or {},
    }
    reprovadas = _faixas_reprovadas(relatorio)
    ciclo = _ciclo_vida(relatorio)
    matriz_confianca = _matriz_odds_confianca(linhas)
    matriz_mercado = _matriz_odds_mercado(linhas)

    write_json(RELATORIO_GOVERNANCA, relatorio)
    write_json(RANKING_ODDS, classificacao)
    write_json(AUDITORIA_ODDS, auditoria)
    write_json(FAIXAS_ODDS_REPROVADAS, reprovadas)
    write_json(CICLO_VIDA_ODDS, ciclo)
    write_json(MATRIZ_ODDS_CONFIANCA, matriz_confianca)
    write_json(MATRIZ_ODDS_MERCADO, matriz_mercado)
    _dashboard(relatorio, classificacao, reprovadas)
    _atualizar_dashboard_executivo()
    _relatorio_markdown(relatorio)
    return relatorio


def main() -> int:
    relatorio = executar()
    print("\nFASE 21 — Governança de Odds")
    print("Modo simulação: ATIVO")
    print("Aposta real habilitada: NÃO")
    print("Recomendações habilitadas: NÃO")
    print("\nResumo das faixas:")
    for _, item in (relatorio.get("faixas") or {}).items():
        print(
            f"- {item.get('faixa')}: {item.get('status')} | "
            f"apostas={item.get('quantidade_apostas')} | "
            f"ROI={item.get('roi_real')} | "
            f"lucro={item.get('lucro_real')} | "
            f"taxa_acerto={item.get('taxa_acerto')} | "
            f"drawdown={item.get('drawdown')}"
        )
    print("\nRelatórios criados em 04_ml/reports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
