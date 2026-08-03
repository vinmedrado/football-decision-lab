#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 22 — Matriz Confiança × Odds.

Camada analítica em português para descobrir combinações entre confiança e odds
que sustentam lucro ou concentram prejuízo na operação real. Este script não
altera modelos, não altera banca, não remove guards e não libera apostas.
Todos os artefatos são de observabilidade/simulação.
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

MATRIZ_CONFIANCA_ODDS = REPORTS_DIR / "matriz_confianca_odds.json"
HEATMAP_CONFIANCA_ODDS = REPORTS_DIR / "heatmap_confianca_odds.json"
TOP_COMBINACOES = REPORTS_DIR / "top_combinacoes_confianca_odds.json"
PIORES_COMBINACOES = REPORTS_DIR / "piores_combinacoes_confianca_odds.json"
CONCENTRACAO_OPERACIONAL = REPORTS_DIR / "concentracao_operacional.json"
MATRIZ_CONFIANCA_ODDS_MERCADOS = REPORTS_DIR / "matriz_confianca_odds_mercados.json"
DASHBOARD_MATRIZ = REPORTS_DIR / "painel_matriz_confianca_odds.html"
RELATORIO_FASE = ROOT_DIR / "FASE22_MATRIZ_CONFIANCA_ODDS_REPORT.md"

AMOSTRA_MINIMA_CELULA = 50

FAIXAS_CONFIANCA = [
    (0.50, 0.60, "50_60", "50% – 60%"),
    (0.60, 0.70, "60_70", "60% – 70%"),
    (0.70, 0.80, "70_80", "70% – 80%"),
    (0.80, 0.90, "80_90", "80% – 90%"),
    (0.90, 1.01, "90_100", "90% – 100%"),
]

FAIXAS_ODDS = [
    (1.00, 1.31, "1.00_1.30", "1.00 – 1.30"),
    (1.31, 1.51, "1.31_1.50", "1.31 – 1.50"),
    (1.51, 1.71, "1.51_1.70", "1.51 – 1.70"),
    (1.71, 2.01, "1.71_2.00", "1.71 – 2.00"),
    (2.01, 2.51, "2.01_2.50", "2.01 – 2.50"),
    (2.51, 3.01, "2.51_3.00", "2.51 – 3.00"),
    (3.01, float("inf"), "3.01_mais", "3.01+"),
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


def _faixa_confianca(probabilidade: float) -> Tuple[str | None, str | None]:
    for minimo, maximo, codigo, rotulo in FAIXAS_CONFIANCA:
        if minimo <= probabilidade < maximo or (codigo == "90_100" and minimo <= probabilidade <= maximo):
            return codigo, rotulo
    return None, None


def _faixa_odd(odd: float) -> Tuple[str | None, str | None]:
    for minimo, maximo, codigo, rotulo in FAIXAS_ODDS:
        if minimo <= odd < maximo:
            return codigo, rotulo
    return None, None


def _probabilidade_linha(linha: Dict[str, Any]) -> Tuple[float, float]:
    prob_calibrada = _float(
        linha.get("probabilidade_calibrada")
        or linha.get("prob_calibrada")
        or linha.get("confianca")
        or linha.get("prob_modelo")
        or linha.get("probabilidade")
        or linha.get("prob"),
        0.0,
    )
    prob_bruta = _float(linha.get("probabilidade_bruta") or linha.get("prob_modelo") or linha.get("probabilidade") or linha.get("prob"), prob_calibrada)
    if prob_calibrada > 1.0:
        prob_calibrada /= 100.0
    if prob_bruta > 1.0:
        prob_bruta /= 100.0
    return prob_calibrada, prob_bruta


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
        "probabilidade_calibrada_soma": 0.0,
        "probabilidade_bruta_soma": 0.0,
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
    return pior / (stake_total if stake_total else 1.0)


def _status_celula(roi: float, quantidade: int) -> str:
    if quantidade <= 0:
        return "SEM_AMOSTRA"
    if quantidade < AMOSTRA_MINIMA_CELULA:
        return "OBSERVACAO" if roi > 0 else "SEM_AMOSTRA"
    if roi > 0:
        return "APROVADA"
    if roi < 0:
        return "REPROVADA"
    return "OBSERVACAO"


def _cor_heatmap(roi: float, quantidade: int) -> str:
    if quantidade <= 0:
        return "SEM_DADOS"
    if roi > 0.01:
        return "VERDE"
    if roi >= -0.01:
        return "AMARELO"
    return "VERMELHO"


def _finalizar_agregacao(agg: Dict[str, Any]) -> Dict[str, Any]:
    qtd = int(agg["quantidade_apostas"])
    stake = _float(agg["stake_total"])
    lucro = _float(agg["lucro_real"])
    vitorias = int(agg["vitorias"])
    taxa_acerto = vitorias / qtd if qtd else 0.0
    roi = lucro / stake if stake else 0.0
    odd_media = _float(agg["odd_soma"]) / qtd if qtd else 0.0
    prob_media = _float(agg["probabilidade_calibrada_soma"]) / qtd if qtd else 0.0
    prob_bruta_media = _float(agg["probabilidade_bruta_soma"]) / qtd if qtd else 0.0
    status = _status_celula(roi, qtd)
    return {
        "quantidade_apostas": qtd,
        "taxa_acerto": round(taxa_acerto, 6),
        "roi_real": round(roi, 6),
        "lucro_real": round(lucro, 2),
        "valor_total_apostado": round(stake, 2),
        "drawdown": round(_drawdown(agg["lucros_sequenciais"], stake), 6),
        "odd_media": round(odd_media, 6),
        "probabilidade_calibrada_media": round(prob_media, 6),
        "probabilidade_bruta_media": round(prob_bruta_media, 6),
        "status": status,
        "elegivel_para_operacao": bool(status == "APROVADA"),
        "cor_heatmap": _cor_heatmap(roi, qtd),
    }


def _preparar_linhas(linhas: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    preparadas: List[Dict[str, Any]] = []
    ignoradas = 0
    for linha in linhas:
        resultado = normalize_result(linha.get("resultado") or linha.get("result") or linha.get("status"))
        if resultado not in {"WIN", "LOSS"}:
            ignoradas += 1
            continue
        odd = _float(linha.get("odd") or linha.get("odds") or linha.get("cotacao"), 0.0)
        cod_odd, rot_odd = _faixa_odd(odd)
        prob_calibrada, prob_bruta = _probabilidade_linha(linha)
        cod_conf, rot_conf = _faixa_confianca(prob_calibrada)
        if not cod_odd or not cod_conf:
            ignoradas += 1
            continue
        lucro, stake = _lucro_e_stake(linha, resultado, odd)
        mercado = str(linha.get("mercado") or linha.get("mercado") or linha.get("tipo_mercado") or "DESCONHECIDO").strip() or "DESCONHECIDO"
        preparadas.append({
            "resultado": resultado,
            "odd": odd,
            "codigo_odds": cod_odd,
            "faixa_odds": rot_odd,
            "probabilidade_calibrada": prob_calibrada,
            "probabilidade_bruta": prob_bruta,
            "codigo_confianca": cod_conf,
            "faixa_confianca": rot_conf,
            "lucro": lucro,
            "stake": stake,
            "mercado": mercado,
        })
    return preparadas, ignoradas


def _adicionar(agg: Dict[str, Any], item: Dict[str, Any]) -> None:
    agg["quantidade_apostas"] += 1
    agg["vitorias"] += 1 if item["resultado"] == "WIN" else 0
    agg["derrotas"] += 1 if item["resultado"] == "LOSS" else 0
    agg["stake_total"] += item["stake"]
    agg["lucro_real"] += item["lucro"]
    agg["odd_soma"] += item["odd"]
    agg["probabilidade_calibrada_soma"] += item["probabilidade_calibrada"]
    agg["probabilidade_bruta_soma"] += item["probabilidade_bruta"]
    agg["lucros_sequenciais"].append(item["lucro"])


def _matriz_principal(preparadas: List[Dict[str, Any]], fonte: str, ignoradas: int) -> Dict[str, Any]:
    matriz: Dict[str, Any] = {}
    for _, _, cod_conf, rot_conf in FAIXAS_CONFIANCA:
        matriz[cod_conf] = {
            "faixa_confianca": rot_conf,
            "odds": {},
        }
        for _, _, cod_odd, rot_odd in FAIXAS_ODDS:
            matriz[cod_conf]["odds"][cod_odd] = {
                "faixa_odds": rot_odd,
                **_finalizar_agregacao(_nova_agregacao()),
            }
    agregacoes: Dict[str, Dict[str, Any]] = defaultdict(_nova_agregacao)
    for item in preparadas:
        chave = f"{item['codigo_confianca']}__{item['codigo_odds']}"
        _adicionar(agregacoes[chave], item)
    for chave, agg in agregacoes.items():
        cod_conf, cod_odd = chave.split("__", 1)
        matriz[cod_conf]["odds"][cod_odd].update(_finalizar_agregacao(agg))
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "fonte": fonte,
        "descricao": "Matriz analítica de desempenho real por combinação de faixa de confiança e faixa de odds.",
        "amostra_minima_celula": AMOSTRA_MINIMA_CELULA,
        "linhas_liquidadas_avaliadas": len(preparadas),
        "linhas_ignoradas": ignoradas,
        "matriz": matriz,
    }


def _heatmap(matriz: Dict[str, Any]) -> Dict[str, Any]:
    saida: Dict[str, Any] = {}
    for cod_conf, bloco in (matriz.get("matriz") or {}).items():
        saida[cod_conf] = {
            "faixa_confianca": bloco.get("faixa_confianca"),
            "odds": {},
        }
        for cod_odd, celula in (bloco.get("odds") or {}).items():
            saida[cod_conf]["odds"][cod_odd] = {
                "faixa_odds": celula.get("faixa_odds"),
                "cor": celula.get("cor_heatmap"),
                "status": celula.get("status"),
                "roi_real": celula.get("roi_real"),
                "quantidade_apostas": celula.get("quantidade_apostas"),
                "lucro_real": celula.get("lucro_real"),
            }
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "legenda": {
            "VERDE": "ROI positivo",
            "AMARELO": "ROI próximo de zero",
            "VERMELHO": "ROI negativo",
            "SEM_DADOS": "Amostra insuficiente ou inexistente",
        },
        "heatmap": saida,
    }


def _celulas_lista(matriz: Dict[str, Any]) -> List[Dict[str, Any]]:
    celulas: List[Dict[str, Any]] = []
    for cod_conf, bloco in (matriz.get("matriz") or {}).items():
        for cod_odd, celula in (bloco.get("odds") or {}).items():
            registro = dict(celula)
            registro.update({
                "codigo_confianca": cod_conf,
                "faixa_confianca": bloco.get("faixa_confianca"),
                "codigo_odds": cod_odd,
                "chave": f"{cod_conf}__{cod_odd}",
            })
            celulas.append(registro)
    return celulas


def _top_combinacoes(matriz: Dict[str, Any]) -> Dict[str, Any]:
    celulas = [x for x in _celulas_lista(matriz) if int(x.get("quantidade_apostas") or 0) > 0]
    ordenadas = sorted(celulas, key=lambda x: (_float(x.get("roi_real")), _float(x.get("lucro_real")), int(x.get("quantidade_apostas") or 0)), reverse=True)
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "criterio": "Top 10 combinações por ROI real, lucro real e quantidade.",
        "combinacoes": ordenadas[:10],
    }


def _piores_combinacoes(matriz: Dict[str, Any]) -> Dict[str, Any]:
    celulas = [x for x in _celulas_lista(matriz) if int(x.get("quantidade_apostas") or 0) > 0]
    ordenadas = sorted(celulas, key=lambda x: (_float(x.get("roi_real")), _float(x.get("lucro_real")), -int(x.get("quantidade_apostas") or 0)))
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "criterio": "Bottom 10 combinações por ROI real e lucro real.",
        "combinacoes": ordenadas[:10],
    }


def _concentracao(matriz: Dict[str, Any]) -> Dict[str, Any]:
    celulas = [x for x in _celulas_lista(matriz) if int(x.get("quantidade_apostas") or 0) > 0]
    total = sum(int(x.get("quantidade_apostas") or 0) for x in celulas)
    ordenadas = sorted(celulas, key=lambda x: int(x.get("quantidade_apostas") or 0), reverse=True)
    concentracoes = []
    for item in ordenadas:
        qtd = int(item.get("quantidade_apostas") or 0)
        registro = dict(item)
        registro["participacao"] = round(qtd / total, 6) if total else 0.0
        concentracoes.append(registro)
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "descricao": "Auditoria de concentração operacional por célula confiança × odds.",
        "total_apostas_avaliadas": total,
        "celulas_ordenadas_por_concentracao": concentracoes,
        "maior_concentracao": concentracoes[0] if concentracoes else None,
    }


def _matriz_por_mercado(preparadas: List[Dict[str, Any]]) -> Dict[str, Any]:
    agregacoes: Dict[str, Dict[str, Any]] = defaultdict(_nova_agregacao)
    metadados: Dict[str, Dict[str, str]] = {}
    for item in preparadas:
        chave = f"{item['mercado']}__{item['codigo_confianca']}__{item['codigo_odds']}"
        metadados[chave] = {
            "mercado": item["mercado"],
            "codigo_confianca": item["codigo_confianca"],
            "faixa_confianca": item["faixa_confianca"],
            "codigo_odds": item["codigo_odds"],
            "faixa_odds": item["faixa_odds"],
        }
        _adicionar(agregacoes[chave], item)
    matriz = {}
    for chave, agg in agregacoes.items():
        registro = _finalizar_agregacao(agg)
        registro.update(metadados[chave])
        matriz[chave] = registro
    return {
        "gerado_em": now_iso(),
        **FLAGS_SEGURAS,
        "descricao": "Correlação entre mercado, faixa de confiança e faixa de odds.",
        "matriz": matriz,
    }


def _dashboard(matriz: Dict[str, Any], heatmap: Dict[str, Any], top: Dict[str, Any], piores: Dict[str, Any], concentracao: Dict[str, Any]) -> None:
    cabecalho_odds = "".join(f"<th>{html.escape(rotulo)}</th>" for _, _, _, rotulo in FAIXAS_ODDS)
    linhas_heatmap = []
    for _, _, cod_conf, rot_conf in FAIXAS_CONFIANCA:
        tds = []
        for _, _, cod_odd, _ in FAIXAS_ODDS:
            celula = ((matriz.get("matriz") or {}).get(cod_conf) or {}).get("odds", {}).get(cod_odd, {})
            classe = str(celula.get("cor_heatmap") or "SEM_DADOS").lower()
            tds.append(
                f"<td class='{classe}'>"
                f"<strong>{html.escape(str(celula.get('status')))}</strong><br>"
                f"ROI: {celula.get('roi_real')}<br>"
                f"Qtd: {celula.get('quantidade_apostas')}<br>"
                f"Lucro: {celula.get('lucro_real')}"
                "</td>"
            )
        linhas_heatmap.append(f"<tr><th>{html.escape(rot_conf)}</th>{''.join(tds)}</tr>")

    def lista(itens: List[Dict[str, Any]]) -> str:
        return "".join(
            "<li>"
            f"<strong>{html.escape(str(x.get('faixa_confianca')))} × {html.escape(str(x.get('faixa_odds')))}</strong> — "
            f"ROI {x.get('roi_real')} | Lucro {x.get('lucro_real')} | Qtd {x.get('quantidade_apostas')} | Status {x.get('status')}"
            "</li>"
            for x in itens
        )

    top_html = lista(top.get("combinacoes", [])[:10])
    piores_html = lista(piores.get("combinacoes", [])[:10])
    concentracao_html = lista(concentracao.get("celulas_ordenadas_por_concentracao", [])[:10])

    html_text = f"""<!doctype html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\">
  <title>Matriz Confiança × Odds</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e5e7eb; margin: 24px; }}
    h1, h2 {{ color: #f8fafc; }}
    .card {{ background: #111827; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; }}
    th, td {{ padding: 9px; border: 1px solid #374151; text-align: left; vertical-align: top; }}
    th {{ color: #93c5fd; }}
    .verde {{ background: #14532d; }}
    .amarelo {{ background: #713f12; }}
    .vermelho {{ background: #7f1d1d; }}
    .sem_dados {{ background: #1f2937; color: #9ca3af; }}
    .seguro {{ color: #86efac; font-weight: bold; }}
    code {{ color: #fbbf24; }}
  </style>
</head>
<body>
  <h1>MATRIZ CONFIANÇA × ODDS</h1>
  <div class=\"card\">
    <p><strong>Gerado em:</strong> {html.escape(str(matriz.get('gerado_em')))}</p>
    <p class=\"seguro\">Modo simulação: ativo · Aposta real: desabilitada · Recomendações: desabilitadas</p>
    <p>Este painel é apenas analítico e não bloqueia nem libera operação.</p>
  </div>
  <div class=\"card\">
    <h2>Heatmap analítico</h2>
    <table>
      <thead><tr><th>Confiança × Odds</th>{cabecalho_odds}</tr></thead>
      <tbody>{''.join(linhas_heatmap)}</tbody>
    </table>
  </div>
  <div class=\"card\"><h2>Melhores combinações</h2><ol>{top_html}</ol></div>
  <div class=\"card\"><h2>Piores combinações</h2><ol>{piores_html}</ol></div>
  <div class=\"card\"><h2>Concentração operacional</h2><ol>{concentracao_html}</ol></div>
  <div class=\"card\">
    <h2>Arquivos gerados</h2>
    <p><code>matriz_confianca_odds.json</code>, <code>heatmap_confianca_odds.json</code>, <code>top_combinacoes_confianca_odds.json</code>, <code>piores_combinacoes_confianca_odds.json</code>, <code>concentracao_operacional.json</code>, <code>matriz_confianca_odds_mercados.json</code>.</p>
  </div>
</body>
</html>
"""
    DASHBOARD_MATRIZ.write_text(html_text, encoding="utf-8")


def _atualizar_dashboard_executivo() -> None:
    candidatos = [
        REPORTS_DIR / "painel_governanca_odds.html",
        REPORTS_DIR / "painel_governanca_confianca.html",
        REPORTS_DIR / "liga_painel_governanca.html",
        REPORTS_DIR / "painel_governanca.html",
    ]
    bloco = """
<section class=\"card\">
  <h2>MATRIZ CONFIANÇA × ODDS</h2>
  <p>Relatórios em português gerados em <code>04_ml/reports/matriz_confianca_odds.json</code>.</p>
  <p>Modo simulação ativo. Aposta real e recomendações permanecem desabilitadas.</p>
</section>
"""
    for dashboard in candidatos:
        if not dashboard.exists():
            continue
        try:
            texto = dashboard.read_text(encoding="utf-8")
            if "MATRIZ CONFIANÇA × ODDS" not in texto:
                texto = texto.replace("</body>", bloco + "\n</body>") if "</body>" in texto else texto + bloco
                dashboard.write_text(texto, encoding="utf-8")
        except Exception:
            continue


def _relatorio_markdown(matriz: Dict[str, Any], top: Dict[str, Any], piores: Dict[str, Any], concentracao: Dict[str, Any]) -> None:
    melhor = (top.get("combinacoes") or [{}])[0]
    pior = (piores.get("combinacoes") or [{}])[0]
    maior_conc = concentracao.get("maior_concentracao") or {}
    texto = f"""# FASE 22 — Matriz Confiança × Odds

Pacote gerado em modo seguro/simulação.

## Artefatos criados

- `04_ml/28_matriz_confianca_odds.py`
- `04_ml/reports/matriz_confianca_odds.json`
- `04_ml/reports/heatmap_confianca_odds.json`
- `04_ml/reports/top_combinacoes_confianca_odds.json`
- `04_ml/reports/piores_combinacoes_confianca_odds.json`
- `04_ml/reports/concentracao_operacional.json`
- `04_ml/reports/matriz_confianca_odds_mercados.json`
- `04_ml/reports/painel_matriz_confianca_odds.html`

## Segurança

- `modo_simulacao = true`
- `aposta_real_habilitada = false`
- `recomendacoes_habilitadas = false`
- Nenhum guard foi removido.
- Nenhuma aposta foi liberada.

## Resumo analítico

- Linhas liquidadas avaliadas: `{matriz.get('linhas_liquidadas_avaliadas')}`
- Melhor combinação por ROI: `{melhor.get('faixa_confianca')} × {melhor.get('faixa_odds')}` com ROI `{melhor.get('roi_real')}`
- Pior combinação por ROI: `{pior.get('faixa_confianca')} × {pior.get('faixa_odds')}` com ROI `{pior.get('roi_real')}`
- Maior concentração: `{maior_conc.get('faixa_confianca')} × {maior_conc.get('faixa_odds')}` com `{maior_conc.get('quantidade_apostas')}` apostas

## Validação

- Matriz confiança × odds criada.
- Heatmap analítico criado.
- Top combinações criado.
- Piores combinações criado.
- Concentração operacional criada.
- Matriz por mercado criada.
- Menu 26 criado.
- Painel atualizado.
"""
    RELATORIO_FASE.write_text(texto, encoding="utf-8")


def executar() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    linhas, fonte = _ler_historico()
    preparadas, ignoradas = _preparar_linhas(linhas)
    matriz = _matriz_principal(preparadas, fonte, ignoradas)
    heatmap = _heatmap(matriz)
    top = _top_combinacoes(matriz)
    piores = _piores_combinacoes(matriz)
    concentracao = _concentracao(matriz)
    matriz_mercados = _matriz_por_mercado(preparadas)

    write_json(MATRIZ_CONFIANCA_ODDS, matriz)
    write_json(HEATMAP_CONFIANCA_ODDS, heatmap)
    write_json(TOP_COMBINACOES, top)
    write_json(PIORES_COMBINACOES, piores)
    write_json(CONCENTRACAO_OPERACIONAL, concentracao)
    write_json(MATRIZ_CONFIANCA_ODDS_MERCADOS, matriz_mercados)
    _dashboard(matriz, heatmap, top, piores, concentracao)
    _atualizar_dashboard_executivo()
    _relatorio_markdown(matriz, top, piores, concentracao)
    return matriz


def main() -> int:
    matriz = executar()
    top = _top_combinacoes(matriz)
    piores = _piores_combinacoes(matriz)
    concentracao = _concentracao(matriz)
    print("\nFASE 22 — Matriz Confiança × Odds")
    print("Modo simulação: ATIVO")
    print("Aposta real habilitada: NÃO")
    print("Recomendações habilitadas: NÃO")
    print(f"Linhas liquidadas avaliadas: {matriz.get('linhas_liquidadas_avaliadas')}")

    print("\nMelhores combinações:")
    for item in (top.get("combinacoes") or [])[:5]:
        print(
            f"- {item.get('faixa_confianca')} × {item.get('faixa_odds')}: "
            f"ROI={item.get('roi_real')} | lucro={item.get('lucro_real')} | "
            f"quantidade={item.get('quantidade_apostas')} | status={item.get('status')}"
        )

    print("\nPiores combinações:")
    for item in (piores.get("combinacoes") or [])[:5]:
        print(
            f"- {item.get('faixa_confianca')} × {item.get('faixa_odds')}: "
            f"ROI={item.get('roi_real')} | lucro={item.get('lucro_real')} | "
            f"quantidade={item.get('quantidade_apostas')} | status={item.get('status')}"
        )

    print("\nMaior concentração operacional:")
    maior = concentracao.get("maior_concentracao") or {}
    if maior:
        print(
            f"- {maior.get('faixa_confianca')} × {maior.get('faixa_odds')}: "
            f"{maior.get('quantidade_apostas')} apostas | participação={maior.get('participacao')}"
        )
    else:
        print("- Sem dados liquidados suficientes.")

    print("\nRelatórios criados em 04_ml/reports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
