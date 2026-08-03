#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catalogo central de motivos de bloqueio/alerta da governanca."""
from __future__ import annotations

from typing import Any, Dict

BLOCK_REASON_CATALOG: Dict[str, Dict[str, str]] = {
    "CALIBRACAO_BLOQUEADA": {
        "severity": "BLOQUEADA",
        "description": "Guard de Calibração detectou erro de calibração acima do limite permitido.",
        "acao_recomendada": "Revisar calibração, Brier Score e relatório de calibração antes de liberar previsões.",
    },
    "PROTECAO_SIMULACAO": {
        "severity": "BLOQUEADA",
        "description": "Guard de Simulação colocou o sistema em modo proteção.",
        "acao_recomendada": "Manter execução em simulação e revisar alertas de segurança operacional.",
    },
    "DERIVA_FEATURES_CRITICA": {
        "severity": "BLOQUEADA",
        "description": "Feature Health detectou drift crítico nas variáveis de entrada.",
        "acao_recomendada": "Revalidar distribuição de features e considerar retreinamento/calibração.",
    },
    "REGISTRO_MODELO_INVALIDO": {
        "severity": "BLOQUEADA",
        "description": "Registry sem modelo_ativo, modelo_campeao ou modelo_base válido.",
        "acao_recomendada": "Corrigir model_registry.json e confirmar artefatos de modelo registrados.",
    },
    "MODELO_ATIVO_NAO_ENCONTRADO": {
        "severity": "BLOQUEADA",
        "description": "Arquivo do modelo ativo não foi encontrado.",
        "acao_recomendada": "Restaurar o artefato do modelo ativo ou atualizar o registry.",
    },
    "BASELINE_NAO_ENCONTRADA": {
        "severity": "BLOQUEADA",
        "description": "Métricas baseline não foram encontradas.",
        "acao_recomendada": "Gerar baseline_metrics.json antes de liberar a rotina operacional.",
    },
    "ENVIRONMENT_INVALID": {
        "severity": "BLOQUEADA",
        "description": "Validação de ambiente encontrou diretórios obrigatórios ausentes.",
        "acao_recomendada": "Corrigir estrutura de pastas e executar novamente a validação.",
    },
    "ROI_NEGATIVO": {
        "severity": "BLOQUEADA",
        "description": "Mercado apresenta ROI real negativo.",
        "acao_recomendada": "Manter mercado bloqueado até nova validação estatística.",
    },
    "INSUFFICIENT_SAMPLE": {
        "severity": "ALERTA",
        "description": "Mercado não possui amostra mínima para decisão confiável.",
        "acao_recomendada": "Aguardar mais histórico em modo simulação antes de liberar.",
    },
    "BAIXA_TAXA_ACERTO_MERCADO": {
        "severity": "BLOQUEADA",
        "description": "Mercado abaixo do win rate mínimo configurado.",
        "acao_recomendada": "Revisar performance por mercado e faixas de odd.",
    },
    "MARKET_CALIBRATION_FAIL": {
        "severity": "BLOQUEADA",
        "description": "Mercado com ECE ou erro de calibração acima do limite.",
        "acao_recomendada": "Recalibrar probabilidades por mercado ou manter bloqueio.",
    },
    "ROI_DEGRADATION": {
        "severity": "ALERTA",
        "description": "ROI recente caiu de forma relevante contra o histórico completo.",
        "acao_recomendada": "Investigar drift operacional, liga, odds e qualidade dos dados recentes.",
    },
    "MERCADO_SUPEREXPOSTO": {
        "severity": "ALERTA",
        "description": "Mercado ultrapassou limite de concentração recente.",
        "acao_recomendada": "Reduzir exposição e aguardar normalização da distribuição.",
    },
    "LIGA_SUPEREXPOSTA": {
        "severity": "ALERTA",
        "description": "Liga ultrapassou limite de concentração recente.",
        "acao_recomendada": "Reduzir concentração por liga e revisar classificacao histórico.",
    },
    "LIMITE_DIARIO_ATINGIDO": {
        "severity": "ALERTA",
        "description": "Limite diário de recomendações foi atingido.",
        "acao_recomendada": "Encerrar novas emissões no dia e reavaliar no próximo ciclo.",
    },
    "INCONSISTENCIA_ROI": {
        "severity": "ALERTA",
        "description": "ROI do provider difere do ROI recalculado pelo histórico.",
        "acao_recomendada": "Comparar performance_por_mercado.json com historico_apostas.csv.",
    },
}


def get_motivo_info(motivo: str) -> Dict[str, Any]:
    key = str(motivo or "DESCONHECIDA").upper()
    return dict(BLOCK_REASON_CATALOG.get(key, {
        "severity": "DESCONHECIDA",
        "description": "Motivo não catalogado.",
        "acao_recomendada": "Revisar relatórios de governança e logs operacionais.",
    }), motivo=key)


def explain_motivo(motivo: str) -> str:
    info = get_motivo_info(motivo)
    return str(info.get("description", "Motivo não catalogado."))


if __name__ == "__main__":
    import json
    print(json.dumps(BLOCK_REASON_CATALOG, ensure_ascii=False, indent=2))
