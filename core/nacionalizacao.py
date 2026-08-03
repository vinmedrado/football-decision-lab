# -*- coding: utf-8 -*-
"""Camada central de nacionalização operacional do Football Lab.

Mantém termos técnicos internacionais e traduz apenas nomenclaturas
operacionais exibidas em relatórios, menus, dashboards e logs.
"""

MAPEAMENTO_STATUS = {
    "ATIVA": "ATIVA",
    "OBSERVACAO": "OBSERVACAO",
    "APOSENTADA": "APOSENTADA",
    "BLOQUEADA": "BLOQUEADA",
    "DESCONHECIDA": "DESCONHECIDA",
    "ALERTA": "ALERTA",
    "FALHA": "FALHA",
    "SUCESSO": "SUCESSO",
    "ERRO": "ERRO",
    "CRITICO": "CRITICO",
    "PROMOVIDA": "PROMOVIDA",
    "REPROVADA": "REPROVADA",
}

MAPEAMENTO_CLASSIFICACOES = {
    "SUSPEITA_DE_SOBREAJUSTE": "SUSPEITA_DE_SOBREAJUSTE",
    "AMOSTRA_REAL_INSUFICIENTE": "AMOSTRA_REAL_INSUFICIENTE",
    "REPROVADA_NA_OPERACAO": "REPROVADA_NA_OPERACAO",
    "APROVADA_NA_OPERACAO": "APROVADA_NA_OPERACAO",
    "MANTER_ATIVA": "MANTER_ATIVA",
    "APOSENTAR_MERCADO": "APOSENTAR_MERCADO",
}

def nacionalizar_valor(valor):
    if isinstance(valor, str):
        return MAPEAMENTO_STATUS.get(valor, MAPEAMENTO_CLASSIFICACOES.get(valor, valor))
    return valor

def nacionalizar_objeto(obj):
    if isinstance(obj, dict):
        return {k: nacionalizar_objeto(nacionalizar_valor(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nacionalizar_objeto(v) for v in obj]
    return nacionalizar_valor(obj)
