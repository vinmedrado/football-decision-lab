# Football Lab — Política de Reentreinamento Periódico

## Objetivo

Definir uma regra operacional simples, versionada e executável para decidir quando reentreinar os modelos do Football Lab sem alterar backtest, predict, thresholds de produção ou regras de aposta.

## Cadência recomendada

- Reentreinamento mensal como rotina padrão.
- Reentreinamento semanal apenas em períodos de alta mudança operacional, como início de temporada, troca relevante de elencos, muitas partidas acumuladas ou queda comprovada de calibração.

## Critérios mínimos antes de reentreinar

O reentreinamento deve ser considerado quando pelo menos um dos critérios abaixo for atendido:

1. Existirem pelo menos 500 novos jogos incorporados à base desde o último treino.
2. Existirem pelo menos 100 novas apostas/previsões liquidadas no histórico real.
3. O erro absoluto médio de calibração ficar acima de 0.10 em apostas/previsões avaliadas.
4. O Brier Score piorar de forma relevante contra o último relatório salvo ou contra o baseline operacional definido pelo projeto.

## Critério de drift para forçar reentreinamento

O reentreinamento deve ser tratado como necessário quando o monitoramento de drift/calibração gerar alerta `CRITICO`.

Limites operacionais padrão:

- `OK`: erro absoluto médio de calibração <= 0.05
- `ALERTA`: erro absoluto médio de calibração > 0.05 e <= 0.10
- `CRITICO`: erro absoluto médio de calibração > 0.10

## Comando oficial para monitorar drift

```bash
python 04_ml/04_monitor_drift.py
```

## Comando oficial para reentreinar

```bash
python 04_ml/01_dataset_builder.py
python 04_ml/02_train_model.py
```

O terminal operacional também pode ser usado para o fluxo de ML já existente, desde que não altere a lógica de backtest ou de aposta.

## Artefatos esperados após reentreino

Após reentreinar, devem existir ou ser atualizados os artefatos padrão do bloco `04_ml`, especialmente:

- datasets/metadados usados pelo treinamento;
- modelos serializados em `04_ml/models/`;
- resumo de modelos, quando gerado pelo treino;
- logs/saídas de avaliação já previstas pelo treinamento existente.

## Regra de segurança

Esta política não autoriza reescrever modelos, backtest, predict, thresholds de produção, gestão de banca ou regras de aposta. Ela apenas define quando avaliar um novo treino e quais comandos usar.
