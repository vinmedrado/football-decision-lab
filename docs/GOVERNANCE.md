# Governança estatística

## Evidências separadas

### Backfill histórico

Serve para depurar o pipeline, estimar sensibilidade e localizar riscos. Pode orientar hipóteses, mas é vulnerável a seleção, iteração e mudanças de regra.

### Paper trading prospectivo

Registra sinais com o relógio correndo e preserva a decisão original. É o conjunto usado para decidir se o sistema merece nova fase de validação.

## Critérios de leitura

- **100 liquidados**: primeira análise diagnóstica;
- **300 liquidados**: análise de decisão;
- **500 liquidados**: leitura mais robusta;
- nenhum limiar isolado autoriza uso real;
- ROI deve ser lido junto com intervalo de confiança, drawdown, estabilidade temporal e calibração.

## Calibração

Probabilidades são avaliadas por Brier score, log loss e erro de calibração. O snapshot histórico apresenta Brier de `0,2585`, pior que o baseline constante de `0,2500`. O número é exibido como alerta metodológico, não escondido por um ROI positivo.

## Modelo congelado

O bundle operacional registra hashes SHA-256 do modelo, metadados, features, encoders e especialistas contextuais. Durante o ciclo:

- não há recalibração automática;
- challengers podem ser avaliados separadamente;
- nenhuma versão substitui o champion sem revisão;
- toda mudança relevante deve iniciar novo ciclo identificável.

## Limites

O projeto ainda não demonstrou lucro real. O snapshot possui intervalo de confiança do ROI de `-3,26%` a `9,32%` e queda da primeira para a segunda metade. Esses sinais justificam a coleta prospectiva em vez de uma conclusão antecipada.
