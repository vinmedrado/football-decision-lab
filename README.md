# Football Decision Lab

Pipeline local de pesquisa e operação simulada para mercados de futebol. O
projeto coleta jogos e odds, valida dados no tempo, gera features, roda
backtests, congela modelos, captura entradas pré-jogo e liquida resultados em
uma banca paper.

> Estado em 23/07/2026: sistema operacional em modo `PAPER_ONLY`, automação
> ativa e apostas reais bloqueadas.

## Estado atual

| Item | Configuração |
|---|---|
| Ciclo | `PAPER_2026_07_TG_FT_O25_V1` |
| Mercado avaliado | Mais de 2,5 gols (`TG_FT_O25`) |
| Captura | 30 a 90 minutos antes do jogo |
| Valor simulado | R$ 5,00 por entrada |
| Limite diário | 10 apostas simuladas |
| Apostas reais | Desativadas |
| Primeira revisão | Não antes de 01/10/2026 |
| Meta principal | 300 apostas liquidadas |

O histórico reconstruído permanece disponível para auditoria, mas não é
misturado com o ciclo prospectivo atual.

## Fluxo do pipeline

```text
Jogos e odds do dia
        |
Validação temporal e preparação dos dados
        |
Feature engineering
        |
Backtest por janela temporal
        |
Treinamento e congelamento do modelo
        |
Captura pré-jogo imutável
        |
Banca simulada
        |
Settlement, ROI, risco e calibração
```

O sistema pode analisar muitas linhas para uma mesma partida. Isso não significa
que todas sejam apostas. O dashboard mostra somente entradas oficialmente
aprovadas pelo ciclo paper atual; análises rejeitadas e arquivos pré-ciclo ficam
guardados apenas para auditoria.

## Estrutura do repositório

```text
01_scripts/            coleta, normalização e rotinas de pipeline
02_validation/         validações temporais e qualidade de dados
03_backtest/           avaliação histórica e métricas de estratégia
04_ml/                 treinamento, modelo congelado, paper trading e settlement
core/                  regras e componentes compartilhados
scripts/               utilitários operacionais
tests/                 testes de regressão e automação
web_dashboard_lux/     dashboard local
docs/                  documentação publicada do projeto
site/                  site estático de demonstração
reference/             contrato público mínimo do paper trading
```

Dados brutos, caches, logs, datasets, artefatos de modelo e outputs locais ficam
fora do Git por padrão. Ajuste `.env` a partir de `.env.example` e não versione
chaves de provedores.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r web_dashboard_lux\requirements.txt
copy .env.example .env
```

## Execução manual

Validar o modelo congelado:

```powershell
python 04_ml\paper_model_manager.py validate
```

Inspecionar a automação sem executar rotinas:

```powershell
python automation_orchestrator.py --status
python automation_orchestrator.py --run-due --dry-run
```

Executar rotinas vencidas:

```powershell
python automation_orchestrator.py --run-due
```

Executar testes:

```powershell
pytest tests -q
```

## Operação automática

O Agendador de Tarefas do Windows mantém dois processos:

- `FootballDecisionLab-Paper-Automation`: verifica as rotinas a cada 15 minutos;
- `FootballDecisionLab-Dashboard`: mantém o painel local disponível.

Instalar ou reaplicar as tarefas:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_automation.ps1
```

Remover as tarefas:

```powershell
powershell -ExecutionPolicy Bypass -File .\uninstall_automation.ps1
```

Após desligamento, suspensão ou falta de energia, o controlador recupera rotinas
vencidas quando o notebook volta. Uma janela pré-jogo já perdida não é recriada
retrospectivamente, pois isso contaminaria o experimento.

## Dashboard

- No notebook: `http://127.0.0.1:8060`
- No celular conectado à mesma rede: `http://IP_DO_NOTEBOOK:8060`

O painel apresenta apostas aprovadas do ciclo atual, saldo simulado, ROI, taxa
de acerto, drawdown, resultados pendentes e liquidados, calibração, Brier,
progresso da amostra, saúde da automação, modelos e mercados para inspeção.

O dashboard é local. Não exponha a porta `8060` diretamente na internet.

## Calendário automático

| Rotina | Frequência |
|---|---:|
| Atualizar jogos e odds | a cada 60 minutos |
| Procurar entradas paper | a cada 15 minutos |
| Atualizar resultados por horário | a cada 15 minutos |
| Liquidação geral | diariamente às 10:00 e 22:30 |
| Monitor e alertas | diariamente às 23:15 |
| Atualizar base oficial | terça-feira às 02:00 |
| Treinar desafiante | dia 2 de cada mês às 03:00 |

O modelo desafiante nunca substitui automaticamente o modelo do ciclo atual.
Qualquer promoção exige revisão humana, validação temporal e abertura de um novo
`cycle_id`.

## Documentação de continuidade

- [Operação paper](OPERACAO_PAPER.md)
- [Caderno de continuidade](CADERNO_CONTINUIDADE.md)
- [Política de retreinamento](04_ml/RETRAINING_POLICY.md)
- [Liquidação pelo Flashscore](04_ml/README_SETTLEMENT_FLASHSCORE.md)

## Licença

MIT.

## Aviso

O projeto trabalha com simulações. Resultado histórico positivo não comprova
lucro futuro, e nenhuma saída do sistema constitui recomendação financeira.
