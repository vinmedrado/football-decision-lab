# Football Decision Lab — operação paper automática

## Estado adotado

- Modo: `PAPER_ONLY`.
- Apostas reais: bloqueadas.
- Estratégia em validação: `TG_FT_O25`.
- Ciclo: `PAPER_2026_07_TG_FT_O25_V1`.
- Stake contábil fixa: R$ 5,00.
- Revisão principal: não antes de 01/10/2026 e, preferencialmente, com pelo menos 300 apostas liquidadas.
- Modelo, especialistas, encoders e lista de features: congelados e verificados por SHA-256.

O histórico de `backfill_simulado` continua preservado, mas não entra nas métricas prospectivas. As novas entradas usam `origem=paper_forward`.

## Calendário automático

O Windows chama `automation_orchestrator.py --run-due` a cada 15 minutos. O controlador mantém lock, estado, tentativas e recuperação de horários perdidos.

| Rotina | Frequência | Finalidade |
|---|---:|---|
| Atualização do `daily` | a cada 60 minutos | Atualiza agenda, horários e odds |
| Captura + registro paper | a cada 15 minutos | Captura uma única vez jogos entre 30 e 90 minutos do kickoff |
| Settlement por kickoff | a cada 15 minutos | Primeira tentativa 130 minutos após o início e retentativas graduais |
| Settlement | diária, 10:00 e 22:30 | Atualiza placares pela base e Flashscore |
| Monitor e alertas | diária, 23:15 | ROI, lucro, drawdown, Brier, calibração e incerteza |
| Base oficial | semanal, terça 02:00 | Atualização incremental e reconstrução dos dados |
| Backtest + ML desafiante | mensal, dia 2 às 03:00 | Gera artefatos versionados sem trocar o modelo paper |

Se o notebook estiver desligado ou suspenso, rotinas gerais serão recuperadas na próxima verificação. Uma janela de captura perdida não gera aposta retrospectiva. O notebook precisa estar ligado e conectado à internet para tarefas de rede.

## Política de calibração e modelos

A calibração do modelo em avaliação não é alterada durante o ciclo. Ela é monitorada diariamente:

- antes de 50 liquidadas: apenas coleta;
- a partir de 50: alerta se o erro de calibração superar 10%;
- 100 liquidadas: diagnóstico inicial;
- 300 liquidadas: primeira decisão estatística;
- 500 liquidadas: leitura mais robusta.

O treino mensal escreve em `04_ml/challengers/AAAA-MM`. Não há promoção automática. Trocar o modelo exige encerrar o ciclo atual, revisar backtest/calibração e iniciar um novo `cycle_id`.

## Instalação e inspeção

Congelar/validar o modelo:

```powershell
python 04_ml\paper_model_manager.py validate
```

Instalar as duas tarefas do Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_automation.ps1
```

Ver o que está pendente sem executar:

```powershell
python automation_orchestrator.py --status
python automation_orchestrator.py --run-due --dry-run
```

Remover as tarefas:

```powershell
powershell -ExecutionPolicy Bypass -File .\uninstall_automation.ps1
```

## Dashboard

O painel inicia no login do Windows:

- notebook: `http://127.0.0.1:8060`;
- celular na mesma rede Wi-Fi: `http://IP_DO_NOTEBOOK:8060`.

A aba Banca tem um bloco “Paper trading isolado” com ROI, lucro, intervalo de 95%, Brier, erro de calibração, progresso amostral, ligas, mercados, meses e saúde da automação.

## E-mail opcional

O alerta local funciona sem configuração. Para e-mail, configure as variáveis descritas em `04_ml/config/email.env.example` no Windows. Para Gmail, use senha de app; nunca grave a senha dentro do projeto.
