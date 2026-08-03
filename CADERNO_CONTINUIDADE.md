# Caderno de continuidade — Football Decision Lab

> Documento principal para retomar o projeto depois de uma pausa.
>
> Última atualização: 27/07/2026, America/Sao_Paulo.
>
> Ao voltar: leia este arquivo antes de executar qualquer backfill, treino, limpeza ou troca de modelo.

## 1. Objetivo atual

Descobrir, em paper trading prospectivo e sem dinheiro real, se a hipótese encontrada no histórico produz lucro em jogos futuros.

O projeto não está autorizado a registrar apostas reais. O estado correto é:

- modo: `PAPER_ONLY`;
- apostas reais: `false`;
- auto-fill real: bloqueado;
- registro automático permitido: somente banca simulada paper;
- estratégia congelada: `TG_FT_O25`;
- ciclo: `PAPER_2026_07_TG_FT_O25_V1`;
- política: `1.1.0`;
- stake fixa contábil: R$ 5,00.

Não apagar nem importar novamente as previsões antigas. Elas são o histórico de descoberta, não a validação prospectiva.

## 2. O que foi encontrado na auditoria

Estado do histórico em 23/07/2026:

- 914 apostas simuladas;
- 911 liquidadas;
- 3 pendentes;
- 452 ganhos;
- 459 perdas;
- stake liquidada: R$ 4.555,00;
- lucro: R$ 137,55;
- ROI: 3,019759%.

Apesar do ROI histórico positivo, a conclusão ainda era inconclusiva:

- intervalo de 95% do ROI aproximadamente entre -3,26% e +9,32%;
- probabilidade bootstrap de ROI menor ou igual a zero: aproximadamente 17,18%;
- primeira metade do histórico: ROI aproximadamente +6,80%;
- segunda metade: ROI aproximadamente -0,75%;
- Brier do modelo: aproximadamente 0,2585;
- Brier de referência: aproximadamente 0,2500;
- Log Loss do modelo: aproximadamente 0,7111;
- Log Loss de referência: aproximadamente 0,6931;
- drawdown histórico aproximado: -24,73%.

Portanto, o histórico sugeriu uma hipótese, mas não provou lucro futuro.

### Pendências antigas preservadas

Estas não bloqueiam o paper:

1. 21/07/2026 — Santa Fe x U. Magdalena — `TG_FT_U45`;
2. 23/07/2026 — Leiknir x Throttur — `TG_FT_U35`;
3. 23/07/2026 — Emelec x Mushuc Runa — `R_FT_H`.

Elas continuam com `origem=backfill_simulado`. Podem ser resolvidas posteriormente pelo Flashscore ou revisão manual.

## 3. Separação obrigatória dos dados

O arquivo continua sendo:

`04_ml/banca/historico_apostas.csv`

Mas existem duas origens:

- `backfill_simulado`: previsões antigas usadas para descobrir a hipótese;
- `paper_forward`: previsões novas, capturadas prospectivamente.

Toda análise de outubro deve filtrar `origem=paper_forward` e o `paper_cycle_id` atual. Nunca misturar as 911 liquidadas antigas com o novo paper para concluir lucro.

## 4. Modelo paper congelado

Modelo principal:

- mercado: `TG_FT_O25`;
- algoritmo registrado: Random Forest calibrado por isotonic;
- threshold paper: 0,60;
- odds aceitas: 1,20 a 3,50;
- registry ID: `373d17dfc75e`;
- SHA-256 do modelo: `0e2a16987ea88f5899bd4bfe765bba85c4199bec4f80c2cc5b7c60979dacdb24`;
- SHA-256 dos especialistas contextuais: `c72416fb265eba27c33ed0acb91dda6da9f85c4bab363de9341b2150f5bba123`;
- SHA-256 do meta: `8d542ca65c79c546211033a423a6a17fbd6bd103f1be6337a63564c1b248294d`;
- SHA-256 das features: `911a69390a802adb4dad410e86cb59cf28b379e5920355e9aa7f885bb2929823`;
- SHA-256 dos encoders: `cd2aed8c71946b3a8adadf7b866414a3521ad24a47d1b1cef518371f624ddb23`.

Bundle:

`04_ml/paper_models/PAPER_2026_07_TG_FT_O25_V1`

Validação:

```powershell
python 04_ml\paper_model_manager.py validate
```

O backtest agregado do modelo não era positivo: o meta registrava ROI de aproximadamente -2,97%, e um resultado agregado observado do backtest estava por volta de -4,15%. Isso é uma razão adicional para não assumir lucratividade antes do paper.

## 5. Captura dinâmica implementada

O `daily` possui `Date` e `Time`. O predict transforma isso em `kickoff_at`.

Fluxo:

1. O `daily` é atualizado a cada 60 minutos.
2. A cada 15 minutos o scanner procura jogos que começam entre 30 e 90 minutos.
3. O modelo é executado somente quando há jogo nessa janela.
4. Cada jogo/mercado é capturado uma única vez.
5. A odd, probabilidade, horário de geração, kickoff, versão da política e hashes ficam registrados.
6. Uma captura antiga nunca é recalculada ou substituída.
7. Novos jogos do mesmo dia podem ser acrescentados ao CSV sem alterar linhas anteriores.
8. A banca paper registra apenas `paper_signal=true`.
9. O limite diário, por liga, por mercado e a exposição consideram tudo que já foi registrado em horários anteriores.
10. Nunca é criada uma entrada depois do início do jogo.

Se o notebook ficar suspenso durante toda a janela de 30–90 minutos, aquele jogo será perdido. Isso é intencional: não criar aposta retrospectiva.

Arquivos acumulados:

`04_ml/previsoes_paper/PAPER_2026_07_TG_FT_O25_V1/previsoes_AAAA-MM-DD.csv`

## 6. Settlement dinâmico

O controlador verifica a cada 15 minutos.

- primeira tentativa: `kickoff_at + 130 minutos`;
- novas tentativas: depois de 30, 60, 180 e 360 minutos;
- após esgotar a lista, continua usando intervalo de 360 minutos enquanto estiver pendente;
- tenta primeiro a base oficial;
- depois usa Flashscore somente para os índices paper que já estão no horário de settlement;
- ao resolver, reconstrói a banca e atualiza o monitor paper.

As varreduras gerais das 10:00 e 22:30 foram mantidas como rede de segurança para adiamentos, partidas sem ID e pendências antigas.

Estado das tentativas:

`04_ml/reports/dynamic_settlement_state.json`

## 7. Automação do Windows

Tarefas:

- `FootballDecisionLab-Paper-Automation`: chama o controlador a cada 15 minutos;
- `FootballDecisionLab-Dashboard`: mantém o dashboard disponível e ignora instâncias duplicadas.

Controlador:

`automation_orchestrator.py`

Configuração:

`04_ml/config/automation_schedule.json`

Rotinas:

| Rotina | Frequência |
|---|---:|
| Atualizar `daily` | 60 minutos |
| Procurar janela e capturar paper | 15 minutos |
| Settlement por kickoff | 15 minutos |
| Varredura geral de settlement | 10:00 e 22:30 |
| Monitor, alertas e relatórios | 23:15 |
| Base oficial incremental | terça-feira, 02:00 |
| Backtest + ML desafiante | dia 2 de cada mês, 03:00 |

O controlador possui:

- lock contra duplicidade;
- heartbeat;
- tentativas;
- estado por slot;
- logs;
- recuperação quando o notebook volta a ficar disponível.

Os quatro scripts antigos `auto_daily*.ps1` foram mantidos apenas como atalhos compatíveis e agora chamam o mesmo controlador.

## 8. Base, backtest, ML e calibração

Política adotada:

- `daily`: a cada hora, pois agenda e odds podem mudar;
- base oficial: semanal;
- backtest: mensal;
- treino de ML: mensal, em diretório desafiante;
- calibração: monitorada diariamente;
- recalibração do modelo paper: proibida durante o ciclo;
- promoção automática: proibida.

Desafiantes mensais:

`04_ml/challengers/AAAA-MM`

O backtest, datasets e modelos do desafiante são versionados. Eles não sobrescrevem o bundle paper. Um desafiante só pode virar novo paper depois de revisão e criação de outro `cycle_id`.

## 9. Dashboard

Endereços na última sessão:

- notebook: `http://127.0.0.1:8060`;
- celular na mesma rede: `http://192.168.0.227:8060`.

O IP local pode mudar depois de reiniciar o roteador.

Na aba Banca existe “Paper trading isolado”, com:

- liquidadas e pendentes;
- lucro e stake;
- ROI;
- intervalo bootstrap de 95%;
- Brier e erro de calibração;
- progresso para 300 liquidadas;
- mercado, liga e mês;
- saúde da automação;
- janela de captura e horário de settlement.

## 10. Critérios para outubro

Data mínima de revisão: 01/10/2026.

Não concluir apenas porque chegou outubro. Verificar também a amostra:

- menos de 30 liquidadas: apenas coleta;
- 30–99: amostra pequena;
- 100–299: diagnóstico inicial;
- 300 ou mais: primeira decisão estatística;
- 500 ou mais: leitura mais robusta.

Na revisão:

1. filtrar somente `paper_forward` e o ciclo atual;
2. conferir quantidade liquidada e pendências;
3. calcular ROI, lucro, win rate e drawdown;
4. conferir intervalo bootstrap de 95%;
5. conferir probabilidade de ROI positivo;
6. calcular Brier, Log Loss e erro de calibração;
7. comparar por mês, liga e faixa de odd;
8. verificar estabilidade temporal, especialmente primeira e segunda metade;
9. consultar os relatórios dos desafiantes mensais;
10. decidir entre continuar, encerrar ou iniciar novo ciclo.

Status interpretativo:

- `EVIDENCIA_POSITIVA`: IC95% do ROI totalmente acima de zero;
- `EVIDENCIA_NEGATIVA`: IC95% totalmente abaixo de zero;
- `INCONCLUSIVO`: o intervalo ainda inclui zero.

## 11. E-mail

O alerta local está implementado. O envio por e-mail não foi ativado porque não foram fornecidas credenciais SMTP.

Exemplo:

`04_ml/config/email.env.example`

Para Gmail, usar senha de app. Nunca salvar senha real no repositório ou neste caderno.

## 12. Primeiras verificações ao retomar

Executar apenas comandos de leitura inicialmente:

```powershell
python automation_orchestrator.py --status
python 04_ml\paper_model_manager.py validate
```

Abrir o dashboard e confirmar:

- modo `PAPER_ONLY`;
- saúde da automação `OK`;
- apostas reais `False`;
- ciclo e política corretos;
- heartbeat recente;
- capturas paper existentes;
- settlements ocorrendo.

Depois conferir:

```powershell
pytest tests -q
```

## 13. Pontos que ainda merecem atenção

1. Conferir manualmente as primeiras capturas para confirmar que `Time` do daily corresponde ao fuso `America/Sao_Paulo`.
2. Confirmar que a odd capturada no CSV corresponde à odd exibida perto do kickoff.
3. Observar por 48 horas se o scanner registra no máximo uma entrada por jogo.
4. Verificar as primeiras resoluções em `kickoff + 130 minutos`.
5. Configurar e-mail somente se desejado.
6. Resolver as três pendências antigas quando houver resultado confiável.
7. Em outubro, não promover automaticamente nenhum desafiante.

## 14. Arquivos principais criados ou alterados

- `automation_orchestrator.py`
- `install_automation.ps1`
- `uninstall_automation.ps1`
- `04_ml/config/paper_mode.json`
- `04_ml/config/automation_schedule.json`
- `04_ml/paper_model_manager.py`
- `04_ml/paper_predict.py`
- `04_ml/06_registrar_paper.py`
- `04_ml/dynamic_settlement.py`
- `04_ml/paper_monitor.py`
- `04_ml/paper_alerts.py`
- `04_ml/challenger_runner.py`
- `04_ml/05_settle_flashscore.py`
- `web_dashboard_lux/app.py`
- `web_dashboard_lux/templates/index.html`
- `web_dashboard_lux/static/app.js`
- `web_dashboard_lux/static/style.css`
- `OPERACAO_PAPER.md`
- `CADERNO_CONTINUIDADE.md`

## 15. Regra final

O sistema está automatizado para medir a estratégia, não para provar antecipadamente que ela dá lucro.

Preservar:

- modelo congelado;
- política congelada;
- origem dos dados;
- horários;
- odds;
- resultados;
- incerteza estatística.

Qualquer mudança de threshold, stake, faixa de odd, features, modelo ou calibração deve encerrar o ciclo atual e criar um novo ciclo identificável.

## 16. Identidade, GitHub e portfólio público — 23/07/2026

### Novo nome

O projeto passou a se chamar **Football Decision Lab**.

A pasta operacional foi migrada de:

`C:\Users\vinimedrado\Desktop\teste - lab football`

para:

`C:\Users\vinimedrado\Desktop\Football Decision Lab`

Os nomes visíveis do dashboard, manifesto PWA, documentação e tarefas automáticas foram atualizados. O desinstalador preserva compatibilidade para remover os nomes antigos.

Tarefas atuais:

- `FootballDecisionLab-Paper-Automation`;
- `FootballDecisionLab-Dashboard`.

A tarefa legada `FootballLab_Daily_Update`, que apontava para o caminho inexistente `C:\Users\vinimedrado\Desktop\football_lab`, foi removida na migração.

### Estratégia de publicação

O projeto operacional completo **não** foi enviado ao GitHub. Ele possui mais de 13 GB e contém datasets, modelos, históricos, caches e configuração privada.

Foi criada uma edição pública separada em:

`public_release/football-decision-lab`

Conteúdo público:

- portfólio estático;
- snapshot agregado e congelado;
- arquitetura e governança;
- contratos de referência e testes;
- configuração de deploy.

Conteúdo deliberadamente excluído:

- `.env` e chaves;
- `web_dashboard_lux/tools/api_football_key.txt`;
- dados raw/processados;
- históricos completos;
- modelos e calibradores;
- logs, backups e cache;
- comandos do dashboard operacional.

O repositório público passou por varredura de segredos e caminhos locais antes do primeiro push.

### Endereços

- GitHub: `https://github.com/vinmedrado/football-decision-lab`
- Portfólio principal: `https://football-decision-lab.netlify.app`
- Publicação conectada secundária: `https://football-decision-lab.medrado-jobs.chatgpt.site`

O GitHub aponta o campo `homepage` para a Netlify.

### Snapshot demonstrativo

O portfólio não consulta o notebook nem recebe atualizações automáticas. Isso é intencional: recrutadores veem uma demonstração estável, sem exposição do backend local.

O snapshot de 23/07/2026 mostra:

- 914 registros de backfill simulado;
- 911 liquidados e 3 pendentes;
- ROI histórico de 3,0198%;
- IC 95% de -3,2586% a 9,3234%;
- drawdown de -24,7337%;
- Brier do modelo pior que o baseline;
- paper prospectivo iniciado com zero liquidados.

O site deixa explícito que esses números não provam lucro real.

Para atualizar o portfólio no futuro:

1. revisar e anonimizar os novos agregados;
2. alterar apenas `site/data/snapshot.json`;
3. executar `npm run build`;
4. executar os testes;
5. revisar segredos e caminhos;
6. criar novo commit;
7. publicar uma nova versão.

Nunca conectar o site público diretamente aos CSVs operacionais.

### Validações feitas

- JavaScript e JSON validados;
- build estático sem dependências externas;
- Worker da hospedagem validado localmente;
- 3 testes do contrato público aprovados;
- 18 testes do sistema operacional aprovados;
- renderização desktop e responsiva inspecionada;
- produção Netlify respondeu HTTP 200;
- CSP, `X-Frame-Options` e demais cabeçalhos de segurança confirmados;
- repositório GitHub confirmado como público.

## 17. Simplificação do dashboard — 23/07/2026

O dashboard local foi revisado para reduzir termos em inglês sem alterar nomes internos, rotas da API ou comandos da automação.

Principais traduções visíveis:

- `paper trading` → simulação ao vivo;
- `settlement` → atualizar resultados;
- `registry` → catálogo de modelos;
- `snapshot paper` → gerar previsões da simulação;
- `win rate` → taxa de acerto;
- `drawdown` → maior queda;
- `stake` → valor ou total simulado;
- `backtest` → teste histórico;
- `logs` → registros técnicos;
- `odd` → cotação;
- `edge`/`EV` → vantagem estimada.

As siglas estatísticas importantes foram preservadas entre parênteses para facilitar futuras auditorias:

- Retorno (`ROI`);
- Vantagem estimada (`EV`);
- Qualidade de separação (`AUC`);
- Erro de probabilidade (`Brier`).

Foi criada, na aba **Sistema**, a seção **Ajuda rápida — O que significam os indicadores**, com explicações curtas desses conceitos.

Durante a execução de rotinas, nomes de arquivos como `paper_predict.py` agora são apresentados de forma humana, por exemplo: **Gerando previsões da simulação**.

Arquivos alterados:

- `web_dashboard_lux/app.py`;
- `web_dashboard_lux/templates/index.html`;
- `web_dashboard_lux/static/app.js`;
- `web_dashboard_lux/static/style.css`.

Validação após a alteração:

- JavaScript válido;
- Python válido;
- 18 testes aprovados;
- dashboard HTTP 200;
- saúde 100;
- automação `OK`;
- apostas reais continuam `False`;
- processo observado estável após o reinício.

## 18. Separação entre previsões antigas e indicações oficiais — 23/07/2026

O dashboard deixou de transformar automaticamente um arquivo legado em sinal do
ciclo paper quando ainda não existe uma previsão paper para a data.

Nova regra:

- somente arquivos dentro de `04_ml/previsoes_paper/<cycle_id>/` podem gerar
  **indicações oficiais**;
- arquivos de `04_ml/previsoes/` e do histórico continuam visíveis para consulta,
  mas aparecem como **pré-ciclo — não é indicação**;
- marcações antigas em `apostar=True` não entram na contagem, no destaque nem nos
  alertas do ciclo atual;
- o painel informa quantas marcações antigas foram ignoradas e mostra o mercado
  oficial configurado para o ciclo.

Para 23/07/2026, as 276 previsões e os 12 jogos permanecem consultáveis, mas as
duas marcações antigas (`TG_FT_U35` e `R_FT_H`) agora resultam em zero indicações
oficiais. O ciclo atual continua restrito a `TG_FT_O25`.

### Simplificação adicional da tela

A aba antes chamada **Jogos e sinais** passou a se chamar **Apostas aprovadas**.
Ela exibe exclusivamente linhas aprovadas e oficiais do ciclo paper atual.

As previsões rejeitadas, análises de outros mercados e arquivos pré-ciclo não
foram apagados: continuam armazenados para auditoria, porém não aparecem mais
como cartões no dashboard. Quando não há entrada oficial, a tela informa
claramente **Nenhuma aposta aprovada para esta data**.

## 19. Execução silenciosa em segundo plano — 23/07/2026

As tarefas do dashboard e da automação foram configuradas para usar
`pythonw.exe`, a variante do Python para Windows que não abre console.
Como o `pythonw.exe` não oferece saída padrão, ele chama
`scripts/run_headless.py`, que redireciona mensagens e exceções para
`logs/headless/`. Isso também evita que o servidor encerre por tentar escrever
em um console inexistente.

Comportamento esperado:

- nenhuma janela de terminal precisa ficar aberta;
- o dashboard local continua disponível na porta `8060`;
- o controlador continua verificando as rotinas a cada 15 minutos;
- a recuperação após desligamento ou falta de energia permanece ativa;
- saídas das rotinas continuam registradas em `logs/automation/`;
- inicialização e falhas dos processos silenciosos ficam em `logs/headless/`;
- erros e heartbeat continuam registrados nos relatórios operacionais.

O instalador `install_automation.ps1` agora localiza obrigatoriamente o
`pythonw.exe` ao lado do interpretador Python e falha com mensagem clara se ele
não estiver disponível, evitando voltar silenciosamente ao modo com console.

## 20. Documentação final — 23/07/2026

Foram revisados três níveis de documentação:

- `README.md`: criado na raiz como manual operacional do projeto privado;
- `web_dashboard_lux/README.md`: atualizado para o dashboard paper atual,
  execução silenciosa e exibição exclusiva de apostas aprovadas;
- `public_release/football-decision-lab/README.md`: reescrito para apresentar
  arquitetura, controles, snapshot auditado e limitações aos recrutadores.

O README público foi publicado no GitHub no commit `f111a13`. O build estático
foi validado e os 3 testes do contrato público foram aprovados. A Netlify não
precisou de nova publicação porque apenas a documentação do repositório mudou.

## 21. Incidente de registro paper e correção — 27/07/2026

Entre 24 e 27/07, os snapshots prospectivos gravaram sinais booleanos como
`1.0` e `0.0` depois da concatenação com linhas anteriores. O gerador aprovou
40 sinais, mas o registrador e o dashboard reconheciam apenas `true` ou `1`.
Por isso, nenhuma entrada chegou inicialmente à banca paper e o atualizador de
resultados não tinha apostas para liquidar.

Correções:

- leitores booleanos agora aceitam representação numérica equivalente a 1;
- o dashboard carrega os dados completos automaticamente na primeira abertura;
- o estado técnico completo dos jobs deixou de ser enviado ao celular;
- a contagem de linhas da base passou a usar cache por assinatura do arquivo;
- o scanner atualiza o monitor paper logo após registrar novas apostas;
- o monitor noturno valida o bundle paper congelado, em vez de falhar porque
  não existe modelo autorizado para dinheiro real;
- “calibrador ausente” foi reescrito para deixar claro que se trata apenas de
  calibrador externo adicional; o modelo congelado já é calibrado internamente.

Recuperação do incidente:

- somente capturas imutáveis com `paper_generated_at < kickoff_at` foram aceitas;
- limite máximo de 10 apostas por dia foi preservado;
- 33 entradas foram recuperadas: 8 de 24/07, 10 de 25/07, 10 de 26/07 e 5 de
  27/07;
- todas receberam `paper_capture_was_prospective=true`,
  `paper_registration_recovered=true` e motivo explícito do incidente;
- quatro backups do histórico foram criados antes das inclusões;
- após a primeira atualização, 28 foram liquidadas: 20 ganhos, 8 perdas,
  lucro paper de R$ 5,30 e 5 pendentes naquele momento;
- a automação seguinte registrou normalmente um novo sinal futuro, confirmando
  o funcionamento do fluxo corrigido.

Validação:

- 23 testes automatizados aprovados;
- bundle paper validado;
- dashboard respondeu HTTP 200 em `http://192.168.0.227:8060`;
- resposta principal reduzida de aproximadamente 215 KB para 50 KB;
- resposta aquecida observada em aproximadamente 0,4 segundo;
- apostas reais continuam bloqueadas.
