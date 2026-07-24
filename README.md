# Football Decision Lab

Sistema de apoio à decisão para mercados de futebol, desenvolvido para
transformar dados de partidas em previsões reproduzíveis e avaliar uma
estratégia em um experimento prospectivo controlado.

[Ver demonstração pública](https://football-decision-lab.netlify.app)

> Estado em 23/07/2026: infraestrutura operacional concluída, ciclo paper
> iniciado e apostas reais bloqueadas.

## O que o projeto demonstra

- pipeline de dados com validação e execução repetível;
- backtest com separação temporal;
- modelos probabilísticos versionados;
- ciclo paper isolado do histórico;
- captura pré-jogo imutável;
- atualização automática de resultados;
- monitoramento de ROI, risco e calibração;
- automação retomável após desligamento;
- dashboard local e portfólio público desacoplados.

O objetivo não é apresentar um “robô de apostas”, mas demonstrar como avaliar
uma hipótese quantitativa sem confundir desempenho histórico com evidência
prospectiva.

## Problema de engenharia

Um backtest positivo não garante resultado futuro. Para reduzir vieses, o
Football Decision Lab separa:

1. desempenho histórico simulado;
2. qualidade probabilística do modelo;
3. resultado prospectivo coletado antes dos jogos.

As previsões rejeitadas continuam disponíveis na auditoria privada, enquanto o
dashboard operacional exibe somente entradas aprovadas pelo ciclo atual. Dessa
forma, uma marcação antiga não pode aparecer como nova indicação paper.

## Arquitetura resumida

```text
Coleta → Validação → Features → Backtest temporal → Modelo congelado
                                                        ↓
Monitoramento ← Settlement ← Banca paper ← Captura pré-jogo
```

Controles principais:

- modo `PAPER_ONLY`;
- apostas reais desativadas;
- modelo e artefatos identificados por SHA-256;
- previsões imutáveis após a captura;
- janela operacional de 30 a 90 minutos antes do jogo;
- tarefas idempotentes com estado persistido;
- settlement com novas tentativas para partidas pendentes;
- treinamento de challengers sem promoção automática;
- decisão estatística adiada até uma amostra suficiente.

## Estado do ciclo prospectivo

| Item | Configuração |
|---|---|
| Ciclo | `PAPER_2026_07_TG_FT_O25_V1` |
| Mercado | Mais de 2,5 gols (`TG_FT_O25`) |
| Valor simulado | R$ 5,00 |
| Máximo diário | 10 entradas |
| Início | 23/07/2026 |
| Primeira revisão | Não antes de 01/10/2026 |
| Meta de decisão | 300 apostas liquidadas |
| Apostas reais | Desativadas |

No ambiente privado, o Agendador do Windows executa o controlador e o dashboard
silenciosamente com `pythonw.exe`. Jogos, captura, liquidação, monitoramento e
treino de challengers têm calendários independentes. Janelas pré-jogo perdidas
não são recriadas depois da partida.

## Snapshot histórico auditado

Os dados abaixo pertencem a um backfill simulado, não a apostas reais nem ao
ciclo prospectivo atual:

| Métrica | Valor |
|---|---:|
| Registros | 914 |
| Liquidados | 911 |
| Pendentes | 3 |
| ROI sobre valor simulado | 3,02% |
| Lucro simulado | R$ 137,55 |
| IC 95% do ROI por dia | -3,26% a 9,32% |
| Probabilidade estimada de ROI ≤ 0 | 17,18% |
| Maior queda | -24,73% |
| Brier do modelo | 0,2585 |
| Brier baseline | 0,2500 |

Leitura correta: o histórico foi positivo, mas o intervalo de incerteza inclui
zero e o Brier não superou o baseline. Por isso, esses números não são tratados
como comprovação de lucro.

## Tecnologias e decisões

- Python, pandas e scikit-learn no pipeline analítico;
- FastAPI no dashboard operacional privado;
- JavaScript e CSS sem framework no portfólio;
- tarefas agendadas e execução headless no Windows;
- testes de regressão para contratos paper e segurança;
- arquivos estáticos com CSP e cabeçalhos restritivos na publicação.

## Por que este repositório é uma edição pública

O ambiente operacional contém dados locais, modelos serializados, logs e
configurações privadas. Esta edição publica somente:

- o portfólio estático e responsivo;
- um snapshot demonstrativo e imutável;
- documentação de arquitetura, governança e segurança;
- contratos de referência que explicam o desenho do ciclo paper.

Não são publicados dados brutos, modelos, credenciais, arquivos `.env` ou
rotinas capazes de acionar o notebook.

## Executar o portfólio

Requisito: Node.js 18 ou superior.

```powershell
npm run build
python -m http.server 4173 --directory dist
```

Abra `http://127.0.0.1:4173`.

Também é possível servir diretamente a pasta `site` sem gerar o build:

```powershell
python -m http.server 4173 --directory site
```

## Estrutura pública

```text
.
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GOVERNANCE.md
│   └── SECURITY.md
├── reference/
│   ├── paper_contract.py
│   └── test_paper_contract.py
├── scripts/
│   └── build-static.mjs
├── site/
│   ├── data/snapshot.json
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── netlify.toml
└── README.md
```

## Documentação

- [Arquitetura](docs/ARCHITECTURE.md)
- [Governança](docs/GOVERNANCE.md)
- [Segurança](docs/SECURITY.md)

## Aviso

Este é um projeto educacional e de engenharia de software. Resultados simulados
não garantem desempenho futuro e não constituem recomendação financeira.
