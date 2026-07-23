# Football Decision Lab

Sistema de apoio à decisão para mercados de futebol, criado para transformar dados de partidas em previsões reproduzíveis e avaliá-las em um ciclo prospectivo de paper trading.

> Estado do projeto em 23/07/2026: infraestrutura operacional pronta e experimento prospectivo iniciado. Nenhuma aposta real é executada.

**Demo pública:** [football-decision-lab.netlify.app](https://football-decision-lab.netlify.app)

## Por que este repositório é uma edição pública

O projeto operacional contém bases locais, artefatos de modelos e configurações privadas. Este repositório publica apenas:

- o portfólio estático e responsivo;
- um snapshot demonstrativo, imutável e sem credenciais;
- a arquitetura e as decisões de engenharia;
- contratos de referência que explicam o desenho do sistema.

Dados brutos, modelos serializados, chaves, arquivos `.env`, logs e rotinas capazes de acionar o ambiente local não fazem parte da edição pública.

## O problema

Um backtest positivo não prova que uma estratégia continuará positiva. O Football Decision Lab separa três coisas que costumam ser misturadas:

1. desempenho histórico simulado;
2. qualidade probabilística do modelo;
3. resultado prospectivo, coletado sem alterar previsões depois do jogo.

Essa separação permite avaliar retorno, risco, calibração e estabilidade sem transformar uma simulação em alegação de lucro real.

## Arquitetura resumida

```text
Coleta → Validação → Features → Backtest temporal → Modelo congelado
                                                      ↓
Monitoramento ← Settlement ← Paper ledger ← Captura pré-jogo
```

Principais controles:

- execução `PAPER_ONLY`;
- previsões imutáveis após a captura;
- modelo identificado por hashes SHA-256;
- tarefas idempotentes e retomada após desligamento;
- settlement com tentativas progressivas;
- challenger pode ser treinado, mas nunca promovido automaticamente;
- decisão adiada até amostra prospectiva suficiente.

## Snapshot auditado

Os números abaixo pertencem a um backfill simulado, não a apostas reais:

| Métrica | Valor |
|---|---:|
| Registros | 914 |
| Liquidados | 911 |
| Pendentes | 3 |
| ROI sobre stake | 3,02% |
| Lucro simulado | R$ 137,55 |
| IC 95% do ROI por dia | -3,26% a 9,32% |
| Probabilidade estimada de ROI ≤ 0 | 17,18% |
| Maior drawdown | -24,73% |
| Brier do modelo | 0,2585 |
| Brier baseline | 0,2500 |

Leitura correta: o histórico foi positivo, mas a incerteza ainda inclui zero e a calibração não superou o baseline. Por isso o sistema está coletando uma amostra prospectiva.

## Executar o portfólio

Não há instalação nem backend:

```powershell
python -m http.server 4173 --directory site
```

Abra `http://127.0.0.1:4173`.

## Publicar na Netlify

O arquivo `netlify.toml` já aponta para a pasta `site`. Ao conectar este repositório na Netlify:

- comando de build: vazio;
- diretório publicado: `site`;
- variáveis de ambiente: nenhuma.

## Estrutura

```text
.
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GOVERNANCE.md
│   └── SECURITY.md
├── reference/
│   ├── paper_contract.py
│   └── test_paper_contract.py
├── site/
│   ├── data/snapshot.json
│   ├── app.js
│   ├── index.html
│   └── styles.css
└── netlify.toml
```

## Aviso

Este projeto é educacional e de engenharia de software. Resultados simulados não garantem desempenho futuro e não constituem recomendação financeira.
