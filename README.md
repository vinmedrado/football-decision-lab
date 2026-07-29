# Football Decision Lab

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-111827?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Sistema de apoio a decisao para mercados de futebol, desenvolvido para transformar dados de partidas em previsoes reproduziveis e avaliar uma estrategia em um experimento prospectivo controlado.

[Ver demonstracao publica](https://football-decision-lab.netlify.app)

## Visao geral

O projeto evolui um fluxo de pesquisa e paper trading: coleta, validacao temporal, feature engineering, backtest, modelo congelado, captura pre-jogo e settlement automatico.

## Problema que resolve

- Evitar data leakage em analises esportivas.
- Separar backtest historico de evidencias prospectivas.
- Manter rastreabilidade das decisoes e do modelo usado.

## Arquitetura

```text
Coleta -> Validacao -> Features -> Backtest temporal -> Modelo congelado
                                            |
                                            v
                           Captura pre-jogo -> Settlement -> Monitoramento
```

## Screenshots

![Portfolio screenshot](https://raw.githubusercontent.com/vinmedrado/portfolio/main/images/footballdecisionlab.png)

## Funcionalidades

- Pipeline de dados com validacao e execucao repetivel.
- Backtest com separacao temporal.
- Modelos probabilisticos versionados.
- Ciclo paper isolado do historico.
- Captura pre-jogo imutavel.
- Atualizacao automatica de resultados.
- Monitoramento de ROI, risco e calibracao.
- Guardrails fail-closed.

## Tecnologias

Python, Pandas, Scikit-learn, XGBoost, LightGBM, Netlify, JavaScript, CSS.

## Como executar

```bash
npm run build
python -m http.server 4173 --directory dist
```

Abra `http://127.0.0.1:4173`.

## Estrutura do projeto

```text
docs/         arquitetura, governanca e seguranca
reference/    contratos e testes de referencia
scripts/      build do site estatico
site/        portfolio publico
```

## Roadmap

- Adicionar mais evidencias visuais no README.
- Evoluir o conteudo de monitoramento e calibracao.
- Consolidar a documentacao de experimentos prospectivos.

## Licenca

TODO.
