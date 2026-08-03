# Dashboard local — Football Decision Lab

Painel responsivo para acompanhar o ciclo paper do notebook ou de um celular
conectado à mesma rede Wi-Fi.

## O que aparece no painel

- somente apostas oficialmente aprovadas pelo ciclo paper atual;
- saldo, lucro e ROI da banca simulada;
- taxa de acerto, maior queda e resultados pendentes;
- progresso da amostra e intervalo de confiança;
- Brier, calibração e saúde da automação;
- modelos, mercados, logs e rotinas operacionais.

Previsões rejeitadas e arquivos pré-ciclo não são exibidos como apostas. Eles
continuam armazenados nos diretórios privados para auditoria.

## Instalação

Na raiz do projeto:

```powershell
pip install -r web_dashboard_lux\requirements.txt
```

## Execução recomendada

O dashboard é instalado como tarefa silenciosa do Windows junto com a automação:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_automation.ps1
```

Ele utiliza `pythonw.exe`, portanto nenhuma janela de terminal precisa ficar
aberta. Inicialização e erros ficam em `logs/headless/dashboard.log`.

Para uma execução manual:

```powershell
python web_dashboard_lux\app.py
```

## Endereços

- Notebook: `http://127.0.0.1:8060`
- Celular: `http://IP_DO_NOTEBOOK:8060`

O painel usa `0.0.0.0` apenas para acesso dentro da rede local. Não faça
redirecionamento da porta `8060` no roteador e não exponha o servidor diretamente
na internet.

## Segurança operacional

- comandos do navegador são limitados a rotinas previamente definidas;
- não existe terminal remoto ou execução de comandos livres;
- o modo atual é `PAPER_ONLY`;
- apostas reais permanecem desativadas;
- o portfólio público não se conecta ao dashboard local.

## Recursos visuais

Escudos e logos locais ficam em:

```text
web_dashboard_lux/static/crests/
web_dashboard_lux/static/leagues/
```

Quando uma imagem não existe, o painel utiliza as iniciais do time ou da liga.
