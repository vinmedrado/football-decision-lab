# Settlement automático pelo Flashscore

O projeto já salva o ID do Flashscore nos arquivos de jogos diários. O
`05_settle_flashscore.py` usa esse ID para consultar a página pública mobile da
partida e recuperar:

- times;
- status;
- data real em que a partida aconteceu;
- placar do primeiro tempo;
- placar final no tempo regulamentar.

Ele não procura um jogo apenas pelo texto dos times. Antes de propor qualquer
resultado, confirma o ID local, os dois times, o status final e o placar.

## Uso recomendado

Primeiro gere uma prévia pequena:

```powershell
python 04_ml/05_settle_flashscore.py --limit 5
```

Esse comando consulta no máximo cinco pendências, não altera o histórico e
gera:

```text
04_ml/banca/flashscore_settlement_preview.csv
```

Se a prévia estiver correta, consulte todas as pendências, ainda sem gravar:

```powershell
python 04_ml/05_settle_flashscore.py
```

Para aplicar somente os resultados classificados como
`PRONTO_PARA_APLICAR`:

```powershell
python 04_ml/05_settle_flashscore.py --apply
```

No modo `--apply`, o script:

1. cria um backup datado de `historico_apostas.csv`;
2. atualiza somente partidas finalizadas e validadas;
3. registra URL, ID, data real, placares e horário do settlement;
4. reconstrói a banca para manter `banca_apos` e o estado consistentes.

## Casos que não são alterados automaticamente

- partida ainda adiada ou agendada;
- cancelada, abandonada ou interrompida;
- ID não encontrado no CSV diário;
- nomes dos times divergentes;
- página incompleta;
- placar regulamentar ausente;
- mercado não suportado.

Esses casos permanecem como `pendente` e aparecem no relatório com
`REVISAO_MANUAL`, `MANTER_PENDENTE` ou `TENTAR_NOVAMENTE`.

## Cache e retomada

Cada resultado consultado é salvo em:

```text
04_ml/banca/flashscore_cache/
```

Partidas finalizadas são reutilizadas nas próximas execuções. Isso permite
interromper e retomar o processo sem consultar novamente os jogos já
resolvidos. Para ignorar o cache:

```powershell
python 04_ml/05_settle_flashscore.py --refresh-cache
```

O intervalo padrão entre acessos é de 1,25 segundo. Ele pode ser aumentado:

```powershell
python 04_ml/05_settle_flashscore.py --delay 2
```

## Filtros úteis

Uma data original:

```powershell
python 04_ml/05_settle_flashscore.py --date 2026-01-04
```

Uma quantidade limitada:

```powershell
python 04_ml/05_settle_flashscore.py --limit 10
```

O campo `data` original da previsão nunca é substituído. Quando um jogo foi
adiado, a nova data é registrada separadamente em `data_realizacao`.

