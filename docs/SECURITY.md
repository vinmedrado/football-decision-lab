# Segurança da edição pública

## Conteúdo deliberadamente excluído

- chaves de APIs e credenciais SMTP;
- arquivos `.env`;
- bases raw, processadas ou históricas;
- modelos `.pkl`, `.joblib` e artefatos equivalentes;
- logs, backups e cache;
- comandos de controle remoto do notebook;
- URLs ou identificadores locais sensíveis.

## Modelo de publicação

O site é totalmente estático. Ele não expõe o backend operacional, não executa tarefas no notebook e não aceita comandos. O arquivo `snapshot.json` contém somente números agregados ou exemplos demonstrativos.

## Cabeçalhos

`netlify.toml` configura CSP, bloqueio de framing, política de referência e bloqueio de sensores. Como o portfólio não usa scripts, fontes ou APIs externas, a política de conteúdo pode permanecer restritiva.

## Regra de contribuição

Antes de qualquer push:

1. verificar o diff completo;
2. procurar nomes de arquivos de segredo;
3. procurar padrões de tokens;
4. confirmar que nenhum dataset ou modelo foi adicionado;
5. executar os testes e a validação do snapshot.
