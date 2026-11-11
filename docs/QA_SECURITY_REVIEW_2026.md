# QA e limites da validação — 11/11/2026

**Escopo:** CI sem credenciais externas, com Python 3.11, teste de regressão e build estático Node 22. A operação do laboratório permanece exclusivamente **PAPER_ONLY** e não foi ativada nenhuma execução com dinheiro real.

## Evidências
- **23 testes aprovados, 1 skip** no checkout público, além do build do portfólio estático.
- A instalação do dashboard FastAPI/Jinja2 é declarada explicitamente no workflow.
- O teste de previsões históricas usa CSV sintético para confirmar que registros legados nunca se tornam sinais oficiais do ciclo paper.
- O teste de hashing verifica integridade contra adulteração de um artefato sintético e mantém a validação original quando o bundle de modelo congelado estiver disponível.

## Limitações
- Os modelos binários congelados são intencionalmente excluídos do Git e **não** foram validados no CI: o teste específico é skipped sem eles.
- A execução real de coleta, dados externos, agenda agendada e os próprios modelos congelados dependem do ambiente operacional e não foram verificados.
- O projeto é um experimento de pesquisa/paper-trading, não um serviço de apostas nem evidência de performance financeira futura.
- Dependências Python possuem limites mínimos abertos; o CI não substitui auditoria de supply chain, segredos ou testes de produção.
