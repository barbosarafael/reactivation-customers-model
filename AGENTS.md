# AGENTS.md

## Visao do projeto

Este repositorio implementa um pipeline de dados e ML no Databricks para
priorizar clientes inativos com probabilidade de realizar uma nova compra nos
proximos 30 dias. O dataset de referencia e o Online Retail II; o caso de uso
simula reativacao de clientes/apostadores.

O repositorio Git e a fonte de verdade. Desenvolva localmente e use Databricks
para executar, validar e operar o pipeline. Nao edite artefatos somente no
Databricks Workspace quando a alteracao tambem precisar existir no Git

## Estrutura relevante

- `src/reactivation_model/`: codigo Python reutilizavel. Mantenha a logica de
  dominio e transformacoes testaveis aqui quando elas nao forem especificas de
  um notebook.
- `conf/dev.yml`: configuracao do ambiente de desenvolvimento, incluindo
  catalog, schema, nomes logicos de tabelas, parametros de modelagem e MLflow.
- `notebooks/`: notebooks Databricks serializados como `.py`. Os notebooks
  `06a`, `07`, `08` e `09` suportam o fluxo operacional de dados sinteticos,
  scoring, monitoramento e explicabilidade.
- `resources/reactivation_job.yml`: definicao do Job do Databricks Asset Bundle.
- `databricks.yml`: configuracao principal do bundle; hoje existe somente o
  target `dev`, em `mode: development`.
- `docs/problem_definition.md`: contrato de negocio e modelagem. Consulte-o
  antes de alterar target, features, populacao elegivel ou avaliacao.
- `tests/`: testes locais com `pytest`. Amplie esta pasta ao extrair ou alterar
  logica Python reutilizavel.

## Contrato de modelagem

- A unidade de analise e `customer_id + reference_date`.
- Um cliente e elegivel quando `inactive_days >= 60`.
- Features usam somente a janela de observacao de 180 dias ate
  `reference_date`, inclusive.
- A target `target_reactivated_30d` usa somente compras posteriores a
  `reference_date`, na janela de 30 dias.
- Treino, validacao e teste devem respeitar a ordem temporal. Nao use split
  aleatorio como estrategia principal.
- Avalie uma baseline simples antes de aumentar a complexidade do modelo.
- Priorize metricas de ranking e negocio (`precision_at_top_*`,
  `lift_at_top_*`) junto a ROC AUC e PR AUC. Acuracia isolada nao e criterio
  suficiente para este caso de uso.

Qualquer alteracao de feature, target ou janela temporal deve documentar como
evita leakage no instante real de scoring. Transformacoes de treinamento e
inferencia devem permanecer equivalentes.

## Dados e Delta Lake

- Preserve a progressao bronze -> silver -> gold e mantenha schemas e chaves
  explicitamente definidos.
- Para cargas incrementais, defina chave natural, deduplicacao, tratamento de
  eventos atrasados e idempotencia antes de implementar.
- Prefira funcoes nativas de PySpark e Spark SQL. Evite Python UDFs, `collect()`
  em dados de producao e conversoes para Pandas fora de amostras pequenas e
  controladas.
- Trate alteracoes de schema como contrato: valide colunas obrigatorias, tipos,
  nulidade e comportamento esperado para schema drift.
- Mantenha verificacoes de qualidade que falhem de forma explicita quando
  bloquearem o resultado do pipeline. `DataQualityError` em
  `src/reactivation_model/data_quality.py` e o padrao atual para esse caso.

## Configuracao e nomes de tabelas

- Centralize novos parametros em `conf/<ambiente>.yml`; nao espalhe valores de
  ambiente, nomes de catalog/schema, janelas ou limiares no codigo.
- Use os helpers de `src/reactivation_model/config.py` para resolver nomes de
  tabelas e configuracoes em codigo Python reutilizavel.
- Nao inclua credenciais, tokens, hosts privados ou dados pessoais em codigo,
  configuracoes ou logs. Use secrets e configuracao de ambiente no Databricks.
- Ha uma divergencia atual que merece cuidado: `conf/dev.yml` usa o schema
  `workspace.bettor_crm_ml_dev`, enquanto os notebooks operacionais contem
  nomes hardcoded como `workspace.synthetic_layer` e `workspace.gold_layer`.
  Antes de modificar ou executar o fluxo, verifique quais tabelas e schemas
  realmente existem. Em uma mudanca que toque esse contrato, alinhe os nomes
  de forma intencional e atualize todos os consumidores afetados.

## Notebooks Databricks

- Preserve os marcadores `# Databricks notebook source` e
  `# COMMAND ----------`; eles sao necessarios para importacao e sincronizacao
  corretas com o Workspace.
- Mantenha parametros de execucao em widgets e valide seus valores antes de
  processar dados.
- Um notebook deve orquestrar e apresentar o fluxo; extraia logica complexa ou
  reutilizada para `src/reactivation_model/` e cubra-a com testes locais.
- Nao instale dependencias repetidamente em varios notebooks sem necessidade.
  Registre uma nova dependencia em `requirements.txt` e confirme a
  compatibilidade com o Databricks Runtime ou environment configurado.

## MLflow, scoring e monitoramento

- Registre parametros, metricas, versao de codigo/configuracao e artefatos de
  treinamento no MLflow.
- Nao mova um modelo para o alias `champion` sem comparar com a baseline e o
  modelo vigente usando uma janela temporal de validacao definida.
- O batch scoring deve registrar data de scoring, identificador do modelo,
  versao/alias, score, decisao derivada e timestamp de execucao para auditoria.
- Mudancas no scoring devem atualizar as verificacoes de monitoramento quando
  alterarem features, distribuicao de score, target ou populacao elegivel.
- Dados sinteticos servem para testar a operacao e cenarios de drift; eles nao
  comprovam performance real do modelo.

## Qualidade de codigo e testes

- Use Python com nomes claros, type hints quando ajudarem a interface e funcoes
  pequenas. Mantenha docstrings nos modulos publicos e comentarios apenas para
  decisoes nao obvias.
- Adicione testes `pytest` para novos helpers, validacoes, regras temporais e
  transformacoes deterministicas. Use entradas minimas que exponham o caso de
  sucesso e a falha protegida.
- Para codigo Spark, prefira testes focados de transformacao/schema em vez de
  testar detalhes de implementacao. Nao exija um cluster para validar logica
  Python pura.
- Nao reformate ou reorganize arquivos sem relacao com a alteracao solicitada.
  Preserve mudancas locais preexistentes.

## Fluxo de desenvolvimento

1. Leia o contrato em `docs/problem_definition.md` e a configuracao em
   `conf/dev.yml` antes de alterar dados ou modelagem.
2. Implemente primeiro em `src/` quando a logica for reutilizavel; atualize o
   notebook ou recurso do bundle que a consome.
3. Execute os testes locais aplicaveis:
   ```bash
   pytest
   ```
4. Valide a configuracao do bundle no ambiente de desenvolvimento:
   ```bash
   databricks bundle validate -t dev
   ```
5. Faca deploy somente no target correto:
   ```bash
   databricks bundle deploy -t dev
   ```
6. Execute o recurso explicitamente e revise logs, tabelas Delta, metricas e
   artefatos MLflow antes de promover qualquer mudanca:
   ```bash
   databricks bundle run reactivation_customers_pipeline -t dev
   ```

Nao suponha que comandos Databricks, schemas, modelos registrados ou tabelas
existam: valide-os no ambiente alvo. Evite alterar producao ate que exista um
target separado, CI basico e uma estrategia de promocao/rollback.

## Criterios de aceite para mudancas relevantes

- Sem leakage temporal introduzido.
- Configuracao, consumidores e nomes de tabela permanecem coerentes.
- Testes locais relevantes passam ou a limitacao e registrada.
- O bundle valida no target afetado.
- O Job preserva dependencias entre tarefas, retries e idempotencia esperada.
- Scoring e monitoramento mantem rastreabilidade suficiente para auditoria e
  diagnostico.
