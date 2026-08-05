# Reactivation Next Purchase

Projeto end-to-end de dados e Machine Learning no Databricks para priorizar clientes inativos com maior probabilidade de realizar uma nova compra nos próximos 30 dias.

## Comece aqui: reproduzir no Databricks Free Edition

**É possível reproduzir este projeto para estudo no Databricks Free Edition**, mas ele não é um projeto de "clone, deploy e execute" ainda. A Free Edition oferece compute serverless, Jobs e um SQL Warehouse pequeno, que são suficientes para este caso educacional. Há cotas de uso e acesso limitado à internet; se o download da UCI não funcionar dentro do workspace, baixe o arquivo no seu computador e faça upload para o Databricks. Consulte as [características da Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition) e suas [limitações atuais](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations) antes de começar.

> Este projeto usa dados públicos e é destinado a estudo. Não envie dados de clientes, credenciais ou outros dados sensíveis para uma conta Free Edition.

### O que você precisa ter

1. Uma conta no [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition) e acesso ao seu workspace.
2. Git e Python instalados na sua máquina para clonar o repositório e executar os testes locais.
3. A Databricks CLI instalada e autenticada no **seu** workspace. A autenticação interativa pode ser feita com:

   ```bash
   databricks auth login --host https://<url-do-seu-workspace>
   ```

   Substitua o valor entre `<...>` pela URL exibida no navegador ao abrir seu workspace. Esse comando abre o login no navegador; não crie nem cole tokens no repositório. Veja a [documentação de autenticação da CLI](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).

4. Um SQL Warehouse iniciado no seu workspace. A Free Edition permite um Warehouse de até `2X-Small`; ele é necessário somente para publicar e atualizar o dashboard.

### Entenda o que é específico de cada workspace

O repositório contém valores que funcionam no workspace em que ele foi desenvolvido, mas **não** no seu automaticamente:

- o `host` e o `sql_warehouse_id` em `databricks.yml`;
- os objetos do catálogo `workspace`, como `workspace.bronze_layer` e `workspace.gold_layer`;
- o experimento MLflow, o modelo registrado e seu alias `champion`.

Antes de fazer deploy, altere em sua cópia local de `databricks.yml` a URL para a do seu workspace e defina `sql_warehouse_id` com o ID do seu SQL Warehouse. Não reutilize o ID já presente no repositório: ele pertence a outro ambiente. Não envie essas adaptações pessoais em um Pull Request.

### Caminho de execução recomendado

Faça a primeira rodada manualmente. Ela ensina as dependências do pipeline e deixa mais fácil localizar um erro do que executar o Job completo de uma vez.

1. Clone o repositório e execute os testes Python puros:

   ```bash
   python -m pip install "pytest>=8.0.0" "PyYAML>=6.0.0"
   PYTHONPATH=src python -m pytest -q
   ```

2. Baixe o [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Crie no seu workspace as tabelas Delta de entrada `workspace.bronze_layer.online_retail_i` e `workspace.bronze_layer.online_retail_ii`, com os dados das duas partes do dataset. Este repositório ainda não possui um notebook de ingestão Bronze: esta é uma preparação manual obrigatória para a primeira execução.

3. Importe ou abra os notebooks no workspace e execute-os nesta ordem:

   ```text
   00_setup
   01_transform_silver_layer
   03_create_modeling_base
   04_train_baseline
   05_train_model
   06_register_best_model
   ```

   O notebook `02_exploratory_analysis` é opcional. Os seis passos acima criam a camada Silver, a base de modelagem, os experimentos MLflow e, por fim, o modelo com alias `champion`. Não pule o último passo: o batch scoring depende desse alias.

4. Só depois que o treinamento manual tiver terminado, valide e publique o Bundle:

   ```bash
   databricks bundle validate -t dev
   databricks bundle deploy -t dev
   databricks bundle run reactivation_customers_pipeline -t dev
   ```

   Um Bundle é a definição versionada dos recursos do Databricks. Neste projeto, ele publica o Job e o dashboard; o Job executa a operação recorrente de geração sintética, scoring, monitoramento e explicabilidade. A CLI valida a configuração antes do deploy e executa o código usando a identidade autenticada. Veja o [guia oficial de Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/work-tasks).

### Se algo falhar

- **Tabela Bronze não encontrada:** volte ao passo 2. O notebook `01_transform_silver_layer` espera as duas tabelas Bronze já criadas.
- **Modelo ou alias `champion` não encontrado:** execute os notebooks de treino e `06_register_best_model` antes do Job.
- **Erro de Warehouse ou dashboard:** confira se o Warehouse existe, está iniciado e se o ID em `databricks.yml` é o seu. Como alternativa de estudo, execute os notebooks operacionais sem publicar o dashboard.
- **Quota de compute atingida:** aguarde a renovação da quota da Free Edition e retome do passo que falhou. As cotas são uma limitação normal desse ambiente, não um erro do código.

Depois da primeira execução, consulte a seção [Databricks Asset Bundle e execução](#databricks-asset-bundle-e-execução) para o fluxo operacional e de desenvolvimento.

## Objetivo de negócio

Em um cenário de CRM, não é eficiente abordar toda a base inativa da mesma forma. O projeto gera um ranking de clientes elegíveis para que campanhas de reativação priorizem quem tem maior propensão de retorno.

> Dado o histórico disponível de um cliente inativo até uma data de referência, qual a probabilidade de ele voltar a comprar nos 30 dias seguintes?

## Contexto, dados e limitações

O dataset de referência é o [Online Retail II, da UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Ele representa transações de varejo e é usado como proxy para um caso de reativação de clientes/apostadores.

Por isso, o projeto não mede resultado real de campanha, custo de incentivo, canal de comunicação ou margem por cliente. Dados sintéticos são usados apenas para testar o comportamento operacional de scoring, monitoramento e idempotência; eles não comprovam performance de negócio.

## Arquitetura e fluxo

```mermaid
flowchart LR
    A[Online Retail II] --> B[Bronze]
    B --> C[Silver: transações válidas]
    C --> D[Gold: base de modelagem e features]
    D --> E[Split temporal e treinamento]
    E --> F[MLflow Tracking e Model Registry]
    F --> G[Alias champion]
    C --> H[Dados sintéticos recorrentes]
    H --> I[Batch scoring]
    G --> I
    I --> J[Monitoramento]
    J --> K[Explicabilidade]
    K --> L[Dashboard Databricks]
```

O Job operacional executa, nesta ordem:

```text
generate_synthetic_transactions
→ batch_scoring
→ monitoring
→ explainability
→ refresh_reactivation_dashboard
```

O Job tem `max_concurrent_runs: 1`, retries nas tasks de notebook e schedule pausado. Assim, a execução final é manual e controlada.

## Contrato de modelagem e prevenção de leakage

- **Unidade de análise:** `customer_id + reference_date`.
- **Elegibilidade:** `inactive_days >= 60`.
- **Janela de observação:** 180 dias até a `reference_date`, inclusive.
- **Target:** `target_reactivated_30d = 1` quando há ao menos uma compra nos 30 dias posteriores à `reference_date`.
- **Validação:** treino, validação e teste respeitam a ordem temporal; não há split aleatório como estratégia principal.

Features usam somente informações disponíveis no momento da decisão. Compras da janela de resposta não são usadas como feature, evitando leakage temporal. O contrato completo está em [docs/problem_definition.md](docs/problem_definition.md).

## Features, modelos e avaliação

As features operacionais incluem recência (`inactive_days`), número de pedidos, itens, gasto, diversidade de produtos e países, tempo de relacionamento e métricas médias de ticket/valor. O repositório contém:

- uma baseline de ranking;
- Logistic Regression;
- XGBoost;
- otimização de hiperparâmetros com Optuna;
- avaliação com ROC-AUC, PR-AUC, precision, recall, F1, Precision@K e Lift@K.

O modelo vencedor é registrado no MLflow Model Registry e consumido pelo alias `champion` no batch scoring. Métricas e a versão final do modelo devem ser preenchidas a partir da execução final no Workspace; não há valores inventados neste README.

## Scoring, monitoramento e explicabilidade

O batch scoring recalcula as features da população elegível, gera `reactivation_score`, ranking e grupos de prioridade `TOP_10`, `TOP_20`, `TOP_30` e `REMAINDER`. A persistência é controlada por `scoring_date` para que uma reexecução da mesma rodada substitua o resultado correspondente em vez de duplicá-lo.

O monitoramento cobre qualidade de dados, volume, distribuição de scores, drift de features e scores por PSI, além de métricas de performance após a maturação da target. A explicabilidade usa contribuições nativas do XGBoost (`pred_contribs=True`) para gerar visões globais e locais.

O dashboard nativo do Databricks é versionado em [`dashboards/Reactivation Model.lvdash.json`](dashboards/Reactivation%20Model.lvdash.json) e atualizado como última task do Job.

## Databricks Asset Bundle e execução

O Bundle possui somente o target `dev`, em modo `development`. O contrato operacional confirmado no Workspace usa `workspace.silver_layer`, `workspace.synthetic_layer`, `workspace.gold_layer`, `workspace.monitoring_layer` e `workspace.explainability_layer`; o modelo registrado é `workspace.default.reactivation_customers_model`, consumido pelo alias `champion`.

```bash
# Dependências para testes locais
python -m pip install "pytest>=8.0.0" "PyYAML>=6.0.0"

# Testes Python puros
PYTHONPATH=src python -m pytest -q

# Validação e deploy no Databricks (requer autenticação válida)
databricks bundle validate -t dev
databricks bundle deploy -t dev

# Execução manual controlada do Job
databricks bundle run reactivation_customers_pipeline -t dev
```

Após a execução, valide status de todas as tasks, `run_id`, parâmetros, lote sintético, tabelas de scoring, monitoramento e explicabilidade, alias `champion`, atualização do dashboard e idempotência da mesma `scoring_date`.

## CI/CD e estratégia de branches

O workflow em [`.github/workflows/databricks-bundle.yml`](.github/workflows/databricks-bundle.yml) executa testes unitários e `databricks bundle validate --target dev` em pull requests para `dev` e `main`. O deploy ocorre somente em push para `dev`; o Job não é executado automaticamente pelo CI.

Antes da release, `dev` deve ser sincronizada com `main`, pois é a branch usada para deploy. O fluxo final esperado é validar a mudança em `dev`, fazer o deploy final, promover `dev` para `main` e então criar a tag `v1.0.0`.

## Estrutura do repositório

```text
conf/                     Configuração por ambiente
dashboards/               Dashboard Databricks versionado
docs/                     Contrato de negócio e documentação
notebooks/                Exploração, treinamento e operação do pipeline
resources/                Definição do Job no Bundle
src/reactivation_model/   Helpers reutilizáveis de configuração e qualidade
tests/                    Testes unitários locais
```

## Evidências de portfólio

As evidências devem ser geradas a partir da rodada final e armazenadas em `docs/assets/portfolio/`, removendo IDs de clientes, URLs privadas, tokens e qualquer outro dado sensível. O conjunto máximo recomendado é:

1. arquitetura ou fluxo do projeto;
2. execução verde do Job;
3. experimento e métricas no MLflow;
4. modelo com alias `champion`;
5. dashboard atualizado;
6. GitHub Actions verde.

## Limitações e trabalhos futuros

O escopo da `v1.0.0` é educacional e batch. Itens como ambiente de produção, Model Serving, Feature Store, alertas complexos, promoção automática de modelos e avaliação com dados reais de campanha são evoluções futuras e não bloqueiam o encerramento desta versão.

## Tecnologias

- Python, PySpark e Delta Lake
- Databricks, Databricks Asset Bundles e Databricks Jobs
- MLflow, scikit-learn, XGBoost e Optuna
- GitHub e GitHub Actions
