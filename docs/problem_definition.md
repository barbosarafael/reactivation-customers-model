# Problem Definition

## 1. Objetivo do projeto

O objetivo deste projeto é prever se um cliente inativo voltará a comprar nos próximos 30 dias.

O projeto simula um caso de reativação de clientes/apostadores, usando o dataset **Online Retail II** como base transacional. A intenção é treinar um fluxo completo de ciência de dados no Databricks, passando por ingestão, transformação, feature engineering, modelagem, scoring, automação e monitoramento.

---

## 2. Contexto de negócio

Em negócios transacionais, como varejo, apostas, assinaturas ou serviços digitais, é comum existir uma base de clientes que compraram ou transacionaram no passado, mas deixaram de interagir recentemente.

A ideia deste projeto é identificar, dentro do grupo de clientes inativos, quais têm maior probabilidade de voltar a comprar em breve.

Essa previsão poderia ser usada para priorizar ações como:

- campanhas de CRM;
- ofertas de reativação;
- réguas de comunicação;
- segmentação de clientes;
- priorização de públicos para contato.

---

## 3. Problema de modelagem

O problema será tratado como uma tarefa de classificação binária.

Para cada cliente inativo em uma determinada data de referência, queremos prever se ele voltará a comprar nos próximos 30 dias.

A pergunta principal é:

> Dado o histórico de comportamento de um cliente até uma data de referência, qual a probabilidade de ele voltar a comprar nos próximos 30 dias?

---

## 4. Unidade de análise

A unidade de análise será:

```text
customer_id + reference_date
```

Cada linha da base de modelagem representa um cliente em uma determinada data de referência.

Exemplo:

| customer_id | reference_date | target_reactivated_30d |
|---|---:|---:|
| 12345 | 2011-06-01 | 1 |
| 67890 | 2011-06-01 | 0 |
| 12345 | 2011-07-01 | 0 |

Isso significa que o mesmo cliente pode aparecer mais de uma vez na base, desde que em diferentes datas de referência.

---

## 5. Definição de cliente inativo

Um cliente será considerado inativo quando estiver há pelo menos 60 dias sem comprar antes da data de referência.

```text
inactive_days >= 60
```

Onde:

```text
inactive_days = reference_date - last_purchase_date
```

Exemplo:

| customer_id | reference_date | last_purchase_date | inactive_days | is_inactive |
|---|---:|---:|---:|---:|
| 12345 | 2011-06-01 | 2011-03-20 | 73 | 1 |
| 67890 | 2011-06-01 | 2011-05-10 | 22 | 0 |

Essa é uma definição inicial e poderá ser revisada após a análise exploratória.

---

## 6. Data de referência

A `reference_date` representa o momento em que simulamos a decisão de negócio.

Em produção, seria a data em que o modelo roda para decidir quais clientes devem ser priorizados em uma campanha.

Exemplo:

```text
reference_date = 2011-06-01
```

Nesse cenário, o modelo deve usar apenas informações disponíveis até `2011-06-01` para prever se o cliente comprará entre `2011-06-02` e `2011-07-01`.

---

## 7. Janela de observação

A janela de observação é o período histórico usado para construir as features do cliente.

Definição inicial:

```text
180 dias antes da reference_date
```

Exemplo:

```text
reference_date = 2011-06-01
observation_window = 2010-12-03 até 2011-06-01
```

Possíveis features calculadas nessa janela:

- dias desde a última compra;
- quantidade de compras;
- valor total comprado;
- ticket médio;
- quantidade total de itens;
- quantidade de produtos distintos;
- quantidade de dias ativos;
- dias desde a primeira compra;
- frequência média de compra;
- variação de gasto recente versus histórico.

Nenhuma informação posterior à `reference_date` poderá ser usada nas features.

---

## 8. Janela de predição

A janela de predição será de 30 dias após a data de referência.

```text
prediction_window = reference_date + 30 dias
```

Exemplo:

```text
reference_date = 2011-06-01
prediction_window = 2011-06-02 até 2011-07-01
```

Essa janela será usada apenas para calcular a variável resposta.

---

## 9. Target

A variável alvo será:

```text
target_reactivated_30d
```

Definição:

| Valor | Significado |
|---:|---|
| 1 | O cliente realizou pelo menos uma compra nos 30 dias após a `reference_date`. |
| 0 | O cliente não realizou compra nos 30 dias após a `reference_date`. |

Exemplo:

| customer_id | reference_date | compra nos próximos 30 dias? | target_reactivated_30d |
|---|---:|---|---:|
| 12345 | 2011-06-01 | Sim | 1 |
| 67890 | 2011-06-01 | Não | 0 |

---

## 10. Regras para evitar vazamento temporal

O projeto deve respeitar rigorosamente a separação entre passado e futuro.

As features devem usar somente dados disponíveis até a `reference_date`.

A target deve usar somente dados após a `reference_date`.

### Regras obrigatórias

- Não usar compras da janela de predição como feature.
- Não calcular métricas do cliente usando o dataset completo.
- Não calcular última compra olhando datas futuras.
- Não fazer split aleatório como estratégia principal.
- Não usar informação agregada que inclua eventos posteriores à data de referência.

### Exemplo de vazamento

Errado:

```text
feature_total_orders = total de compras do cliente em todo o dataset
```

Certo:

```text
feature_total_orders = total de compras do cliente até a reference_date
```

---

## 11. Estratégia de validação

A separação entre treino, validação e teste deve respeitar a ordem temporal.

Estratégia inicial:

| Conjunto | Período |
|---|---|
| Treino | Datas de referência mais antigas |
| Validação | Datas de referência intermediárias |
| Teste | Datas de referência mais recentes |

Não será usado split aleatório como estratégia principal.

Essa abordagem simula melhor o cenário real, em que o modelo é treinado com dados históricos e aplicado em períodos futuros.

---

## 12. Baseline

Antes de treinar modelos de machine learning, será criado um baseline simples.

Possíveis baselines:

- priorizar clientes com menor recência;
- priorizar clientes com maior frequência histórica;
- priorizar clientes com maior valor monetário histórico;
- priorizar clientes com maior ticket médio.

O modelo final precisa superar esse baseline para justificar sua complexidade.

---

## 13. Métricas de avaliação

Como o problema simula uma campanha de reativação, a avaliação não deve depender apenas de acurácia.

Métricas técnicas:

- ROC AUC;
- PR AUC;
- Precision;
- Recall;
- F1-score.

Métricas de negócio:

- Precision no top 10%;
- Precision no top 20%;
- Lift no top 10%;
- Lift no top 20%;
- taxa de reativação esperada por faixa de score.

As métricas mais importantes para decisão serão:

```text
Precision no topo da base
Lift no top 10%
Lift no top 20%
```

Isso porque, em uma campanha real, normalmente não acionamos todos os clientes, mas sim os clientes com maior propensão de retorno.

---

## 14. Saída esperada do modelo

O modelo deverá gerar uma tabela de scoring com os clientes inativos e sua probabilidade estimada de reativação.

Estrutura esperada:

| Coluna | Descrição |
|---|---|
| customer_id | Identificador do cliente |
| scoring_date | Data em que o score foi gerado |
| reactivation_score | Probabilidade estimada de reativação |
| score_rank | Ranking do cliente pelo score |
| priority_group | Grupo de prioridade para ação |
| model_version | Versão do modelo utilizado |

Exemplo de grupos de prioridade:

| Grupo | Regra inicial |
|---|---|
| Alta prioridade | Top 10% dos scores |
| Média prioridade | Entre top 10% e top 30% |
| Baixa prioridade | Demais clientes elegíveis |

---

## 15. Uso esperado da previsão

A previsão pode ser usada para simular uma campanha de CRM.

Exemplo de fluxo:

1. Identificar clientes inativos.
2. Gerar features históricas.
3. Aplicar o modelo treinado.
4. Ordenar clientes por score de reativação.
5. Selecionar o top 10% ou top 20%.
6. Simular uma campanha de reativação.
7. Avaliar quantos clientes voltaram a comprar nos próximos 30 dias.

---

## 16. Limitações iniciais

O dataset **Online Retail II** representa compras de varejo, não apostas.

Portanto, neste projeto, o comportamento de compra será usado como aproximação para comportamento transacional de apostadores.

Essa adaptação é aceitável para fins de estudo, mas deve ser documentada como uma limitação do projeto.

Outras limitações esperadas:

- ausência de dados reais de campanha;
- ausência de canal de comunicação com cliente;
- ausência de custo de incentivo;
- ausência de margem financeira por cliente;
- possível sazonalidade específica do varejo;
- possível diferença entre comportamento de compra e comportamento de aposta.

---

## 17. Decisões iniciais

| Decisão | Valor inicial |
|---|---|
| Problema | Reativação de clientes inativos |
| Tipo de problema | Classificação binária |
| Unidade de análise | `customer_id + reference_date` |
| Cliente inativo | Sem compra há pelo menos 60 dias |
| Janela de observação | 180 dias antes da `reference_date` |
| Janela de predição | 30 dias após a `reference_date` |
| Target | Voltou a comprar nos próximos 30 dias |
| Split | Temporal |
| Baseline | Regras simples de recência, frequência e valor |
| Métricas técnicas | ROC AUC, PR AUC, Precision, Recall, F1 |
| Métricas de negócio | Precision e Lift no topo da base |

---

## 18. Próximos passos

Após esta definição, os próximos passos do projeto serão:

1. Carregar o dataset bruto na camada Bronze.
2. Criar a camada Silver com dados limpos.
3. Construir a base de clientes elegíveis por `reference_date`.
4. Criar a target `target_reactivated_30d`.
5. Criar as features históricas.
6. Montar a base de modelagem.
7. Treinar e avaliar modelos.
8. Criar batch scoring.
9. Automatizar o pipeline no Databricks.
10. Monitorar saúde do job, dados e modelo.