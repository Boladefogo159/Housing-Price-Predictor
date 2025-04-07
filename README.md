# House Prices - Advanced Regression Techniques (Kaggle)

Este projeto tem como objetivo prever os preços de venda de imóveis residenciais com base em um conjunto de dados de Ames, Iowa — uma competição clássica do Kaggle focada em regressão e pré-processamento de dados.
Além disso, aprender mais sobre regressão, pipeline e principalmente EDA em cima de um assunto que tenho pouca familiaridade.
---

## Objetivo

Desenvolver um pipeline robusto e interpretável para prever o valor de venda (`SalePrice`) de imóveis utilizando:

- Engenharia de features inteligente
- Modelagem preditiva com validação cruzada
- Otimização de hiperparâmetros
- Tratamento de heterocedasticidade
- Submissão no Kaggle com resultados competitivos

---

## Abordagem

### 1. Pré-processamento
- Conversão de `SalePrice` para escala logarítmica com `np.log1p()` 
> Obs: Para tratar assimetria, uma dica que está no tutorial do projeto.
- Remoção e imputação de valores ausentes com `SimpleImputer`. 
> Obs: Como Guardrail para encoders diferentes que apareceram no test.csv

### 2. Engenharia de Features (`FeatureEngineer`)
Criação de variáveis para capturar padrões de valor em casas mais caras, como:

- `TotalBathrooms`: Soma ponderada dos banheiros.
- `TotalSF`: Área total útil (porão + andares superiores).
- `LuxoExterno`: Presença de itens como piscina, deck, alpendre grande.
- `OverallQualCond`: Produto da qualidade geral com a condição.
- `YearsSinceRemodel`: Tempo desde a última reforma.
 > Obs: Foi a etapa que levou mais tempo, testei outras diversas variáveis mas que não apresentaram correlação direta suficiente por meio de testes de seleção automática de features
### 3. Pipelines com Scikit-Learn 
Uso de `ColumnTransformer` com:

- `StandardScaler` para variáveis numéricas
- `OrdinalEncoder` para ordinais
- `OneHotEncoder` para variáveis nominais
> Obs: Biblioteca maravilhosa.
---

## Modelos testados

| Modelo               | RMSE log (validação) | Público Kaggle | Comentário |
|----------------------|----------------------|----------------|------------|
| Ridge Regression     | ~0.2292              | N/A            | Modelo base linear regularizado |
| Random Forest (básico) | ~0.1875            | ~0.180         | Melhora inicial com não-linearidade |
| Random Forest + tuning + features | **~0.147** | **0.14766**   | Modelo final submetido |

> Obs: Modelos foram treinados com `GridSearchCV`, otimizando hiperparâmetros como `n_estimators` e `max_depth`, busquei otimizar os demais hiperparâmetros mas andei de lado com os resultados.

---

## Resultados (Kaggle Submission)

- Erro absoluto médio: **~R$ 8.115,09**
- Erro percentual médio: **~4.57%**
- Score público no Kaggle: **0.14766**

Segundo o GPT esse modelo representa: **top 15-25% da competição**, mesmo sem uso de ensemble ou boosting.

---

## Estratégias testadas

- `sample_weight`: Testei dar mais peso a casas com valor acima de R$ 300 mil para tratar heterocedasticidade mas aumentei o erro das casas com valores menores e acabei andando de lado.

---

## Arquivos do Projeto

- `main.ipynb` – Notebook com todo o pipeline de modelagem.
- `submission.csv` – Submissão gerada para o Kaggle.
- `README.md` – Documentação e storytelling do projeto.
- `stepbystep.ipynb` - Demonstra o início do EDA.
---

## Sugestões de melhoria futura

- Experimentar `HistGradientBoostingRegressor`, `XGBoost` ou `LightGBM`


## Principais desafios
- EDA e seleção de features, dataset robusto e amplo.
- Heterocedasticidade.
- Montar um modelo utilizando pipeline 📋

---

📎 [LinkedIn](https://www.linkedin.com/in/jorgepereira-/)

---
