# 🏠 House Prices - Advanced Regression Techniques (Kaggle)

Este projeto tem como objetivo prever os preços de venda de imóveis residenciais com base em um conjunto de dados de Ames, Iowa — uma competição clássica do Kaggle focada em regressão e pré-processamento de dados.

---

## 📌 Objetivo

Desenvolver um pipeline robusto e interpretável para prever o valor de venda (`SalePrice`) de imóveis utilizando:

- Engenharia de features inteligente
- Modelagem preditiva com validação cruzada
- Otimização de hiperparâmetros
- Tratamento de heterocedasticidade
- Submissão no Kaggle com resultados competitivos

---

## 🧠 Abordagem

### 1. Pré-processamento
- Conversão de `SalePrice` para escala logarítmica com `np.log1p()` para tratar assimetria.
- Remoção e imputação de valores ausentes com `SimpleImputer`.

### 2. Engenharia de Features (`FeatureEngineer`)
Criação de variáveis para capturar padrões de valor em casas mais caras, como:

- `TotalBathrooms`: Soma ponderada dos banheiros.
- `TotalSF`: Área total útil (porão + andares superiores).
- `LuxoExterno`: Presença de itens como piscina, deck, alpendre grande.
- `OverallQualCond`: Produto da qualidade geral com a condição.
- `YearsSinceRemodel`: Tempo desde a última reforma.

### 3. Pipelines com Scikit-Learn
Uso de `ColumnTransformer` com:

- `StandardScaler` para variáveis numéricas
- `OrdinalEncoder` para ordinais
- `OneHotEncoder` para variáveis nominais

---

## 🔍 Modelos testados

| Modelo               | RMSE log (validação) | Público Kaggle | Comentário |
|----------------------|----------------------|----------------|------------|
| Ridge Regression     | ~0.2292              | N/A            | Modelo base linear regularizado |
| Random Forest (básico) | ~0.1875            | ~0.180         | Melhora inicial com não-linearidade |
| Random Forest + tuning + features | **~0.147** | **0.14766**   | Modelo final submetido |

> Obs: Modelos foram treinados com `GridSearchCV`, otimizando hiperparâmetros como `n_estimators` e `max_depth`, busquei otimizar os demais hiperparâmetros mas andei de lado com os resultados.

---

## 📈 Resultados (Kaggle Submission)

- ✅ Erro absoluto médio: **~R$ 8.115,09**
- ✅ Erro percentual médio: **~4.57%**
- ✅ Score público no Kaggle: **0.14766**

Isso representa um modelo competitivo, com desempenho dentro do **top 15-25% da competição**, mesmo sem uso de ensemble ou boosting.

---

## ⚖️ Estratégias de Robustez

- `sample_weight`: Testei dar mais peso a casas com valor acima de R$ 300 mil para tratar heterocedasticidade.
- `handle_unknown='ignore'`: Tratei diversas falhas do encoder ao lidar com categorias novas no `test.csv`.
> Obs: Ainda assim na submissão final há algum problema com encoder na column 0
---

## 📦 Arquivos do Projeto

- `main.ipynb` – Notebook com todo o pipeline de modelagem.
- `submission.csv` – Submissão gerada para o Kaggle.
- `FeatureEngineer()` – Classe customizada para engenharia de variáveis.
- `README.md` – Documentação e storytelling do projeto.
- `stepbystep.ipynb` - Demonstra o início do EDA.
---

## 🚀 Melhorias possíveis

- Experimentar `HistGradientBoostingRegressor`, `XGBoost` ou `LightGBM`
- Implementar Stacking com Ridge, Random Forest e GBM
- Analisar erros residuais por região (`Neighborhood`) e outliers
- Avaliar uso de `Quantile Regression` para prever faixas de preços

## 🥶 Principais desafios
- EDA e seleção de features, dataset robusto e amplo.
- Heterocedasticidade.
- Montar um modelo utilizando pipeline 📋

---

## 🧑‍💻 Autor

Projeto conduzido por Jorge Silva Pereira, analista de dados e estudante de cientista de dados, engenharia de atributos e aplicação prática de soluções de machine learning.

📎 [LinkedIn](https://www.linkedin.com/in/jorgepereira-/)

---
