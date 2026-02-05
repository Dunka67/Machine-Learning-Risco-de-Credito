# Análise de Risco de Crédito com Machine Learning

## Descrição do Projeto
Este projeto tem como objetivo aplicar algoritmos de classificação para prever a inadimplência em solicitações de crédito. O foco principal foi estudar o comportamento de diferentes modelos de Machine Learning e entender quais variáveis (como renda, idade ou histórico) mais influenciam na decisão de aprovar ou negar um empréstimo.

Utilizei uma base de dados pública com históricos financeiros reais, simulando um cenário de triagem bancária.

## Etapas do Desenvolvimento

### 1. Tratamento de Dados
A base original continha ruídos que impediriam o aprendizado do modelo. O trabalho de limpeza envolveu:
- Padronização de textos (remoção de acentos e espaços extras).
- Tratamento de valores nulos e inconsistentes.
- Transformação de variáveis categóricas em numéricas (Encoding), permitindo que os algoritmos processassem colunas como "Motivo do Empréstimo" e "Grau de Risco".

### 2. Comparativo de Modelos
Para garantir que o resultado não fosse enviesado, testei e comparei três algoritmos diferentes:
- **Random Forest Classifier:** Para testar a eficácia de árvores de decisão.
- **K-Nearest Neighbors (KNN):** Para comparação com um modelo baseado em distância.
- **XGBoost:** Para testar a performance de Gradient Boosting.

### 3. Resultados
Os dados foram divididos em treino e teste para validar a precisão. O modelo **XGBoost** apresentou a melhor performance, superando os demais na métrica de acurácia.

- **XGBoost:** 93.73%
- **Random Forest:** 93.30%
- **KNN:** 83.85%

### 4. Análise de Fatores (Feature Importance)
Além da previsão, o projeto identificou o peso de cada variável. A análise mostrou que a relação entre **Dívida/Renda** e o **Histórico de Crédito** são fatores mais decisivos para o risco do que apenas o valor do salário.

## Tecnologias
- Python
- Pandas (Manipulação de dados)
- Scikit-Learn & XGBoost (Modelagem)