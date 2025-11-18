# 🛳️ Predição de Sobrevivência no Titanic — Projeto de Machine Learning (PI1)

Este projeto aplica técnicas de **Aprendizado de Máquina supervisionado** para prever se um passageiro do Titanic sobreviveu ao desastre, utilizando dados reais disponibilizados pela competição *Titanic - Machine Learning from Disaster* da Kaggle.

O objetivo é construir um pipeline completo de machine learning, incluindo:

- Exploração de dados (EDA)
- Limpeza e transformação
- Engenharia de atributos
- Treinamento de modelos (Logistic Regression e Random Forest)
- Avaliação com métricas, gráficos e análise de importância das features

---

## 📁 **Estrutura do Projeto**

titanic-pi1/
│── data/
│ └── train.csv
│
│── models/
│ ├── logistic_regression.joblib
│ └── random_forest.joblib
│
│── notebooks/
│ └── 01_exploracao_etl.ipynb
│
│── src/
│ ├── data_preprocessing.py
│ ├── evaluate.py
│ ├── features.py
│ ├── models.py
│ ├── train.py
│ └── visualization.py
│
│── main.py
│── requirements.txt
│── README.md
└── .gitignore

markdown
Copiar código

---

## 🧠 **Técnicas e Modelos Utilizados**

### 🔹 Pré-processamento
- Remoção de colunas inúteis ou com muitos valores ausentes
- Preenchimento de valores faltantes
- Label Encoding e One-Hot Encoding
- Normalização de variáveis numéricas

### 🔹 Engenharia de Atributos
- `FamilySize`
- `IsAlone`
- Extração e categorização de títulos do nome: *Mr, Miss, Mrs, Master, Rare*

### 🔹 Modelos Treinados
- **Regressão Logística**
- **Random Forest**

### 🔹 Métricas de Avaliação
- Accuracy
- Classification Report
- Matriz de Confusão
- Curva ROC e AUC
- Precision-Recall Curve
- Importância das Features

---

## 🚀 **Como Executar o Projeto**

### 1️⃣ **Instale as dependências**

```bash
pip install -r requirements.txt
2️⃣ Garanta que o arquivo train.csv esteja em:
bash
Copiar código
data/train.csv
3️⃣ Execute o pipeline completo
bash
Copiar código
python main.py
Ao final, você verá:

Métricas dos modelos

Gráficos de avaliação

Importância das features

Modelos salvos na pasta /models

📊 Exploração Inicial (EDA)
O notebook 01_exploracao_etl.ipynb realiza:

Visualização de sobreviventes

Distribuição de idade

Correlação entre variáveis

Análises por sexo e classe

Principais observações:

Mulheres e crianças tiveram maior taxa de sobrevivência

Passageiros da 1ª classe sobreviveram muito mais

Cabin tem muitos valores ausentes → removido

Age e Embarked exigem tratamento

Fare e Pclass são fortemente relevantes

🧪 Resultados Esperados
Accuracy média: ~78% a 82%

Random Forest costuma ter desempenho superior

Features mais importantes:

Sex

Pclass

Fare

FamilySize

Title

📦 Dependências
Todas listadas em requirements.txt:

nginx
Copiar código
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyterlab
notebook
👨‍💻 Como o Código Funciona
🔸 main.py
Executa todo o pipeline:

Carrega os dados

Faz limpeza e encoding

Divide treino/teste

Cria pipeline de preprocessamento

Treina Logistic Regression e Random Forest

Avalia ambos

Plota gráficos

Exibe importância das features

🔸 src/train.py
Contém a lógica principal de treinamento e avaliação.

🔸 src/models.py
Cria o pipeline de transformação + classificador.

🔸 src/data_preprocessing.py
Toda a limpeza e preparação dos dados.

🔸 src/evaluate.py
Métricas e gráficos de desempenho.

🔸 src/visualization.py
Gráficos do EDA e importância das features.

🏁 Conclusão
Este projeto demonstra de ponta a ponta como estruturar um pipeline robusto de Machine Learning, desde a análise exploratória até a avaliação final dos modelos.

Serve como base para trabalhos acadêmicos, portfólio e aprendizado prático de ML.