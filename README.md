Projeto Individual (PI1) — Predição de Sobrevivência no Titanic
Este projeto aplica técnicas de Machine Learning supervisionado para prever se um passageiro do navio Titanic sobreviveu ao desastre, utilizando o dataset original disponibilizado pela competição “Titanic — Machine Learning from Disaster” do Kaggle.

O objetivo é construir um pipeline completo de machine learning, envolvendo exploração dos dados, tratamento, engenharia de atributos, treinamento de modelos e avaliação.

Seções do Projeto

Exploração de Dados (EDA)
• Visualização da distribuição de idade
• Análise de sobrevivência por sexo e classe
• Mapa de correlação
• Análise inicial de variáveis faltantes e insights gerais

Pré-processamento
• Remoção de colunas irrelevantes (Cabin, Name, Ticket, PassengerId)
• Preenchimento de valores nulos (Age, Fare, Embarked)
• Codificação de variáveis categóricas
• Separação em treino e teste

Engenharia de Atributos
• Criação de FamilySize
• Feature IsAlone (viaja sozinho)
• Extração e padronização de Titles (Mr, Miss, Mrs, Master, Rare)

Modelos Treinados
• Regressão Logística
• Random Forest
Ambos integrados a um pipeline com padronização numérica e OneHotEncoding para categorias.

Avaliação de Modelos
• Matriz de confusão
• Curva ROC e cálculo do AUC
• Curva Precision-Recall
• Relatório de classificação
• Importância das features (Random Forest)

Estrutura do Projeto

titanic-pi1/
├── data/
│ └── train.csv
├── notebooks/
│ └── 01_exploracao_etl.ipynb
├── src/
│ ├── data_preprocessing.py
│ ├── features.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ └── visualization.py
├── models/
│ ├── logistic_regression.joblib
│ └── random_forest.joblib
├── main.py
├── README.md
└── requirements.txt

Como Executar

Instale as dependências:
pip install -r requirements.txt

Execute o pipeline completo:
python main.py

Para rodar as análises e visualizações do notebook:
Abra o arquivo notebooks/01_exploracao_etl.ipynb no Jupyter Lab ou Jupyter Notebook.

Sobre as Técnicas Utilizadas
• Pandas e NumPy para preparação de dados
• Visualizações com Seaborn e Matplotlib
• Modelagem com Scikit-Learn
• Pipelines para padronizar todo o fluxo de transformação + modelo
• Visualização de importância das features
• Avaliação completa de métricas de classificação

Resultado Final
O projeto treina dois modelos, avalia suas curvas e métricas e identifica quais variáveis mais influenciam na sobrevivência dos passageiros. Ele representa um pipeline realista e bem estruturado de machine learning aplicado a um dataset clássico.